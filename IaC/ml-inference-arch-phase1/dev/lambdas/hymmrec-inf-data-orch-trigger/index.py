import json
import boto3
import os
import uuid
from datetime import datetime, timezone
import logging


# ---------- AWS ----------
ssm = boto3.client('ssm')
stepfunctions = boto3.client('stepfunctions')

PARAMETER_NAME = os.environ.get('EXECUTION_PLAN_MANIFEST', '/hymmrec/config/master_manifest_template')
STATE_MACHINE_ARN = os.environ.get('STATE_MACHINE_ARN')
ID_DOMAIN_PARAMETER = os.environ.get('PIPELINE_CONFIG_ID', '01')
INFERENCE_PIPELINE_ID = os.environ.get('INFERENCE_PIPELINE_ID', 'INFERENCE-001')
DOMAIN_CONTEXT_PARAMETER = os.environ.get('DOMAIN_CONTEXT_PARAMETER', '/hymmrec/config/domain-context')

_ssm_cache = {}
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def _validate_manifest_structure(manifest):
    """
    Valida la jerarquía del árbol: Plan -> Processes -> Task Groups.
    Retorna True si es válido, levanta excepción si no.
    """
    if "workflow_execution_plan" not in manifest:
        raise ValueError("Falta el nodo raíz 'workflow_execution_plan'")
    
    plan = manifest["workflow_execution_plan"]
    
    # Validar que existan procesos y que sea una lista
    if "processes" not in plan or not isinstance(plan["processes"], list):
        raise ValueError("El nodo 'processes' debe ser una lista activa.")
    
    # Validar cada proceso en el árbol
    for i, process in enumerate(plan["processes"]):
        if "process_cod" not in process:
            raise ValueError(f"Proceso en el índice {i} no tiene 'process_cod'")
        
        # Validar que existan task_groups (puede ser lista o el string 'all')
        if "task_groups" not in process:
            raise ValueError(f"El proceso {process['process_cod']} no tiene definido 'task_groups'")
        
        tg = process["task_groups"]
        if not (isinstance(tg, list) or tg == "all"):
            raise ValueError(f"En el proceso {process['process_cod']}, 'task_groups' debe ser 'all' o una lista")
            
    return True

def get_ssm_json(param_name: str) -> dict:
    """
    Utilidad centralizada para obtener y parsear un parámetro JSON desde SSM.
    Incluye caché en memoria para evitar lecturas repetidas en la misma invocación.
    """
    if param_name in _ssm_cache:
        return _ssm_cache[param_name]
    try:
        logger.info(f"Leyendo parámetro de SSM: {param_name}")
        resp = ssm.get_parameter(Name=param_name, WithDecryption=True)
        data = json.loads(resp["Parameter"]["Value"])
        _ssm_cache[param_name] = data
        return data
    except json.JSONDecodeError as e:
        raise RuntimeError(f"El parámetro SSM '{param_name}' no contiene JSON válido: {e}")
    except Exception as e:
        raise RuntimeError(f"No se pudo leer el parámetro SSM '{param_name}': {e}")

def _get_pipeline_context_from_ssm(pipeline_config_id):
    """
    Obtiene el contexto de dominio y proyecto desde AWS Systems Manager Parameter Store.
    Lee el parámetro JSON de domain-context y extrae los valores correspondientes
    al pipeline_config_id proporcionado.
    
    Args:
        pipeline_config_id: Clave del contexto dentro del JSON (ej: '01')
    
    Returns:
        dict con keys: domain, project, pipeline_id
    """
    domain_context = get_ssm_json(DOMAIN_CONTEXT_PARAMETER)
    
    # Buscar el config_id dentro del JSON (la clave puede ser string o int)
    config_key = str(pipeline_config_id)
    
    if config_key not in domain_context:
        raise ValueError(
            f"No se encontró configuración para el ID '{config_key}' "
            f"en el parámetro SSM '{DOMAIN_CONTEXT_PARAMETER}'. "
            f"Claves disponibles: {list(domain_context.keys())}"
        )
    
    ctx = domain_context[config_key]
    domain = ctx['domain']
    project = ctx['project']
    
    # Construcción del pipeline_id según el estándar definido
    pipeline_id = f"DOM#{domain}#PROJ#{project}#ID#{INFERENCE_PIPELINE_ID}"
    
    logger.info(f"Contexto obtenido - Domain: {domain}, Project: {project}, Pipeline ID: {pipeline_id}")
    
    return {
        "domain": domain,
        "project": project,
        "pipeline_id": pipeline_id
    }

def lambda_handler(event, context):
    try:
        # 1. Obtener Manifest de SSM (reutilizando utilidad centralizada)
        manifest = get_ssm_json(PARAMETER_NAME)
        workflow_start_time = event.get('time', datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'))
        # 2. VALIDACIÓN DE ESTRUCTURA DE ÁRBOL
        print("Validando estructura jerárquica del manifest...")
        _validate_manifest_structure(manifest)
        print("Validación exitosa.")
        
        context_data = _get_pipeline_context_from_ssm(ID_DOMAIN_PARAMETER)
        
        pipeline_id = context_data['pipeline_id']
        domain = context_data['domain']
        project = context_data['project']
        # 3. Generación de ID y reemplazo
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        unique_suffix = str(uuid.uuid4())[:8]
        execution_id = f"EXEC-{timestamp}-{unique_suffix}"
        manifest['workflow_execution_plan']['workflow-start-time'] = workflow_start_time
        manifest['workflow_execution_plan']['execution_id'] = execution_id
        manifest['workflow_execution_plan']['domain'] = domain
        manifest['workflow_execution_plan']['project'] = project
        manifest['workflow_execution_plan']['pipeline_id'] = pipeline_id
        manifest['workflow_execution_plan']['domain_project_id'] = ID_DOMAIN_PARAMETER
        # 4. Ejecución
        response = stepfunctions.start_execution(
            stateMachineArn=STATE_MACHINE_ARN,
            name=execution_id,
            input=json.dumps(manifest)
        )
        # El SDK de boto3 devuelve metadatos en 'ResponseMetadata'
        http_status = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
        execution_arn = response.get('executionArn')

        if http_status == 200 and execution_arn:
            print(f"✅ Step Function iniciada exitosamente. ARN: {execution_arn}")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'execution_id': execution_id,
                    'execution_arn': execution_arn,
                    'status': 'SUCCESS',
                    'message': 'Master Pipeline orchestration triggered successfully'
                })
            }
        else:
            # Caso donde la respuesta no es 200 pero no lanzó excepción (raro, pero posible)
            raise Exception(f"Respuesta inesperada de AWS Step Functions: Status {http_status}")

    except ValueError as ve:
        error_msg = f"Error de esquema en Manifest: {str(ve)}"
        print(error_msg)
        return {'statusCode': 400, 'body': json.dumps({'error': error_msg})}
    except Exception as e:
        print(f"Critical Error: {str(e)}")
        return {'statusCode': 500, 'body': json.dumps({'error': 'Internal Server Error'})}