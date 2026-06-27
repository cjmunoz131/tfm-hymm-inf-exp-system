import boto3
import time
import os
import uuid
from datetime import datetime, timezone
import logging
from botocore.exceptions import ClientError

dynamodb = boto3.client('dynamodb') # Usamos el cliente para transacciones
TABLE_NAME = os.getenv('DDB_TABLE_NAME', 'hymmrec-pipelines-executions-history')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    pipeline_id = event['pipeline_id']
    now = datetime.now(timezone.utc)
    execution_time = int(now.strftime("%Y%m%d%H%M%S"))
    correlation_id = event.get('execution_id', str(uuid.uuid4()))
    logger.info(f"Validando y registrando la ejecución para pipeline_id: {pipeline_id} con correlation_id: {correlation_id}")
    # Llave del ítem de control (Lock)
    # Usamos una SK fija para que sea el punto único de verdad sobre el estado actual
    lock_key = {
        'pipeline_id': {'S': pipeline_id},
        'execution_time': {'N': '0'} # 0 representará siempre el estado actual/lock
    }

    try:
        # OPERACIÓN TRANSACCIONAL ÚNICA
        dynamodb.transact_write_items(
            TransactItems=[
                {
                    "Update": {
                        "TableName": TABLE_NAME,
                        "Key": lock_key,
                        "ConditionExpression": "attribute_not_exists(#s) OR #s <> :ip",
                        "UpdateExpression": "SET #s = :ip, last_exec_id = :cid",
                        "ExpressionAttributeNames": {"#s": "status"},
                        "ExpressionAttributeValues": {
                            ":ip": {"S": "IN_PROGRESS"},
                            ":cid": {"S": correlation_id}
                        }
                    }
                },
                {
                    "Put": {
                        "TableName": TABLE_NAME,
                        "Item": {
                            "pipeline_id": {"S": pipeline_id},
                            "execution_time": {"N": str(execution_time)},
                            "status": {"S": "IN_PROGRESS"},
                            "correlation-id": {"S": correlation_id},
                            "domain": {"S": event['domain']},
                            "project": {"S": event['project']},
                            "domain_project_sk": {"S": f"{event['domain']}#{event['project']}"},
                            "start_time": {"S": now.isoformat()},
                            "orchestration_level": {"S": event.get('orchestration_level', 'PROCESS')},
                        }
                    }
                }
            ]
        )
        logger.info(f"Lock adquirido y registro creado para pipeline_id: {pipeline_id} con correlation_id: {correlation_id}")
        return {
            "statusCode": 201,
            "body": {
                "status": "IN_PROGRESS",
                "execution_time": str(execution_time),
                "message": "Transacción exitosa: Lock adquirido y registro creado."
            }
        }

    except ClientError as e:
        if e.response['Error']['Code'] == 'TransactionCanceledException':
            # Si la condición del lock falla, la transacción se cancela entera
            logger.warning(f"Ejecución omitida: {pipeline_id} ya se encuentra en curso.")
            return {
                "statusCode": 409,
                "body": {
                    "status": "SKIPPED",
                    "execution_time": str(execution_time),
                    "message": "Transacción No Iniciada: El pipeline ya está IN_PROGRESS o bloqueado."
                }
            }
        raise e