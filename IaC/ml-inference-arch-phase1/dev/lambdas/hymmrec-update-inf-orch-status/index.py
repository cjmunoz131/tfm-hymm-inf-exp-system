"""
Lambda: Update Master Orchestration Status
============================================
Actualiza el estado final de la ejecución del pipeline maestro en DynamoDB.
Responsabilidades:
  - Liberar el lock (status → COMPLETED o FAILED)
  - Registrar end_time en el historial
  - Si hay error (masterProcessError en el evento), registrar el detalle

Invocada por Step Functions en dos puntos:
  - UpdateMasterOrchestrationStatus (flujo exitoso → sin masterProcessError)
  - UpdateMasterOrchestrationStatusError (flujo con error → con masterProcessError)
"""

import boto3
import os
import json
import logging
from datetime import datetime, timezone
from botocore.exceptions import ClientError

dynamodb = boto3.client('dynamodb')
TABLE_NAME = os.getenv('DDB_TABLE_NAME', 'hymmrec-pipelines-executions-history')

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def determine_final_status(event: dict) -> str:
    """
    Determina el estado final basado en la presencia de masterProcessError.
    Si existe el nodo masterProcessError → FAILED, sino → SUCCESS.
    """
    error_info = event.get('masterProcessError')
    if error_info and (error_info.get('Error') or error_info.get('Cause')):
        return 'FAILED'
    return 'SUCCESS'


def lambda_handler(event, context):
    logger.info(f"Event recibido: {json.dumps(event, default=str)}")

    # Validación mínima
    if 'pipeline_id' not in event or 'registerOrchestrationStatus' not in event:
        raise ValueError("Event debe contener 'pipeline_id' y 'registerOrchestrationStatus'")

    pipeline_id = event['pipeline_id']
    correlation_id = event.get('execution_id', 'UNKNOWN')
    execution_time = str(event['registerOrchestrationStatus']['execution_time'])

    # Determinar estado final
    final_status = determine_final_status(event)
    lock_status = 'COMPLETED' if final_status == 'SUCCESS' else 'FAILED'
    error_info = event.get('masterProcessError', {})

    logger.info(
        f"Actualizando ejecución: pipeline_id={pipeline_id} | "
        f"correlation_id={correlation_id} | status={final_status}"
    )

    # Keys de DynamoDB
    lock_key = {
        'pipeline_id': {'S': pipeline_id},
        'execution_time': {'N': '0'}  # Lock fijo
    }
    execution_key = {
        'pipeline_id': {'S': pipeline_id},
        'execution_time': {'N': execution_time}
    }

    # Construcción del update para el registro histórico
    update_expression = 'SET #s = :s, end_time = :et'
    expression_attr_names = {'#s': 'status'}
    expression_attr_values = {
        ':s': {'S': final_status},
        ':et': {'S': datetime.now(timezone.utc).isoformat()},
    }

    # Agregar error si existe
    if error_info and (error_info.get('Error') or error_info.get('Cause')):
        update_expression += ', #err = :err'
        expression_attr_names['#err'] = 'error'
        expression_attr_values[':err'] = {'S': json.dumps(error_info)}

    # Construcción del update para el lock
    lock_update_expression = 'SET #s = :ls'
    lock_attr_values = {
        ':ls': {'S': lock_status},
        ':cid': {'S': correlation_id},
    }

    try:
        # Transacción atómica: liberar lock + cerrar historial
        dynamodb.transact_write_items(
            TransactItems=[
                {
                    'Update': {
                        'TableName': TABLE_NAME,
                        'Key': lock_key,
                        'ConditionExpression': 'last_exec_id = :cid',
                        'UpdateExpression': lock_update_expression,
                        'ExpressionAttributeNames': {'#s': 'status'},
                        'ExpressionAttributeValues': lock_attr_values,
                    }
                },
                {
                    'Update': {
                        'TableName': TABLE_NAME,
                        'Key': execution_key,
                        'UpdateExpression': update_expression,
                        'ExpressionAttributeNames': expression_attr_names,
                        'ExpressionAttributeValues': expression_attr_values,
                    }
                }
            ]
        )

        logger.info(f"Lock liberado y registro actualizado: {final_status}")

        return {
            'statusCode': 200,
            'body': {
                'pipeline_id': pipeline_id,
                'execution_id': correlation_id,
                'status': final_status,
                'message': f'Pipeline finalizado con estado: {final_status}',
            }
        }

    except ClientError as e:
        logger.error(f"Error en transacción DynamoDB: {e}")
        raise
