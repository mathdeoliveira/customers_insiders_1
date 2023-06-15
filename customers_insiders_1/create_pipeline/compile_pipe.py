import os
import sys

local_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(local_path, "../../config")
utils_path = os.path.join(local_path, "../utils")
sys.path.append(config_path)
sys.path.append(utils_path)

from kfp.v2 import compiler
from kfp.v2.dsl import component, pipeline

from config import (
    DATASET_ID,
    PIPELINE_NAME,
    PIPELINE_ROOT,
    PROJECT_ID,
    TABLE_ID,
    TABLE_RAW_ID,
)


@component(
    packages_to_install=["google-cloud-bigquery", "db-dtypes", "pandas"],
    base_image="python:3.10.6",
)
def get_data(project_id: str, dataset_id: str, table_raw_id: str, table_id: str):
    import logging
    from typing import Union

    import pandas as pd
    from google.cloud import bigquery

    def run_bq_query(sql: str, project_name: str) -> Union[str, pd.DataFrame]:
        """
        Run a BigQuery query and return the job ID or result as a DataFrame
        Args:
            sql: SQL query, as a string, to execute in BigQuery
        Returns:
            df: DataFrame of results from query,  or error, if any
        """

        bq_client = bigquery.Client(project=project_name)

        # Try dry run before executing query to catch any errors
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        bq_client.query(sql, job_config=job_config)

        # If dry run succeeds without errors, proceed to run query
        job_config = bigquery.QueryJobConfig()
        client_result = bq_client.query(sql, job_config=job_config)

        job_id = client_result.job_id

        # Wait for query/job to finish running. then get & return data frame
        df = client_result.result().to_arrow().to_pandas()
        print(f"Finished job_id: {job_id}")
        return df

    query = f"""
                CREATE OR REPLACE TABLE
               `{project_id}.{dataset_id}.{table_id}` (InvoiceNo STRING,
                StockCode STRING,
                Description STRING,
                Quantity INT64,
                InvoiceDate DATE,
                UnitPrice FLOAT64,
                CustomerID FLOAT64,
                Country STRING)
            PARTITION BY
              InvoiceDate AS (
              WITH
                not_nulls AS (
                SELECT
                  *
                FROM
                  `{project_id}.{dataset_id}.{table_raw_id}`
                WHERE
                  InvoiceDate <= CURRENT_DATE()
                  AND CustomerID IS NOT NULL
                  AND Description IS NOT NULL),
                filtering_features AS (
                SELECT
                  *
                FROM
                  not_nulls
                WHERE
                  UnitPrice >= 0.04
                  AND Country NOT IN ('European Community',
                    'Unspecified')
                  AND StockCode NOT IN ('POST',
                    'D',
                    'DOT',
                    'M',
                    'S',
                    'AMAZONFEE',
                    'm',
                    'DCGSSBOY',
                    'DCGSSGIRL',
                    'PADS',
                    'B',
                    'CRUK')
                  AND CustomerID != 16446)
              SELECT
                *
              FROM
                filtering_features);
    """

    run_bq_query(query, project_name=project_id)
    logging.info(f"Tabela criada: {project_id}.{dataset_id}.{table_id}")


@pipeline(pipeline_root=PIPELINE_ROOT, name=PIPELINE_NAME.replace("_", "-"))
def ecommerce_pipeline():
    dataset_op = get_data(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_raw_id=TABLE_RAW_ID,
        table_id=TABLE_ID,
    )
    # data_prep_op = data_preparation()
    # data_prep_op.after(dataset_op)

    # feature_engineering_op = feature_engineering()
    # feature_engineering_op.after(data_prep_op)

    # feature_store_op = create_feature_store()
    # feature_store_op.after(feature_engineering_op)

    # batch_serve_fs_op = create_batch_serve_fs()
    # batch_serve_fs_op.after(feature_store_op)

    # model_train_op = model_train()
    # model_train_op.after(batch_serve_fs_op)

    # model_upload_op = gcc_aip.ModelUploadOp(
    #     project=PROJECT_ID,
    #     location=REGION,
    #     display_name=f"{MODEL_NAME}",
    #     unmanaged_container_model=model_train_op.outputs["model"],
    # ).after(model_train_op)


#     endpoint_create_op = gcc_aip.EndpointCreateOp(
#         project=PROJECT_ID,
#         location=REGION,
#         display_name=f"{MODEL_NAME}-endpoint",
#     )

#     model_deploy_op = gcc_aip.ModelDeployOp(
#         endpoint=endpoint_create_op.outputs["endpoint"],
#         model=model_upload_op.outputs["model"],
#         deployed_model_display_name=f"{MODEL_NAME}",
#         dedicated_resources_machine_type="n1-standard-4",
#         dedicated_resources_min_replica_count=1,
#         dedicated_resources_max_replica_count=1,
#     )

# batch_predict_op = batch_prediction(model=model_train_op.outputs["model"])
# batch_predict_op.after(model_upload_op)

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=ecommerce_pipeline, package_path=f"{PIPELINE_NAME}.json"
    )
