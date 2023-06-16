import os
import sys

local_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(local_path, "../../config")
sys.path.append(config_path)

from kfp.v2 import compiler
from google_cloud_pipeline_components import aiplatform as gcc_aip
from kfp.v2.dsl import component, pipeline, Model, Input, Output, Artifact

from config import (
    DATASET_ID,
    PROJECT_ID,
    FEATURESTORE_ID,
    VALUES_ENTITY_ID,
    SERVING_FEATURE_IDS,
    TABLE_ID,
    TABLE_RAW_ID,
    TABLE_FILTERED_TEMP_ID,
    TABLE_TRAIN_ID,
    TABLE_PURCHASES_TEMP_ID,
    TABLE_RETURNS_TEMP_ID,
    TABLE_FEATURE_ENGINEER_ID,
    TABLE_SAVE_PREDICTIONS_ID,
    TABLE_INSTACES_ID,
    PIPELINE_NAME,
    PIPELINE_ROOT,
    REGION,
    FEATURE_TIME,
    FEATURES,
    TARGET,
    DEPLOY_VERSION,
    FRAMEWORK,
    REGION,
    MODEL_NAME,
    MODEL_REGISTRY_NAME,
    N_CLUSTERS,
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


@component(
    packages_to_install=["pandas", "google-cloud-bigquery", "db-dtypes", "pandas-gbq"],
    base_image="python:3.10.6"
)
def data_preparation(
    project_id: str,
    dataset_id: str,
    table_id: str,
    table_filtered_temp_id: str,
    table_purchases_temp_id: str,
    table_returns_temp_id: str,
):
    import os
    import logging
    from typing import Tuple

    import pandas as pd
    import pandas_gbq

    logging.info("Iniciando o componente")

    PROJECT_NUMBER = os.environ["CLOUD_ML_PROJECT_ID"]

    def keep_features(dataframe: pd.DataFrame, keep_columns: list) -> pd.DataFrame:
        """
        Retorna um DataFrame com as colunas especificadas em keep_columns.

        Args:
            dataframe (pd.DataFrame): O DataFrame a ser processado.
            keep_columns (list): A lista de nomes de colunas a serem mantidas no DataFrame resultante.

        Returns:
            pd.DataFrame: O DataFrame resultante com apenas as colunas especificadas em keep_columns.
        """
        return dataframe[keep_columns]

    def column_to_int(dataframe: pd.DataFrame, column_name: str) -> bool:
        """
        Converte a coluna especificada em um dataframe para o tipo inteiro.

        Args:
            dataframe (pd.DataFrame): O dataframe a ser processado.
            column_name (str): O nome da coluna a ser convertida.

        Returns:
            bool: True se a conversão foi bem sucedida, False caso contrário.
        """
        try:
            dataframe[column_name] = dataframe[column_name].astype(int)
        except (ValueError, TypeError):
            # Lidar com valores ausentes e conversões inválidas
            return False

        # Retorna True se a conversão foi bem sucedida
        return True

    def column_to_date(
        dataframe: pd.DataFrame, column_name: str, date_format: str = None
    ) -> bool:
        """
        Converte a coluna especificada em um dataframe para o tipo data.

        Args:
            dataframe (pd.DataFrame): O dataframe a ser processado.
            column_name (str): O nome da coluna a ser convertida.
            date_format (str, opcional): O formato de data personalizado. Se nenhum formato for especificado, o pandas usará o padrão 'YYYY-MM-DD'.

        Returns:
            bool: True se a conversão foi bem sucedida, False caso contrário.
        """
        try:
            if date_format:
                dataframe[column_name] = pd.to_datetime(
                    dataframe[column_name], format=date_format
                )
            else:
                dataframe[column_name] = pd.to_datetime(dataframe[column_name])
        except (ValueError, TypeError):
            # Lidar com valores ausentes e conversões inválidas
            return False

        # Retorna True se a conversão foi bem sucedida
        return True

    def change_column_type(dataframe_raw: pd.DataFrame):
        """
        Changes the data type of a given column in a DataFrame.

        Args:
            dataframe_raw: A pandas DataFrame.

        Returns:
            None.
        """
        column_to_int(dataframe_raw, "CustomerID")
        column_to_date(dataframe_raw, "InvoiceDate")

    def filtering_features(
        dataframe_raw: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Filters and preprocesses the input dataframe.

        Args:
            dataframe_raw: A pandas DataFrame containing raw sales data.

        Returns:
            Three pandas DataFrames containing the filtered returns and purchases data, and the filtered main data.
        """
        # Filter returns and purchases data
        df_returns = dataframe_raw.loc[
            dataframe_raw["Quantity"] < 0, ["CustomerID", "Quantity"]
        ]
        df_purchases = dataframe_raw.loc[dataframe_raw["Quantity"] >= 0, :]

        # Filter main data
        df_filtered = keep_features(
            dataframe_raw,
            [
                "InvoiceNo",
                "StockCode",
                "Quantity",
                "InvoiceDate",
                "UnitPrice",
                "CustomerID",
                "Country",
            ],
        )

        return df_filtered, df_purchases, df_returns

    def run_data_preparation(
        dataframe_raw: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Preprocesses the input dataframe by performing column type conversion and filtering features.

        Args:
            dataframe_raw (pd.DataFrame): A pandas DataFrame containing raw sales data.

        Returns:
            A tuple of three pandas DataFrames: df_filtered, df_purchases, and df_returns.
            - df_filtered: A DataFrame containing the filtered main data.
            - df_purchases: A DataFrame containing the filtered purchases data.
            - df_returns: A DataFrame containing the filtered returns data.
        """
        change_column_type(dataframe_raw)
        return filtering_features(dataframe_raw)

    query_sql = f"""SELECT *
                    FROM  `{project_id}.{dataset_id}.{table_id}`
                    WHERE InvoiceDate <= CURRENT_DATE """

    data = pd.read_gbq(query=query_sql, project_id=PROJECT_NUMBER)
    logging.info(f"Tabela carregada: `{project_id}.{dataset_id}.{table_id}`")

    df_filtered, df_purchases, df_returns = run_data_preparation(data)

    pandas_gbq.to_gbq(
        df_filtered,
        f"{project_id}.{dataset_id}.{table_filtered_temp_id}",
        project_id=PROJECT_NUMBER,
        if_exists="replace",
    )
    pandas_gbq.to_gbq(
        df_purchases,
        f"{project_id}.{dataset_id}.{table_purchases_temp_id}",
        project_id=PROJECT_NUMBER,
        if_exists="replace",
    )
    pandas_gbq.to_gbq(
        df_returns,
        f"{project_id}.{dataset_id}.{table_returns_temp_id}",
        project_id=PROJECT_NUMBER,
        if_exists="replace",
    )

    logging.info(
        f"Tabelas criadas no BigQuery: {table_filtered_temp_id} e {table_purchases_temp_id} e {table_returns_temp_id}"
    )


@component(
    packages_to_install=[
        "pandas",
        "google-cloud-bigquery",
        "db-dtypes",
        "pandas-gbq",
        "google-cloud",
    ],
    base_image="python:3.10.6"
)
def feature_engineering(
    project_id: str,
    dataset_id: str,
    table_id: str,
    table_raw_id: str,
    table_filtered_temp_id: str,
    table_purchases_temp_id: str,
    table_returns_temp_id: str,
):
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound
    import pandas as pd
    import pandas_gbq
    import os
    import logging
    from functools import reduce
    from typing import Union

    PROJECT_NUMBER = os.environ["CLOUD_ML_PROJECT_ID"]

    logging.info("Iniciando o componente")

    def column_to_date(
        dataframe: pd.DataFrame, column_name: str, date_format: str = None
    ) -> bool:
        """
        Converte a coluna especificada em um dataframe para o tipo data.

        Args:
            dataframe (pd.DataFrame): O dataframe a ser processado.
            column_name (str): O nome da coluna a ser convertida.
            date_format (str, opcional): O formato de data personalizado. Se nenhum formato for especificado, o pandas usará o padrão 'YYYY-MM-DD'.

        Returns:
            bool: True se a conversão foi bem sucedida, False caso contrário.
        """
        try:
            if date_format:
                dataframe[column_name] = pd.to_datetime(
                    dataframe[column_name], format=date_format
                )
            else:
                dataframe[column_name] = pd.to_datetime(dataframe[column_name])
        except (ValueError, TypeError):
            # Lidar com valores ausentes e conversões inválidas
            return False

        # Retorna True se a conversão foi bem sucedida
        return True

    def keep_features(dataframe: pd.DataFrame, keep_columns: list) -> pd.DataFrame:
        """
        Retorna um DataFrame com as colunas especificadas em keep_columns.

        Args:
            dataframe (pd.DataFrame): O DataFrame a ser processado.
            keep_columns (list): A lista de nomes de colunas a serem mantidas no DataFrame resultante.

        Returns:
            pd.DataFrame: O DataFrame resultante com apenas as colunas especificadas em keep_columns.
        """
        return dataframe[keep_columns]

    def calculate_gross_revenue(dataframe_purchases: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula a receita bruta de cada cliente com base nas colunas 'Quantity' e 'UnitPrice' e retorna
        um DataFrame com as colunas 'CustomerID' e 'gross_revenue'.

        Args:
            dataframe_purchases (pd.DataFrame): O DataFrame das compras contendo as colunas 'CustomerID', 'Quantity' e 'UnitPrice'.

        Returns:
            pd.DataFrame: O DataFrame resultante contendo as colunas 'CustomerID' e 'gross_revenue'.
        """
        # Verifica se as colunas necessárias estão presentes no DataFrame de entrada
        required_columns = {"CustomerID", "Quantity", "UnitPrice"}
        missing_columns = required_columns - set(dataframe_purchases.columns)
        if missing_columns:
            raise ValueError(
                f"O DataFrame de entrada está faltando as seguintes colunas: {missing_columns}"
            )

        # Calcula a receita bruta e agrupa por CustomerID
        dataframe_purchases.loc[:, "gross_revenue"] = (
            dataframe_purchases.loc[:, "Quantity"]
            * dataframe_purchases.loc[:, "UnitPrice"]
        )
        grouped_df = (
            dataframe_purchases.groupby("CustomerID")
            .agg({"gross_revenue": "sum"})
            .reset_index()
        )

        return grouped_df

    def create_recency(
        dataframe_purchases: pd.DataFrame, dataframe_filtered: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calcula a recência da última compra para cada cliente.

        Args:
            dataframe_purchases (pd.DataFrame): DataFrame com as informações de compras de todos os clientes.
            dataframe_filtered (pd.DataFrame): DataFrame filtrado apenas com as informações dos clientes que desejamos calcular a recência.

        Returns:
            pd.DataFrame: DataFrame com as colunas 'CustomerID' e 'recency_days', indicando a recência em dias da última compra para cada cliente.

        """
        # calcula a data da última compra de cada cliente
        df_recency = (
            dataframe_purchases.loc[:, ["CustomerID", "InvoiceDate"]]
            .groupby("CustomerID")
            .max()
            .reset_index()
        )

        # calcula a recência em dias da última compra de cada cliente em relação à data mais recente da base de dados filtrada
        df_recency.loc[:, "recency_days"] = (
            dataframe_filtered["InvoiceDate"].max() - df_recency["InvoiceDate"]
        ).dt.days

        # retorna o DataFrame apenas com as colunas 'CustomerID' e 'recency_days'
        return df_recency[["CustomerID", "recency_days"]]

    def create_quantity_purchased(dataframe_purchases: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula a quantidade de produtos adquiridos por cada cliente.

        Args:
            dataframe_purchases (pd.DataFrame): DataFrame com as informações de compras de todos os clientes.

        Returns:
            pd.DataFrame: DataFrame com as colunas 'CustomerID' e 'qty_products', indicando a quantidade de produtos adquiridos por cada cliente.
        """
        # agrupa as informações de compras por CustomerID e conta o número de StockCode para cada grupo
        qty_purchased = (
            dataframe_purchases.loc[:, ["CustomerID", "StockCode"]]
            .groupby("CustomerID")
            .count()
        )

        # renomeia a coluna StockCode para qty_products e reseta o índice para transformar o CustomerID em uma coluna
        qty_purchased = qty_purchased.reset_index().rename(
            columns={"StockCode": "qty_products"}
        )

        # retorna o DataFrame com as colunas 'CustomerID' e 'qty_products'
        return qty_purchased

    def create_freq_purchases(dataframe_purchases: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the purchase frequency of each customer based on the purchase history.

        Parameters
        ----------
        dataframe_purchases : pd.DataFrame
            DataFrame with purchase history of each customer, containing columns CustomerID, InvoiceNo, and InvoiceDate.

        Returns
        -------
        pd.DataFrame
            DataFrame with the purchase frequency of each customer, containing columns CustomerID and frequency.
        """

        # Calculate time range of purchases for each customer
        df_aux = (
            dataframe_purchases[["CustomerID", "InvoiceNo", "InvoiceDate"]]
            .drop_duplicates()
            .groupby("CustomerID")
            .agg(
                max_=("InvoiceDate", "max"),
                min_=("InvoiceDate", "min"),
                days_=("InvoiceDate", lambda x: ((x.max() - x.min()).days) + 1),
                buy_=("InvoiceNo", "count"),
            )
            .reset_index()
        )

        # Calculate frequency of purchases for each customer
        df_aux["frequency"] = df_aux[["buy_", "days_"]].apply(
            lambda x: x["buy_"] / x["days_"] if x["days_"] != 0 else 0, axis=1
        )

        return df_aux

    def create_qty_returns(dataframe_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the total quantity of returned products for each customer.

        Args:
            dataframe_returns: A pandas DataFrame containing information about returns.

        Returns:
            A pandas DataFrame with the total quantity of returned products for each customer.
        """
        # Validate input data
        # if dataframe_returns is None:
        #     raise ValueError("Input DataFrame is empty")
        # if not all(col in dataframe_returns.columns for col in ['CustomerID', 'Quantity']):
        #     raise ValueError("Input DataFrame must contain 'CustomerID' and 'Quantity' columns")

        # Compute quantity of returns
        df_returns = (
            dataframe_returns[["CustomerID", "Quantity"]]
            .groupby("CustomerID")
            .sum()
            .reset_index()
            .rename(columns={"Quantity": "qty_returns"})
        )
        df_returns["qty_returns"] = df_returns["qty_returns"] * -1

        return df_returns

    def run_feature_engineering(
        dataframe_filtered: pd.DataFrame,
        dataframe_purchases: pd.DataFrame,
        dataframe_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Performs feature engineering on the input dataframes and returns a new dataframe with the engineered features.

        Args:
            dataframe_filtered: A pandas DataFrame containing filtered customer order data.
            dataframe_purchases: A pandas DataFrame containing customer purchase data.
            dataframe_returns: A pandas DataFrame containing customer return data.

        Returns:
            A pandas DataFrame with the engineered features for each customer.
        """
        # Check if input dataframes are empty
        if dataframe_filtered.empty:
            raise ValueError("Input DataFrame 'dataframe_filtered' is empty")
        if dataframe_purchases.empty:
            raise ValueError("Input DataFrame 'dataframe_purchases' is empty")
        # if dataframe_returns.empty:
        #     raise ValueError("Input DataFrame 'dataframe_returns' is empty")

        # Check if required columns are present in input dataframes
        required_columns = [
            "CustomerID",
            "InvoiceDate",
            "StockCode",
            "Quantity",
            "UnitPrice",
        ]
        for df, name in zip(
            [dataframe_filtered, dataframe_purchases],
            ["dataframe_filtered", "dataframe_purchases"],
        ):
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(
                    f"Missing columns {missing_columns} in input DataFrame '{name}'"
                )
        if "CustomerID" not in dataframe_returns.columns:
            raise ValueError(
                "Column 'CustomerID' not found in input DataFrame 'dataframe_returns'"
            )
        if "Quantity" not in dataframe_returns.columns:
            raise ValueError(
                "Column 'Quantity' not found in input DataFrame 'dataframe_returns'"
            )

        # Perform feature engineering
        df_fengi = keep_features(dataframe_filtered, ["CustomerID"]).drop_duplicates(
            ignore_index=True
        )
        gross_revenue = calculate_gross_revenue(dataframe_purchases)
        df_recency = create_recency(dataframe_purchases, dataframe_filtered)
        df_qty_products = create_quantity_purchased(dataframe_purchases)
        df_freq = create_freq_purchases(dataframe_purchases)
        returns = create_qty_returns(dataframe_returns)

        # Merge dataframes
        dfs = [df_fengi, gross_revenue, df_recency, df_qty_products, df_freq, returns]
        df_fengi = reduce(
            lambda left, right: pd.merge(left, right, on="CustomerID", how="left"), dfs
        )

        # Fill NaN values
        df_fengi["qty_returns"] = df_fengi["qty_returns"].fillna(0)

        # Select final features and return dataframe
        features = [
            "CustomerID",
            "gross_revenue",
            "recency_days",
            "qty_products",
            "frequency",
            "qty_returns",
        ]
        return keep_features(df_fengi, features).dropna()

    def table_exists(dataset_table_id: str) -> bool:
        client = bigquery.Client()

        try:
            client.get_table(dataset_table_id)  # Make an API request.
            return True
        except NotFound:
            return False

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

    logging.info("Carregando as tabelas da preparacao de dados")
    query_filtered = f"""SELECT *
                    FROM  `{project_id}.{dataset_id}.{table_filtered_temp_id}`
                    WHERE InvoiceDate <= CURRENT_TIMESTAMP() """
    df_filtered = pd.read_gbq(query=query_filtered, project_id=PROJECT_NUMBER)

    query_purchases = f"""SELECT *
                    FROM  `{project_id}.{dataset_id}.{table_purchases_temp_id}`
                    WHERE InvoiceDate <= CURRENT_TIMESTAMP() """
    df_purchases = pd.read_gbq(query=query_purchases, project_id=PROJECT_NUMBER)

    query_returns = f"""SELECT *
                    FROM  `{project_id}.{dataset_id}.{table_returns_temp_id}`"""
    df_returns = pd.read_gbq(query=query_returns, project_id=PROJECT_NUMBER)

    logging.info("Transformando a coluna InvoiceDate para o tipo DATE")
    column_to_date(df_filtered, "InvoiceDate")
    column_to_date(df_purchases, "InvoiceDate")

    logging.info(
        f"Iniciando a verificacao de existencia da tabela: {dataset_id}.{table_id}"
    )
    # Verifica se a tabela existe
    if table_exists(f"{project_id}.{dataset_id}.{table_id}"):
        logging.info("Tabela existente, inicia insercao de novos dados")

        sql_new_customers = f"""SELECT
                                      DISTINCT CustomerID
                                    FROM
                                      `{project_id}.{dataset_id}.{table_raw_id}`
                                    WHERE
                                      InvoiceDate = CURRENT_DATE()"""
        new_customers = pd.read_gbq(sql_new_customers, project_id=PROJECT_NUMBER)[
            "CustomerID"
        ].tolist()

        df_fengi = run_feature_engineering(
            df_filtered.loc[df_filtered["CustomerID"].isin(new_customers)],
            df_purchases.loc[df_purchases["CustomerID"].isin(new_customers)],
            df_returns.loc[df_returns["CustomerID"].isin(new_customers)],
        )

        # Inserir os dados na tabela usando SQL
        pandas_gbq.to_gbq(
            df_fengi,
            f"{project_id}.{dataset_id}.{table_id}",
            project_id=PROJECT_NUMBER,
            if_exists="append",
        )
        sql_update_new_customer = f"""
                                        UPDATE `{project_id}.{dataset_id}.{table_id}`
                                        SET values = generate_uuid(),
                                        timestamp = current_timestamp()
                                        WHERE CustomerID IN {tuple(new_customers)}"""
        logging.info(sql_update_new_customer)
        run_bq_query(sql_update_new_customer, project_name=project_id)
    else:
        # Cria a tabela e insere os dados
        logging.info("Tabela nao existente, cria a tabela e inicia insercao dos dados")
        df_fengi = run_feature_engineering(df_filtered, df_purchases, df_returns)
        pandas_gbq.to_gbq(
            df_fengi,
            f"{project_id}.{dataset_id}.{table_id}",
            project_id=PROJECT_NUMBER,
            if_exists="fail",
        )
        query = f"""CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_id}` as (
                    SELECT
                        *,
                        generate_uuid() as values,
                        current_timestamp() as timestamp,
                    FROM 
                        `{project_id}.{dataset_id}.{table_id}`);"""
        run_bq_query(query, project_name=project_id)


@component(
    packages_to_install=["google-cloud-aiplatform", "pyarrow"],
    base_image="python:3.10.6"
)
def create_feature_store(
    project_id: str,
    dataset_id: str,
    table_id: str,
    featurestore_id: str,
    values_entity_id: str,
    feature_time: str,
    region: str,
):
    import os
    import logging

    from google.cloud import aiplatform

    logging.info("Iniciando o componente")
    VALUES_BQ_SOURCE_URI = f"bq://{project_id}.{dataset_id}.{table_id}"
    PROJECT_NUMBER = os.environ["CLOUD_ML_PROJECT_ID"]
    aiplatform.init(project=PROJECT_NUMBER, location=region)

    try:
        # Checks if there is already a Featurestore
        ecommerce_feature_store = aiplatform.Featurestore(f"{featurestore_id}")
        logging.info(f"""A feature store {featurestore_id} ja existe.""")
    except:
        # Creates a Featurestore
        logging.info(f"""Criando a feature store: {featurestore_id}.""")
        ecommerce_feature_store = aiplatform.Featurestore.create(
            featurestore_id=f"{featurestore_id}",
            online_store_fixed_node_count=1,
            sync=True,
        )

    try:
        # get entity type, if it already exists
        values_entity_type = ecommerce_feature_store.get_entity_type(
            entity_type_id=values_entity_id
        )
    except:
        # else, create entity type
        values_entity_type = ecommerce_feature_store.create_entity_type(
            entity_type_id=values_entity_id, description="Values Entity", sync=True
        )

    values_feature_configs = {
        "gross_revenue": {
            "value_type": "DOUBLE",
            "description": "Gross Revenue",
            "labels": {"status": "passed"},
        },
        "recency_days": {
            "value_type": "DOUBLE",
            "description": "Recency Days",
            "labels": {"status": "passed"},
        },
        "qty_products": {
            "value_type": "DOUBLE",
            "description": "Quantity products",
            "labels": {"status": "passed"},
        },
        "frequency": {
            "value_type": "DOUBLE",
            "description": "Frequency",
            "labels": {"status": "passed"},
        },
        "qty_returns": {
            "value_type": "INT64",
            "description": "Quantity returns",
            "labels": {"status": "passed"},
        },
    }

    values_feature_ids = values_entity_type.batch_create_features(
        feature_configs=values_feature_configs, sync=True
    )

    values_features_ids = [
        feature.name for feature in values_feature_ids.list_features()
    ]

    logging.info(f"""Ingerindo os dados na feature store: {featurestore_id}.""")
    values_entity_type.ingest_from_bq(
        feature_ids=values_features_ids,
        feature_time=feature_time,
        bq_source_uri=VALUES_BQ_SOURCE_URI,
        entity_id_field=values_entity_id,
        disable_online_serving=True,
        worker_count=2,
        sync=True,
    )


@component(
    packages_to_install=[
        "google-cloud-aiplatform",
        "google-cloud-bigquery",
        "db-dtypes",
        "pandas",
    ],
    base_image="python:3.10.6"
)
def create_batch_serve_fs(
    project_id: str,
    dataset_id: str,
    table_feature_engineer_id: str,
    featurestore_id: str,
    serving_feature_ids: dict,
    table_instaces_id: str,
    region: str,
    table_train_id: str,
):
    import os
    import logging
    from typing import Union

    import pandas as pd
    from google.cloud import bigquery
    from google.cloud import aiplatform

    TRAIN_TABLE_URI = f"bq://{project_id}.{dataset_id}.{table_train_id}"
    PROJECT_NUMBER = os.environ["CLOUD_ML_PROJECT_ID"]
    aiplatform.init(project=PROJECT_NUMBER, location=region)

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

    read_instances_query = f"""
                CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_instaces_id}` as (
                    SELECT   
                        values,
                        timestamp,
                    FROM 
                        `{project_id}.{dataset_id}.{table_feature_engineer_id}` 
                );
                """

    logging.info("Criando a tabela de instancia")
    run_bq_query(read_instances_query, project_name=project_id)

    logging.info(f"Iniciando o fornecimento das features da: {featurestore_id}")
    ecommerce_feature_store = aiplatform.Featurestore(featurestore_name=featurestore_id)

    logging.info(
        f"Executando o comando para o destino: {TRAIN_TABLE_URI} a partir da tabela: {table_instaces_id}"
    )
    ecommerce_feature_store.batch_serve_to_bq(
        bq_destination_output_uri=TRAIN_TABLE_URI,
        serving_feature_ids=serving_feature_ids,
        read_instances_uri=f"bq://{project_id}.{dataset_id}.{table_instaces_id}",
    )


@component(
    packages_to_install=[
        "google-cloud-aiplatform",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "google-cloud-bigquery",
        "db-dtypes",
    ],
    base_image="python:3.10.6"
)
def model_train(
    project_id: str,
    dataset_id: str,
    features: list,
    target: str,
    deploy_version: str,
    framework: str,
    region: str,
    table_train_id: str,
    model_name: str,
    n_clusters: int,
    model: Output[Artifact],
):
    import os
    import pickle
    import pathlib
    import logging
    from typing import Union

    import pandas as pd

    from google.cloud import bigquery
    from sklearn.cluster import KMeans
    from google.cloud import aiplatform
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler

    logging.info("Iniciando o componente")

    REGION_SPLITTED = "us-central1".split("-")[0]
    DEPLOY_IMAGE = (
        f"{REGION_SPLITTED}-docker.pkg.dev/vertex-ai/prediction/{deploy_version}:latest"
    )

    PROJECT_NUMBER = os.environ["CLOUD_ML_PROJECT_ID"]

    aiplatform.init(project=PROJECT_NUMBER, location=region)
    scaler = MinMaxScaler()

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

    logging.info("Carregando dados para o treinamento")
    df = run_bq_query(
        f"select * from `{project_id}.{dataset_id}.{table_train_id}`",
        project_name=project_id,
    )

    logging.info("Iniciando o treinamento")

    X = df[features].copy()
    y = df[target]

    model_pipeline = Pipeline(
        [
            ("pca", PCA(n_components=2)),
            ("scaler", scaler),
            ("clustering", KMeans(n_clusters=n_clusters, random_state=42)),
        ]
    )

    model_pipeline.fit(X, y)

    logging.info("Criando o modelo output")
    model.metadata["framework"] = framework
    model.metadata["containerSpec"] = {"imageUri": DEPLOY_IMAGE}

    file_name = model.path + f"/{model_name}"

    pathlib.Path(model.path).mkdir()
    with open(file_name, "wb") as file:
        pickle.dump(model_pipeline, file)


@component(
    packages_to_install=[
        "google-cloud-aiplatform",
        "google-cloud-bigquery",
        "db-dtypes",
        "pandas",
        "scikit-learn",
        "pandas-gbq",
    ],
    base_image="python:3.10.6"
)
def batch_prediction(
    project_id: str,
    dataset_id: str,
    features: list,
    region: str,
    table_save_predictions_id: str,
    model_name: str,
    table_train_id: str,
    model: Input[Model],
):
    import os
    import pickle
    import logging

    import pandas_gbq
    import pandas as pd
    from google.cloud import aiplatform

    SQL_TO_PREDICT_DATA = f"""SELECT * 
                        FROM `{project_id}.{dataset_id}.{table_train_id}`"""
    PROJECT_NUMBER = os.environ["CLOUD_ML_PROJECT_ID"]

    aiplatform.init(project=PROJECT_NUMBER, location=region)

    logging.info("Iniciando o componente")

    logging.info(f"Carregando o modelo: {model_name}")
    file_name = model.path + f"/{model_name}"
    with open(file_name, "rb") as file:
        model_pipeline = pickle.load(file)

    logging.info("Carregando dados a serem preditos")
    predict_data = pd.read_gbq(SQL_TO_PREDICT_DATA, project_id=PROJECT_NUMBER)

    logging.info(
        f"Iniciando a predicao dos dados: {project_id}.{dataset_id}.{table_train_id}"
    )
    labels = model_pipeline.predict(predict_data[features])
    predict_data["Clusters"] = labels

    logging.info(
        f"Substituindo os dados preditos: {project_id}.{dataset_id}.{table_save_predictions_id}"
    )
    pandas_gbq.to_gbq(
        predict_data,
        f"{project_id}.{dataset_id}.{table_save_predictions_id}",
        project_id=PROJECT_NUMBER,
        if_exists="replace",
    )


@pipeline(pipeline_root=PIPELINE_ROOT, name=PIPELINE_NAME.replace("_", "-"))
def ecommerce_pipeline():
    dataset_op = get_data(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_raw_id=TABLE_RAW_ID,
        table_id=TABLE_ID,
    )
    data_prep_op = data_preparation(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=TABLE_ID,
        table_filtered_temp_id=TABLE_FILTERED_TEMP_ID,
        table_purchases_temp_id=TABLE_PURCHASES_TEMP_ID,
        table_returns_temp_id=TABLE_RETURNS_TEMP_ID,
    ).after(dataset_op)

    feature_engineering_op = feature_engineering(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=TABLE_FEATURE_ENGINEER_ID,
        table_raw_id=TABLE_RAW_ID,
        table_filtered_temp_id=TABLE_FILTERED_TEMP_ID,
        table_purchases_temp_id=TABLE_PURCHASES_TEMP_ID,
        table_returns_temp_id=TABLE_RETURNS_TEMP_ID,
    )
    feature_engineering_op.after(data_prep_op)

    feature_store_op = create_feature_store(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=TABLE_FEATURE_ENGINEER_ID,
        featurestore_id=FEATURESTORE_ID,
        values_entity_id=VALUES_ENTITY_ID,
        feature_time=FEATURE_TIME,
        region=REGION,
    ).after(feature_engineering_op)

    batch_serve_fs_op = create_batch_serve_fs(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_feature_engineer_id=TABLE_FEATURE_ENGINEER_ID,
        featurestore_id=FEATURESTORE_ID,
        serving_feature_ids=SERVING_FEATURE_IDS,
        table_instaces_id=TABLE_INSTACES_ID,
        region=REGION,
        table_train_id=TABLE_TRAIN_ID,
    ).after(feature_store_op)

    model_train_op = model_train(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        features=FEATURES,
        target=TARGET,
        deploy_version=DEPLOY_VERSION,
        framework=FRAMEWORK,
        region=REGION,
        table_train_id=TABLE_TRAIN_ID,
        model_name=MODEL_NAME,
        n_clusters=N_CLUSTERS,
    ).after(batch_serve_fs_op)

    model_upload_op = gcc_aip.ModelUploadOp(
        project=PROJECT_ID,
        location=REGION,
        display_name=f"{MODEL_REGISTRY_NAME}",
        unmanaged_container_model=model_train_op.outputs["model"],
    ).after(model_train_op)

    batch_predict_op = batch_prediction(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        features=FEATURES,
        region=REGION,
        table_save_predictions_id=TABLE_SAVE_PREDICTIONS_ID,
        model_name=MODEL_NAME,
        table_train_id=TABLE_TRAIN_ID,
        model=model_train_op.outputs["model"],
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=ecommerce_pipeline, package_path=f"{PIPELINE_NAME}.json"
    )
