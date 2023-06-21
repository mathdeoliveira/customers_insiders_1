#https://cloud.google.com/bigquery/docs/arima-single-time-series-forecasting-tutorial#step_six_use_your_model_to_forecast_the_time_series
#https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-create-time-series
#https://github.com/statmike/vertex-ai-mlops/blob/main/03%20-%20BigQuery%20ML%20(BQML)/03a%20-%20BQML%20Logistic%20Regression.ipynb
#https://github.com/GoogleCloudPlatform/analytics-componentized-patterns/blob/master/gaming/propensity-model/bqml/bqml_ga4_gaming_propensity_to_churn.ipynb
CREATE MODEL
  `gcp-vertex.gcp_bq.ts_teste` OPTIONS(MODEL_TYPE='ARIMA_PLUS',
    time_series_timestamp_col='InvoiceDate',
    time_series_data_col='Quantity',
    auto_arima = TRUE,
   data_frequency = 'DAILY',
    SEASONALITIES=['DAILY'],
   decompose_time_series = TRUE) AS
SELECT
  InvoiceDate,
  SUM(Quantity) AS Quantity
FROM
  `gcp_bq.ecommerce_cds`
WHERE
  InvoiceDate BETWEEN '2016-11-29'
  AND '2017-11-29'
GROUP BY
  InvoiceDate

SELECT
 *
FROM
 ML.ARIMA_EVALUATE(MODEL `gcp-vertex.gcp_bq.ts_teste`)

SELECT
  *
FROM
  ML.EXPLAIN_FORECAST(MODEL `gcp-vertex.gcp_bq.ts_teste`,
                      STRUCT(30 AS horizon, 0.8 AS confidence_level))

SELECT
 *
FROM
 ML.FORECAST(MODEL `gcp-vertex.gcp_bq.ts_teste`,
             STRUCT(10 AS horizon, 0.8 AS confidence_level))