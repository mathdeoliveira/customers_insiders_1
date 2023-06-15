import os
import sys

from flask import jsonify

local_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(local_path, "config")
utils_path = os.path.join(local_path, "customers_insiders_1/utils")
sys.path.append(config_path)
sys.path.append(utils_path)

from config.config import PIPELINE_NAME

from google.cloud.aiplatform import pipeline_jobs


def run_pipeline_clustering(request) -> bool:
    job = pipeline_jobs.PipelineJob(
        display_name="ecommerce-pipeline",
        template_path=f"{PIPELINE_NAME}.json",
        enable_caching=False,
    )
    job.submit()
    return True


def start_clustering(request):
    response = run_pipeline_clustering(request)
    return jsonify({"message": response}), 200
