import json

from google.cloud import storage


def read_gcs_file(bucket_name: str, file_name: str, project_name: str) -> json:
    """Reads the content of a file from Google Cloud Storage (GCS) bucket.

    Args:
        bucket_name (str): Name of the GCS bucket.
        file_name (str): Name of the file within the bucket.

    Returns:
        str: Content of the file.

    Raises:
        google.cloud.exceptions.NotFound: If the specified bucket or file is not found.
    """
    # Create a storage client
    storage_client = storage.Client(project=project_name)

    # Retrieve the bucket and file object
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.get_blob(file_name)

    # Download the file content as a string
    file_path = f"/tmp/{file_name}"
    blob.download_to_filename(file_path)
    return file_path
