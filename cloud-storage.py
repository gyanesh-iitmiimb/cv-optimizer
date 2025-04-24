import os
from google.cloud import storage

# Initialize GCS client
def init_storage_client():
    return storage.Client()

# Upload file to GCS bucket
def upload_to_gcs(bucket_name, source_file, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = init_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_file(source_file)
    return f"gs://{bucket_name}/{destination_blob_name}"

# Download file from GCS bucket
def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = init_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)
    return destination_file_name

# Delete file from GCS bucket
def delete_from_gcs(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    storage_client = init_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()

# Add these functions to your Streamlit app and modify file upload sections
# to use GCS for temporary storage of uploaded files

