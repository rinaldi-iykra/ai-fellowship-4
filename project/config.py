# Configuration for the app
import os
from google.cloud import bigquery, secretmanager, storage
from io import StringIO, BytesIO

project_id = 'b-508911'

def access_secret(secret_name):
    client = secretmanager.SecretManagerServiceClient()
    secret_path = f'projects/{project_id}/secrets/{secret_name}/versions/latest'
    response = client.access_secret_version(request={'name':secret_path})

    return response.payload.data.decode('UTF-8')

# Retrieve values from the environment variables
dataset_id = access_secret('DATASET_ID')
test_table_name = access_secret('TEST_TABLE_NAME')
cost_table_name = access_secret('COST_TABLE_NAME')
view_name = access_secret('VIEW_NAME')
bucket_name = access_secret('BUCKET_NAME')
open_ai_key = access_secret('OPEN_AI_API_KEY')

test_table_path = f'{project_id}.{dataset_id}.{test_table_name}'
cost_table_path = f'{project_id}.{dataset_id}.{cost_table_name}'

# Initialize the BigQuery and GCS client
client = bigquery.Client()
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

# Define function to upload to GCS
def upload_to_gcs(df, blob_name):
    """
    Upload dataframe given in argument into Google Cloud Storage
    Args:
        df: pd.DataFrame
        blob_name: destination file name in GCS
    """
    # Define the bucket and blob (file) name
    blob = bucket.blob(blob_name)

    # Save the DataFrame to a CSV file in Google Cloud Storage using a buffer
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    # Upload the buffer to Google Cloud Storage with a timeout
    blob.upload_from_file(buffer, content_type='text/csv', timeout=600)

def download_from_gcs(blob_name):
    """
    Download a file from Google Cloud Storage and return its content.
    Args:
        blob_name: destination file name in GCS
    Return:
        content: BytesIO(content)
    """
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_bytes()
    return BytesIO(content)

def download_and_save_csv(gcs_path, local_path):
    try:
        df = pd.read_csv(download_from_gcs(gcs_path))
        df.to_csv(local_path, index=False)
        print(f"Downloaded and saved {local_path}")
    except Exception as e:
        print(f"Failed to download or save {local_path}: {e}")