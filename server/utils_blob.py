"""
azure_blob_manager.py

Brief description:
------------------
This module provides functionality to upload and download files to/from Azure Blob Storage.
It uses the Azure Blob Storage SDK to interact with blob containers defined in the user's Azure account.
Connection information is read from a file named `connection_string` in the current working directory.

Functions:
----------
- upload_file_to_blob(file_path, blob_name, container_name):
    Uploads a file from the local filesystem to a specified blob container in Azure.

- download_file_from_blob(blob_name, download_path, container_name):
    Downloads a file (blob) from Azure Blob Storage and saves it to the local filesystem.

Usage:
------
Run the module as a script to interactively upload or download files:

    $ python azure_blob_manager.py

    > Enter 'upload' to upload a file, 'download' to download a file: upload
    > Choose a container to upload to (['gamevideos', 'submissions']): gamevideos
    > Enter the path of the file to upload: ./video.mp4
    > Enter the name of the blob: match123.mp4
    > File match123.mp4 uploaded successfully.
    > File available at: https://<account>.blob.core.windows.net/gamevideos/match123.mp4

Notes:
------
- Requires a file named `connection_string` containing the Azure Storage connection string.
- The `overwrite=True` option allows replacing existing blobs with the same name.
- The script supports a hardcoded list of containers: ["gamevideos", "submissions"].

Author: Ambrose Ling  
Date: 2025-05-25
"""



from azure.storage.blob import BlobServiceClient
import os

with open("connection_string", "r") as f:
    CONNECTION_STRING = f.read().strip()

blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)

def upload_file_to_blob(file_path, blob_name, container_name):
    try:
        container_client = blob_service_client.get_container_client(container_name)
        
        with open(file_path, "rb") as data:
            container_client.upload_blob(name=blob_name, data=data, overwrite=True)
        
        print(f"File {blob_name} uploaded successfully.")
        return f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"
    except Exception as e:
        print(f"Error uploading file: {e}")
        return None

def download_file_from_blob(blob_name, download_path, container_name):
    try:
        container_client = blob_service_client.get_container_client(container_name)
        
        blob_client = container_client.get_blob_client(blob_name)
        
        with open(download_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        
        print(f"File {blob_name} downloaded successfully to {download_path}.")
    except Exception as e:
        print(f"Error downloading file: {e}")

if __name__ == '__main__':

    response = input("Enter 'upload' to upload a file, 'download' to download a file: ")

    CONTAINER_NAMES = ["gamevideos", "submissions"]

    chosen_container = input(f"Choose a container to upload to ({CONTAINER_NAMES}): ")

    if response == "upload":
        file_path = input("Enter the path of the file to upload: ")
        blob_name = input("Enter the name of the blob: ")
        uploaded_url = upload_file_to_blob(file_path, blob_name, chosen_container)
        if uploaded_url:
            print(f"File available at: {uploaded_url}")
    else:
        blob_name = input("Enter the name of the blob to download: ")
        download_path = input("Enter the path to download the file: ") + blob_name
        download_file_from_blob(blob_name, download_path, chosen_container)