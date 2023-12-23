import os
import shutil
from torchvision.datasets.utils import download_file_from_google_drive, extract_archive

if __name__ == "__main__":
    # Define the unique identifier for the file on Google Drive
    # file url: https://drive.google.com/file/d/1dm_djCERttnHEpSAZAQFsK6Z7iV00ivQ/
    file_id = '1dm_djCERttnHEpSAZAQFsK6Z7iV00ivQ'

    # Define the root folder to store downloaded and extracted data
    root_folder = 'datasets/'

    # Define the name of the zip file to be downloaded
    file_name = "digital_twins.zip"

    # Download the file from Google Drive using its ID
    download_file_from_google_drive(file_id=file_id, root=root_folder, filename=file_name)

    # Create the complete path to the downloaded zip file
    file_path = os.path.join(root_folder, file_name)

    # Extract the contents of the downloaded zip file to the root folder
    extract_archive(from_path=file_path, to_path=root_folder, remove_finished=True)

    # Remove system-related metadata (if any) generated during extraction
    shutil.rmtree(os.path.join(root_folder, "__MACOSX"))
