import os
import requests
import tarfile
import zipfile
from pathlib import Path


class DataDownloader:
    def __init__(self, url, save_dir):
        self.url = url
        self.save_dir = save_dir
        self.file_name = self.url.split("/")[-1]
        self.file_name_no_ext = Path(self.file_name).stem
        self.archive_file = os.path.join(self.save_dir, self.file_name)
        self.unpacked_file_dir = os.path.join(self.save_dir, self.file_name_no_ext)

    def download_data(self):
        if os.path.exists(self.archive_file):
            print("Data file already exists. Skipping download.")
            return

        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Downloading data from {self.url}")

        response = requests.get(self.url, stream=True)
        if response.status_code == 200:
            with open(self.archive_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download completed.")
        else:
            print(f"Failed to download data. Status code: {response.status_code}")

    def _is_tar_file(self):
        """Check if the file is a tar archive"""
        return tarfile.is_tarfile(self.archive_file)

    def _is_zip_file(self):
        """Check if the file is a zip archive"""
        return zipfile.is_zipfile(self.archive_file)

    def extract_data(self):
        if os.path.exists(self.unpacked_file_dir):
            print("Data directory already exists. Skipping extraction.")
            return

        if not os.path.exists(self.archive_file):
            print("Archive file not found. Please download the data first.")
            return

        print("Extracting data...")
        try:
            if self._is_tar_file():
                self._extract_tar()
            elif self._is_zip_file():
                self._extract_zip()
            else:
                print(f"Unsupported archive format for file: {self.archive_file}")
                return
            print("Extraction completed.")
        except Exception as e:
            print(f"Error during extraction: {str(e)}")

    def _extract_tar(self):
        """Extract tar archive"""
        with tarfile.open(self.archive_file, "r:*") as tar:
            tar.extractall(path=self.save_dir)

    def _extract_zip(self):
        """Extract zip archive"""
        with zipfile.ZipFile(self.archive_file, "r") as zip_ref:
            zip_ref.extractall(self.save_dir)

    def process(self):
        """Download and extract the data"""
        self.download_data()
        self.extract_data()
        return self.unpacked_file_dir
