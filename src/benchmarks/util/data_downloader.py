import os
import requests
import tarfile


class DataDownloader:
    def __init__(self, url, save_dir):
        self.url = url
        self.save_dir = save_dir
        self.file_name = self.url.split("/")[-1]
        self.file_name_no_ext = self.file_name.split(".")[0]
        self.tar_file = os.path.join(self.save_dir, self.file_name)
        self.unpacked_file_dir = os.path.join(self.save_dir, self.file_name_no_ext)

    def download_data(self):
        if os.path.exists(self.tar_file):
            print("Data file already exists. Skipping download.")
            return

        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Downloading data from {self.url}")

        response = requests.get(self.url, stream=True)
        if response.status_code == 200:
            with open(self.tar_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download completed.")
        else:
            print(f"Failed to download data. Status code: {response.status_code}")

    def extract_data(self):
        if os.path.exists(self.unpacked_file_dir):
            print("Data directory already exists. Skipping extraction.")
            return

        if not os.path.exists(self.tar_file):
            print("Tar file not found. Please download the data first.")
            return

        print("Extracting data...")
        with tarfile.open(self.tar_file, "r") as tar:
            tar.extractall(path=self.save_dir)
        print("Extraction completed.")

    def process(self):
        self.download_data()
        self.extract_data()
        return self.unpacked_file_dir
