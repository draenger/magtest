import os
import requests
import tarfile
import pandas as pd


class MMULDataLoader:
    def __init__(
        self,
        url="https://people.eecs.berkeley.edu/~hendrycks/data.tar",
        save_dir="test_data/mmlu_data",
        data_set_dir="test",
    ):

        self.url = url
        file_name = self.url.split("/")[-1]
        file_name_no_ext = file_name.split(".")[0]
        self.save_dir = save_dir
        self.tar_file = os.path.join(self.save_dir, file_name)
        self.unpacked_file_dir = os.path.join(self.save_dir, file_name_no_ext)
        self.sub_dir = os.path.join(self.unpacked_file_dir, data_set_dir)
        self.sub_dir_name = data_set_dir

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

    def load_data(self, max_tests_per_benchmark=1):
        if not os.path.exists(self.sub_dir):
            print("Test data directory not found. Please extract the data first.")
            return None

        file_suffix = f"_{self.sub_dir_name}.csv"
        all_data = []
        for file in os.listdir(self.sub_dir):
            if file.endswith(file_suffix):
                category = file.replace(file_suffix, "")
                df = pd.read_csv(os.path.join(self.sub_dir, file))
                df["category"] = category
                df.columns = ["question", "A", "B", "C", "D", "answer", "category"]
                all_data.append(df)

        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Combined data shape: {combined_data.shape}")

        return combined_data

    def process_data(self, max_tests_per_benchmark):
        self.download_data()
        self.extract_data()
        data = self.load_data(max_tests_per_benchmark)
        if max_tests_per_benchmark > 0:  # 0 is a special case that means no limit
            data = data.head(max_tests_per_benchmark)
            print(f"Filtered Combined data shape: {data.shape}")
        return data
