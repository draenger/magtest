import os
import pandas as pd
from ....util import DataDownloader
from .bbh_file_data_loader import BBHFileDataLoader
from .bbh_db_data_loader import BBHDBDataLoader


class BBHDataProvider:
    FILE_MAPPINGS = {
        "test": {"dir": "bbh", "ext": ".json"},
        "icl": {"dir": "cot-prompts", "ext": ".txt"},
    }

    def __init__(
        self,
        bbh_question_repository,
        url="https://github.com/suzgunmirac/BIG-Bench-Hard/archive/refs/heads/main.zip",
        save_dir="test_data/bbh_data",
    ):
        self.downloader = DataDownloader(url, save_dir)
        self.file_data_loader = BBHFileDataLoader()
        self.db_data_loader = BBHDBDataLoader(bbh_question_repository)

    def process_data(self, max_tests_per_benchmark):
        data = {
            "test": self._load_data("test", max_tests_per_benchmark),
            "icl": self._load_data("icl", 0),  # 0 means no limit
        }
        return data

    def _load_data(self, data_type="test", max_tests_per_benchmark=0):

        # Load from files if not in DB
        data_dir = self.downloader.process()
        mapping = self.FILE_MAPPINGS.get(data_type)
        if not mapping:
            raise ValueError(f"Unsupported data type: {data_type}")

        data_dir = os.path.join(data_dir, mapping["dir"]).replace(
            "main", "BIG-Bench-Hard-main"
        )
        file_extension = mapping["ext"]
        all_data = []

        # Find all files with the specified extension
        files = [f for f in os.listdir(data_dir) if f.endswith(file_extension)]

        for file in files:
            file_path = os.path.join(data_dir, file)
            category = os.path.splitext(file)[0]  # Remove file extension
            try:
                data = self.file_data_loader.load_data(
                    file_path, category=category, data_type=data_type
                )
                all_data.append(data)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")

        if not all_data:
            raise FileNotFoundError(f"No BBH data files found in {data_dir}")

        combined_data = pd.concat(all_data, ignore_index=True)
        filtered_data = self._filter_data(combined_data, max_tests_per_benchmark)

        # Save to DB
        self.db_data_loader.save_data(filtered_data, data_type)

        return filtered_data

    def _filter_data(self, data, max_tests_per_benchmark):
        if max_tests_per_benchmark > 0:
            filtered_data = (
                data.groupby("category")
                .apply(lambda x: x.sample(n=min(len(x), max_tests_per_benchmark)))
                .reset_index(drop=True)
            )
            print(f"Filtered data shape: {filtered_data.shape}")
            return filtered_data
        return data
