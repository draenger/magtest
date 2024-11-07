import os
import pandas as pd
from ....util import DataDownloader
from .gsm8k_file_data_loader import GSM8KFileDataLoader
from .gsm8k_db_data_loader import GSM8KDBDataLoader


class GSM8KDataProvider:
    FILE_MAPPINGS = {
        "test": {"standard": "test.jsonl", "socratic": "test_socratic.jsonl"},
        "train": {"standard": "train.jsonl", "socratic": "train_socratic.jsonl"},
        # "example": {"unknown": "example_model_solutions.jsonl"}, this one has diffrent format
    }

    def __init__(
        self,
        gsm8k_question_repository,
        url="https://github.com/openai/grade-school-math/archive/refs/heads/master.zip",
        save_dir="test_data\\gsm8k_data",
    ):
        self.downloader = DataDownloader(url, save_dir)
        self.file_data_loader = GSM8KFileDataLoader()
        self.db_data_loader = GSM8KDBDataLoader(gsm8k_question_repository)

    def process_data(self, max_tests_per_benchmark):
        data = {
            "test": self.__load_data__(max_tests_per_benchmark, "test"),
            "train": self.__load_data__(0, "train"),  # 0 means load all data
            # "example": self.__load_data__(0, "example"),
        }
        return data

    def __load_data__(self, max_tests_per_benchmark=0, data_set="test"):
        data_from_db = self.db_data_loader.load_data(data_set)
        if not data_from_db.empty:
            return self.__filter_data__(data_from_db, max_tests_per_benchmark, data_set)

        all_data = []
        data_dir = self.downloader.process()
        data_dir = os.path.join(data_dir, "grade_school_math", "data").replace(
            "master", "grade-school-math-master"
        )
        print(f"Data directory: {data_dir}")
        for category, filename in self.FILE_MAPPINGS[data_set].items():
            file_path = os.path.join(data_dir, filename)
            print(f"File path: {file_path}")
            if os.path.exists(file_path):
                data = self.file_data_loader.load_data(file_path, category, data_set)
                all_data.append(data)

        if not all_data:
            raise FileNotFoundError(f"No files found for data_set: {data_set}")
        combined_data = pd.concat(all_data, ignore_index=True)

        filtered_data = self.__filter_data__(
            combined_data, max_tests_per_benchmark, data_set
        )

        self.db_data_loader.save_data(filtered_data, data_set)
        return filtered_data

    def __filter_data__(self, data, max_tests_per_benchmark, data_set):
        if max_tests_per_benchmark > 0:
            # Filter keeping category proportions
            filtered_data = (
                data.groupby("category")
                .apply(lambda x: x.sample(n=min(len(x), max_tests_per_benchmark)))
                .reset_index(drop=True)
            )
            print(f"Filtered {data_set} data shape: {filtered_data.shape}")
            return filtered_data
        return data
