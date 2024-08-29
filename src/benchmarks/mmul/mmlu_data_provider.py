import os
import pandas as pd
from benchmarks.util.data_downloader import DataDownloader
from .mmul_file_data_loader import MMULFileDataLoader
from .mmul_db_data_loader import MMULDBDataLoader
from .data_filterer import DataFilterer


class MMULDataProvider:
    def __init__(
        self,
        database,
        url="https://people.eecs.berkeley.edu/~hendrycks/data.tar",
        save_dir="test_data/mmlu_data",
        data_set_dir="test",
    ):
        self.unpacked_file_dir = None
        self.sub_dir = None
        self.sub_dir_name = data_set_dir
        self.downloader = DataDownloader(url, save_dir)
        self.file_data_provider = MMULFileDataLoader()
        self.db_data_loader = MMULDBDataLoader(database)
        self.data_filterer = DataFilterer()

    def process_data(self, max_tests_per_benchmark):
        test_data = self.__load_data__(max_tests_per_benchmark, data_set="test")
        dev_data = self.__load_data__(max_tests_per_benchmark=0, data_set="dev")
        return {"test": test_data, "dev": dev_data}

    def __load_data__(self, max_tests_per_benchmark=0, data_set="test"):
        data_from_db = self.db_data_loader.load_data(data_set)
        if not data_from_db.empty:
            return self.data_filterer.filter_data(
                data_from_db, max_tests_per_benchmark, data_set
            )

        self.__ensure_data_directory__(data_set)
        file_suffix = f"_{data_set}.csv"
        all_data, files_with_less_records = self.file_data_provider.process_files(
            self.sub_dir, file_suffix, max_tests_per_benchmark
        )
        combined_data = self.__combine_data__(all_data, data_set)
        self.__print_files_with_less_records__(
            files_with_less_records, max_tests_per_benchmark, data_set
        )
        filtered_data = self.data_filterer.filter_data(
            combined_data, max_tests_per_benchmark, data_set
        )
        self.db_data_loader.save_data(filtered_data, data_set)
        return filtered_data

    def __ensure_data_directory__(self, data_set):
        if not self.unpacked_file_dir:
            self.unpacked_file_dir = self.downloader.process()
        self.sub_dir = os.path.join(self.unpacked_file_dir, data_set)
        print(f"Data directory: {self.sub_dir}")
        if not os.path.exists(self.sub_dir):
            raise FileNotFoundError(
                f"{data_set.capitalize()} data directory not found. Please extract the data first."
            )

    def __combine_data__(self, all_data, data_set):
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Combined {data_set} data shape: {combined_data.shape}")
        return combined_data

    def __print_files_with_less_records__(
        self, files_with_less_records, max_tests_per_benchmark, data_set
    ):
        if files_with_less_records:
            print(
                f"\n{data_set.capitalize()} files record counts (max {max_tests_per_benchmark}):"
            )
            for file, count in files_with_less_records:
                print(f"{file}: {count} records")

            avg_count = sum(count for _, count in files_with_less_records) / len(
                files_with_less_records
            )
            print(f"\nAverage record count: {avg_count:.2f}")
            print(f"Total files: {len(files_with_less_records)}")
            print()
