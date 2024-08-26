import os
import pandas as pd
from benchmarks.util.data_downloader import DataDownloader
from .categories import categories, subcategories


class MMULDataLoader:
    def __init__(
        self,
        url="https://people.eecs.berkeley.edu/~hendrycks/data.tar",
        save_dir="test_data/mmlu_data",
        data_set_dir="test",
    ):
        self.downloader = DataDownloader(url, save_dir)
        self.unpacked_file_dir = None
        self.sub_dir = None
        self.sub_dir_name = data_set_dir

    def process_data(self, max_tests_per_benchmark):
        test_data = self.__load_data__(max_tests_per_benchmark, data_set="test")
        dev_data = self.__load_data__(max_tests_per_benchmark=0, data_set="dev")
        return {"test": test_data, "dev": dev_data}

    def __load_data__(self, max_tests_per_benchmark=0, data_set="test"):
        self.__ensure_data_directory__(data_set)
        file_suffix = f"_{data_set}.csv"
        all_data, files_with_less_records = self.__process_files__(
            file_suffix, max_tests_per_benchmark
        )
        combined_data = self.__combine_data__(all_data, data_set)
        self.__print_files_with_less_records__(
            files_with_less_records, max_tests_per_benchmark, data_set
        )
        filtered_data = self.__filter_data__(
            combined_data, max_tests_per_benchmark, data_set
        )
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

    def __process_files__(self, file_suffix, max_tests_per_benchmark):
        all_data = []
        files_with_less_records = []
        for file in os.listdir(self.sub_dir):
            if file.endswith(file_suffix):
                df = self.__process_single_file__(
                    file, file_suffix, max_tests_per_benchmark
                )
                all_data.append(df)
                if max_tests_per_benchmark > 0 and len(df) < max_tests_per_benchmark:
                    files_with_less_records.append((file, len(df)))
        return all_data, files_with_less_records

    def __process_single_file__(self, file, file_suffix, max_tests_per_benchmark):
        subcategory = file.replace(file_suffix, "")
        df = pd.read_csv(os.path.join(self.sub_dir, file), header=None)
        df["subcategory"] = subcategory
        df["category"] = self.__get_category__(subcategory)
        df["group"] = self.__get_group__(df["category"].iloc[0])
        df.columns = [
            "question",
            "A",
            "B",
            "C",
            "D",
            "answer",
            "subcategory",
            "category",
            "group",
        ]
        return df

    def __get_category__(self, subcategory):
        return next(
            (v[0] for k, v in subcategories.items() if k == subcategory), "unknown"
        )

    def __get_group__(self, category):
        return next((k for k, v in categories.items() if category in v), "unknown")

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

    def __filter_data__(self, combined_data, max_tests_per_benchmark, data_set):
        if max_tests_per_benchmark > 0:
            filtered_data = (
                combined_data.groupby("subcategory")
                .apply(lambda x: x.sample(min(len(x), max_tests_per_benchmark)))
                .reset_index(drop=True)
            )
            print(f"Filtered {data_set} data shape: {filtered_data.shape}")
            return filtered_data
        return combined_data
