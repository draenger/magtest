import os
import pandas as pd
from .mmul_categories import categories, subcategories


class MMULFileDataLoader:
    def process_files(self, directory, file_suffix, max_tests_per_benchmark):
        all_data = []
        files_with_less_records = []
        for file in os.listdir(directory):
            if file.endswith(file_suffix):
                df = self.process_single_file(
                    directory, file, file_suffix, max_tests_per_benchmark
                )
                all_data.append(df)
                if max_tests_per_benchmark > 0 and len(df) < max_tests_per_benchmark:
                    files_with_less_records.append((file, len(df)))
        return all_data, files_with_less_records

    def process_single_file(
        self, directory, file, file_suffix, max_tests_per_benchmark
    ):
        subcategory = file.replace(file_suffix, "")
        df = pd.read_csv(os.path.join(directory, file), header=None)
        df["subcategory"] = subcategory
        df["category"] = self.get_category(subcategory)
        df["group"] = self.get_group(df["category"].iloc[0])
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

    @staticmethod
    def get_category(subcategory):
        return next(
            (v[0] for k, v in subcategories.items() if k == subcategory), "unknown"
        )

    @staticmethod
    def get_group(category):
        return next((k for k, v in categories.items() if category in v), "unknown")
