class MMLUDataFilterer:
    @staticmethod
    def filter_data(data, max_tests_per_benchmark, data_set):
        if max_tests_per_benchmark > 0:
            filtered_data = (
                data.groupby("subcategory")
                .apply(lambda x: x.sample(min(len(x), max_tests_per_benchmark)))
                .reset_index(drop=True)
            )
            print(f"Filtered {data_set} data shape: {filtered_data.shape}")
            return filtered_data
        return data
