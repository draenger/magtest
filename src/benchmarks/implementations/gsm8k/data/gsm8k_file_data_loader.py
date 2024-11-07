import pandas as pd
import json


class GSM8KFileDataLoader:
    def load_data(self, file_path: str, category: str, data_type: str) -> pd.DataFrame:
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                solution_parts = item["answer"].split(
                    "####"
                )  # Part after #### is answer
                full_solution = solution_parts[0].strip()
                answer = solution_parts[1].strip()

                data.append(
                    {
                        "question": item["question"],
                        "full_solution": full_solution,
                        "answer": answer,
                        "category": category,
                        "data_type": data_type,
                    }
                )
        return pd.DataFrame(data)
