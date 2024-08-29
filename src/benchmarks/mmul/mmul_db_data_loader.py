import pandas as pd


class MMULDBDataLoader:
    def __init__(self, repository):
        self.repository = repository

    def load_data(self, data_set):
        questions = self.repository.get_by_data_type(data_set)
        if questions:
            return pd.DataFrame(
                [
                    {
                        "question": q.question,
                        "A": q.option_a,
                        "B": q.option_b,
                        "C": q.option_c,
                        "D": q.option_d,
                        "answer": q.answer,
                        "subcategory": q.subcategory,
                        "category": q.category,
                        "group": q.group,
                    }
                    for q in questions
                ]
            )
        return pd.DataFrame()

    def save_data(self, data, data_set):
        for _, row in data.iterrows():
            self.repository.add(
                question=row["question"],
                option_a=row["A"],
                option_b=row["B"],
                option_c=row["C"],
                option_d=row["D"],
                answer=row["answer"],
                subcategory=row["subcategory"],
                category=row["category"],
                group=row["group"],
                data_type=data_set,
            )
