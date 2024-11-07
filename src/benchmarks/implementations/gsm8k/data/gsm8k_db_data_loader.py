import pandas as pd


class GSM8KDBDataLoader:
    def __init__(self, repository):
        self.repository = repository

    def load_data(self, data_set):
        questions = self.repository.get_by_data_type(data_set)
        if questions:
            return pd.DataFrame(
                [
                    {
                        "question": q.question,
                        "full_solution": q.full_solution,
                        "answer": q.answer,
                        "category": q.category,
                    }
                    for q in questions
                ]
            )
        return pd.DataFrame()

    def save_data(self, data, data_set):
        for _, row in data.iterrows():
            self.repository.add(
                question=row["question"],
                full_solution=row["full_solution"],
                answer=row["answer"],
                category=row["category"],
                data_type=data_set,
            )
