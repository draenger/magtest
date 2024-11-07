import pandas as pd


class BBHDBDataLoader:
    def __init__(self, repository):
        self.repository = repository

    def load_data(self, data_type):
        questions = self.repository.get_by_data_type(data_type)
        if questions:
            return pd.DataFrame(
                [
                    {
                        "question": q.question,
                        "answer": q.answer,
                        "explanation": q.explanation,
                        "helper_text": q.helper_text,
                        "category": q.category,
                        "data_type": q.data_type,
                    }
                    for q in questions
                ]
            )
        return pd.DataFrame()

    def save_data(self, data, data_type):
        for _, row in data.iterrows():
            self.repository.add(
                question=row["question"],
                answer=row["answer"],
                category=row["category"],
                data_type=data_type,
                explanation=row.get("explanation", None),
                helper_text=row.get("helper_text", None),
            )
