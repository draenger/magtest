from .repository import Repository
from data.models import GSM8KQuestion


class GSM8KQuestionRepository(Repository):
    def __init__(self, database):
        super().__init__(database)
        self.model = GSM8KQuestion

    def add(
        self,
        question,
        full_solution,
        answer,
        category,
        data_type,
    ):
        entity = GSM8KQuestion(
            question=question,
            full_solution=full_solution,
            answer=answer,
            category=category,
            data_type=data_type,
        )
        try:
            super().add(entity)
            return entity
        except Exception as e:
            print(f"Error adding GSM8KQuestion: {e}")
            return None

    def get_by_data_type(self, data_type):
        session = self.db.get_session()
        questions = (
            session.query(self.model).filter(self.model.data_type == data_type).all()
        )
        session.close()
        return questions
