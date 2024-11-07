from .repository import Repository
from data.models import BBHQuestion


class BBHQuestionRepository(Repository):
    def __init__(self, database):
        super().__init__(database)
        self.model = BBHQuestion

    def add(
        self,
        question,
        answer,
        category,
        data_type,
        explanation=None,
        helper_text=None,
    ):
        entity = BBHQuestion(
            question=question,
            answer=answer,
            category=category,
            data_type=data_type,
            explanation=explanation,
            helper_text=helper_text,
        )
        try:
            super().add(entity)
            return entity
        except Exception as e:
            print(f"Error adding BBHQuestion: {e}")
            return None

    def get_by_data_type(self, data_type):
        session = self.db.get_session()
        questions = (
            session.query(self.model).filter(self.model.data_type == data_type).all()
        )
        session.close()
        return questions

    def get_by_category(self, category):
        session = self.db.get_session()
        questions = (
            session.query(self.model).filter(self.model.category == category).all()
        )
        session.close()
        return questions
