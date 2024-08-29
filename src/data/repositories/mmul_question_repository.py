from .repository import Repository
from data.models import MMULQuestion


class MMULQuestionRepository(Repository):
    def __init__(self, database):
        super().__init__(database)
        self.model = MMULQuestion

    def add(
        self,
        question,
        option_a,
        option_b,
        option_c,
        option_d,
        answer,
        subcategory,
        category,
        group,
        data_type,
    ):
        entity = MMULQuestion(
            question=question,
            option_a=option_a,
            option_b=option_b,
            option_c=option_c,
            option_d=option_d,
            answer=answer,
            subcategory=subcategory,
            category=category,
            group=group,
            data_type=data_type,
        )
        try:
            super().add(entity)
            return entity
        except Exception as e:
            print(f"Error adding MMULQuestion: {e}")
            return None

    def get_by_category(self, category):
        session = self.db.get_session()
        questions = (
            session.query(self.model).filter(self.model.category == category).all()
        )
        session.close()
        return questions

    def get_by_subcategory(self, subcategory):
        session = self.db.get_session()
        questions = (
            session.query(self.model)
            .filter(self.model.subcategory == subcategory)
            .all()
        )
        session.close()
        return questions

    def get_by_group(self, group):
        session = self.db.get_session()
        questions = session.query(self.model).filter(self.model.group == group).all()
        session.close()
        return questions

    def get_by_data_type(self, data_type):
        session = self.db.get_session()
        questions = (
            session.query(self.model).filter(self.model.data_type == data_type).all()
        )
        session.close()
        return questions
