from .repository import Repository
from data.models import PreparedQuestion


class PreparedQuestionRepository(Repository):
    def __init__(self, database):
        super().__init__(database)
        self.model = PreparedQuestion

    def add(self, **kwargs):
        entity = PreparedQuestion(**kwargs)
        try:
            session = self.db.get_session()
            session.add(entity)
            session.commit()
            session.close()
            return entity
        except Exception as e:
            session.rollback()
            print(f"Error adding PreparedQuestion: {e}")
            return None
        finally:
            session.close()

    def get_by_test_session(self, test_session_id):
        session = self.db.get_session()
        questions = (
            session.query(self.model)
            .filter(self.model.test_session_id == test_session_id)
            .all()
        )
        session.close()
        return questions
