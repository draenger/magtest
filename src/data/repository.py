class Repository:
    def __init__(self, database):
        self.db = database

    def add(self, entity):
        session = self.db.get_session()
        session.add(entity)
        session.commit()
        session.close()

    def get_by_id(self, id):
        session = self.db.get_session()
        entity = session.query(self.model).filter(self.model.id == id).first()
        session.close()
        return entity

    def get_all(self):
        session = self.db.get_session()
        entities = session.query(self.model).all()
        session.close()
        return entities

    def update(self, id, **kwargs):
        session = self.db.get_session()
        entity = session.query(self.model).filter(self.model.id == id).first()
        if entity:
            for key, value in kwargs.items():
                setattr(entity, key, value)
            session.commit()
        session.close()

    def delete(self, id):
        session = self.db.get_session()
        entity = session.query(self.model).filter(self.model.id == id).first()
        if entity:
            session.delete(entity)
            session.commit()
        session.close()
