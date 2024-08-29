from data import Database, ModelResultRepository


def get_database():
    db = Database()
    db.create_all_tables()
    return db


def get_model_result_repository():
    db = get_database()
    return ModelResultRepository(db)
