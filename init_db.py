from app.db.database import create_db_and_tables
from app.models.review import Review

if __name__ == "__main__":
    create_db_and_tables()
    print("Database created")