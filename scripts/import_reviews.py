import pandas as pd
from datetime import datetime
from sqlmodel import Session

from app.db.database import engine
from app.models.review import Review

CSV_PATH = "data/raw/reviews_sample.csv"


def parse_date(value):
    if pd.isna(value) or value == "":
        return None
    return datetime.strptime(str(value), "%Y-%m-%d")


def main():
    df = pd.read_csv(CSV_PATH)

    reviews = []
    for _, row in df.iterrows():
        review = Review(
            source=str(row.get("source", "")),
            product_name=row.get("product_name"),
            rating=int(row["rating"]) if pd.notna(row.get("rating")) else None,
            review_title=row.get("review_title"),
            review_text=str(row.get("review_text", "")),
            review_date=parse_date(row.get("review_date")),
            review_language=row.get("review_language"),
        )
        reviews.append(review)

    with Session(engine) as session:
        session.add_all(reviews)
        session.commit()

    print(f"{len(reviews)} reviews imported successfully")


if __name__ == "__main__":
    main()