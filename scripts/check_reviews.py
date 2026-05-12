from sqlmodel import Session, select

from app.db.database import engine
from app.models.review import Review


def main():
    with Session(engine) as session:
        reviews = session.exec(select(Review)).all()

        print(f"Total reviews: {len(reviews)}")
        for review in reviews:
            print("-" * 40)
            print(f"ID: {review.id}")
            print(f"Source: {review.source}")
            print(f"Product: {review.product_name}")
            print(f"Rating: {review.rating}")
            print(f"Title: {review.review_title}")
            print(f"Text: {review.review_text}")
            print(f"Language: {review.review_language}")


if __name__ == "__main__":
    main()