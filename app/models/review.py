from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field


class Review(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    source: str
    product_name: Optional[str] = None
    rating: Optional[int] = None
    review_title: Optional[str] = None
    review_text: str
    review_date: Optional[datetime] = None
    review_language: Optional[str] = None
    sentiment_label: Optional[str] = None
    topic_label: Optional[str] = None
    generated_response: Optional[str] = None