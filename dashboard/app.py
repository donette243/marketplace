import os
import re
import sqlite3
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Marketplace AI", layout="wide")
st.title(" Анализ отзывов (Marketplace AI)")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(BASE_DIR, "reviews.db")

conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT * FROM review", conn)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\sа-яА-ЯёЁ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def get_sentiment(rating):
    if rating >= 4:
        return "положительный"
    elif rating <= 2:
        return "отрицательный"
    return "нейтральный"

def detect_topic(text):
    text = text.lower()
    if "достав" in text:
        return "доставка"
    elif "слом" in text or "работ" in text:
        return "проблема_с_товаром"
    elif "звук" in text or "качеств" in text:
        return "качество_товара"
    return "другое"

def detect_review_type(text, rating):
    text = text.lower()

    if "но" in text or "однако" in text:
        return "mixed"
    elif rating >= 4:
        return "positive"
    elif rating <= 2:
        return "negative"
    else:
        return "neutral"

def detect_aspects(text):
    text = text.lower()
    aspects = []

    if "достав" in text:
        aspects.append("доставка")
    if "слом" in text or "работ" in text:
        aspects.append("проблема_с_товаром")
    if "звук" in text or "качеств" in text:
        aspects.append("качество_товара")

    if not aspects:
        aspects.append("другое")

    return aspects

knowledge_base = {
    "доставка": [
    "При задержке доставки мы приносим извинения за неудобства.",
    "Иногда доставка может занять больше времени, чем ожидалось."
],
    "проблема_с_товаром": [
        "Возврат товара возможен в течение 14 дней после получения заказа.",
        "Если товар пришёл повреждённым или сломанным, клиент может оформить возврат или обмен.",
        "Если зарядное устройство перестало работать в первые дни использования, доступен возврат или замена товара.",
        "Если клиент получил повреждённый товар, возможен возврат средств или замена."
    ],
    "качество_товара": [
        "Спасибо за положительную оценку качества товара."
    ],
    "другое": [
        "Спасибо за ваш отзыв."
    ]
}

@st.cache_resource
def load_embedder():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

embedder = load_embedder()

@st.cache_resource
def build_kb_embeddings():
    return {
        topic: embedder.encode(texts)
        for topic, texts in knowledge_base.items()
    }

kb_embeddings = build_kb_embeddings()

def retrieve_context_with_threshold(query, topic, threshold=0.3):
    texts = knowledge_base.get(topic, knowledge_base["другое"])
    embeddings = kb_embeddings.get(topic, kb_embeddings["другое"])

    query_embedding = embedder.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    best_context = texts[best_idx]

    if best_score >= threshold:
        return best_context

    return None

def filter_context(context):
    if context is None:
        return None

    bad_words = ["можно", "рекомендуем", "следует"]

    if any(word in context.lower() for word in bad_words):
        return None

    return context
def generate_smart_response(review, rating):
    review_type = detect_review_type(review, rating)
    aspects = detect_aspects(review)

    if review_type == "positive":
        return "Спасибо за ваш отзыв! Мы рады, что вам понравился товар."

    elif review_type == "negative":
        if "проблема_с_товаром" in aspects:
            return "Нам очень жаль, что вы столкнулись с проблемой с товаром. Вы можете оформить возврат или обмен."
        elif "доставка" in aspects:
            return "Нам очень жаль, что доставка не оправдала ожиданий. Мы обязательно учтём ваш отзыв."
        else:
            return "Нам очень жаль, что у вас остались негативные впечатления."

    elif review_type == "mixed":
        parts = []

        if "качество_товара" in aspects:
            parts.append("Спасибо за отзыв! Мы рады, что вам понравилось качество товара.")

        if "доставка" in aspects:
            parts.append("При этом нам жаль, что доставка заняла больше времени, чем ожидалось.")

        if "проблема_с_товаром" in aspects:
            parts.append("Нам также жаль, что вы столкнулись с проблемой с товаром.")

        if not parts:
            parts.append("Спасибо за ваш отзыв! Мы обязательно учтём ваши замечания.")

        return " ".join(parts)

    return "Спасибо за ваш отзыв!"

def generate_better_rag_response(review, rating, topic):
    base_response = generate_smart_response(review, rating)
    review_type = detect_review_type(review, rating)

    if review_type == "positive":
        return base_response

    context = retrieve_context_with_threshold(review, topic, threshold=0.3)
    context = filter_context(context)

    if context is None:
        return base_response

    if context in base_response:
        return base_response

    return f"{base_response} {context}"

df["clean_text"] = df["review_text"].apply(clean_text)
df["sentiment_label"] = df["rating"].apply(get_sentiment)
df["topic_label"] = df["clean_text"].apply(detect_topic)

df["generated_response"] = df.apply(
    lambda row: generate_smart_response(row["review_text"], row["rating"]),
    axis=1
)

df["rag_response"] = df.apply(
    lambda row: generate_better_rag_response(
        row["review_text"],
        row["rating"],
        row["topic_label"]
    ),
    axis=1
)

# METRICS 
col1, col2, col3 = st.columns(3)
col1.metric("Всего отзывов", len(df))
col2.metric("Положительных", (df["sentiment_label"] == "положительный").sum())
col3.metric("Отрицательных", (df["sentiment_label"] == "отрицательный").sum())

st.subheader("Данные")
st.dataframe(df)

# CHARTS 
st.subheader("Тональность")
st.bar_chart(df["sentiment_label"].value_counts())

st.subheader("Темы")
st.bar_chart(df["topic_label"].value_counts())

st.subheader("Примеры ответов")

for _, row in df.iterrows():
    st.markdown("---")
    st.write(f"**Отзыв:** {row['review_text']}")
    st.write(f"**Тональность:** {row['sentiment_label']}")
    st.write(f"**Тема:** {row['topic_label']}")
    st.write(f"**Обычный ответ:** {row['generated_response']}")
    st.write(f"**RAG-ответ:** {row['rag_response']}")