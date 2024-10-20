import os
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import CacheBackedEmbeddings
from langchain.schema import Document
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
import os
import google.generativeai as genai
from dotenv import load_dotenv
import sqlite3


load_dotenv()


model = "llama2"


st.title("jeju!")

embeddings = OllamaEmbeddings(model="llama3.1:latest")

api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

generation_config = {
    "temperature": 0.05,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}


if model == "llama":
    llm = ChatOllama(model="llama3.1:latest", temperature=0.1, streaming=True)
else:
    llm = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )
    chat_session = llm.start_chat(
        history=[
        ]
    )


theme_retrievers = {}
food_type_file_retrievers = {}

MD_DIR = r"./.cache/data"


FOOD_TYPE_FILES = {
    "FOODTYPE_NAVER": [
        "test1.txt"
    ],
}


conn = sqlite3.connect('./jeju.db')
cursor = conn.cursor()


def embed_file(file_path: str):
    cache_dir = f"./.cache/embeddings/{os.path.basename(file_path)}"
    if os.path.exists(cache_dir):
        cache_store = LocalFileStore(cache_dir)
        embeddings = OllamaEmbeddings()
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, cache_store)
        docs = [Document(page_content=open(
            file_path, "r", encoding='utf-8').read())]
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        return vectorstore.as_retriever()
    else:
        with open(file_path, "r", encoding='utf-8') as file:
            file_content = file.read()
        cache_store = LocalFileStore(cache_dir)
        docs = [Document(page_content=file_content)]
        embeddings = OllamaEmbeddings()
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, cache_store)
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        return vectorstore.as_retriever()


def format_docs(docs):
    combined_docs = "\n\n".join(doc.page_content for doc in docs)
    return combined_docs


def recommend(restaurant_type=None):
    # 데이터베이스 연결
    conn = sqlite3.connect('./jeju.db')
    cursor = conn.cursor()

    # 기본 쿼리
    query = """WITH MaxVisitCounts AS (
    SELECT 
        r.placeID,
        MAX(r.visit_count) AS max_visit_count
    FROM 
        Review r
    WHERE 
        r.visit_count >= 2
    GROUP BY 
        r.placeID
),
TotalReviewCounts AS (
    SELECT 
        r.placeID,
        COUNT(*) AS total_review_count
    FROM 
        Review r
    GROUP BY 
        r.placeID
),
AggregatedData AS (
    SELECT 
        i.placeID,
        i.MCT_NM,
        i.UE_AMT_GRP,
        COALESCE(SUM(m.max_visit_count), 0) AS total_visit_count,
        COALESCE(trc.total_review_count, 0) AS total_review_count,
        i.MCT_TYPE
    FROM 
        Information i
    LEFT JOIN 
        MaxVisitCounts m ON i.placeID = m.placeID
    LEFT JOIN 
        TotalReviewCounts trc ON i.placeID = trc.placeID
    GROUP BY 
        i.placeID, i.MCT_NM, i.UE_AMT_GRP, i.MCT_TYPE
)
SELECT 
    a.placeID,
    a.MCT_NM,
    a.UE_AMT_GRP,
    (6 * (6 - a.UE_AMT_GRP) + 4 * (a.total_visit_count / NULLIF(a.total_review_count, 0))) AS score
FROM 
    AggregatedData a

"""

    if restaurant_type:
        query += f"WHERE a.MCT_TYPE = ?"

    query += " ORDER BY score DESC LIMIT 10;"

    if restaurant_type:
        cursor.execute(query, (restaurant_type,))
    else:
        cursor.execute(query)

    results = cursor.fetchall()

    for row in results:
        print(row)

    cursor.close()
    conn.close()

    return results


with open('.cache/data/test1.txt', 'r', encoding='utf-8') as file:
    context = file.read()


question = st.text_input("question111")

if question:

    try:
        response = chat_session.send_message(f"""
글: {question} 

위 글에서 나타나는 음식 카테고리는 뭐야? 음식의 카테고리만 말하고, 그 이외의 것은 절대 말하면 안돼. 만약 제주도 향토 음식이라면, "향토"라는 답변을 해야돼. 해당 질문에 음식 카테고리가 없다면 "없음"을 답해야 해.

음식 카테고리: {context}
""")
        print(response.text)

        print(response)
        if isinstance(response, dict):
            st.success(response)
        else:
            st.success(str(response.text))

        results = recommend('피자')  # 원래는 response.text를 넣어야 되겠죠..?
        print(results)
        st.success(results)

    except Exception as e:
        st.error(f"체인 try except 부분, {str(e)}")

    is_first_question = False
    question = None
    print("=======================================================")
    print(is_first_question, question)

    question2 = st.text_input("question222")

    if question2:

        try:
            response = chat_session.send_message(question2)
            print(response.text)

            print(response)
            if isinstance(response, dict):
                st.success(response)
            else:
                st.success(str(response.text))
        except Exception as e:
            st.error(f"체인 try except 부분, {str(e)}")

        # 첫 번째 질문 말고, 이후에 이어서 질문하는 거....
        print(1)
