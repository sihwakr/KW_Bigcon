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

st.title("ga111o!")

embeddings = OllamaEmbeddings(model="llama3.1:latest")
llm = ChatOllama(model="llama3.1:latest", temperature=0.1, streaming=True)


theme_retrievers = {}


MD_DIR = r"./.cache/data"


THEME_FILES = {
    "ttest": ["test1.md"],
    "ttest2": [
        "test2.md",
        "test2-1.md"
    ],
}


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


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the user's question in one line

            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

theme = st.selectbox("theme?", list(THEME_FILES.keys()))

question = st.text_input("question!")


if st.button("테마 선택"):
    if theme in THEME_FILES:
        combined_content = ""
        for file_name in THEME_FILES[theme]:
            file_path = os.path.join(MD_DIR, file_name)
            try:
                with open(file_path, "r", encoding='utf-8') as file:
                    combined_content += file.read() + "\n\n"
            except FileNotFoundError:
                st.error(f"FileNotFoundError {file_name}")
                continue
            except Exception as e:
                st.error(f"{file_name} - {str(e)}")
                continue

        cache_dir = f"./.cache/embeddings/{theme}"
        os.makedirs(cache_dir, exist_ok=True)

        docs = [Document(page_content=combined_content)]
        vectorstore = FAISS.from_documents(docs, embeddings)
        theme_retrievers[theme] = vectorstore.as_retriever()
        st.session_state.selected_theme = theme
        st.success(f"{theme} selected!")
    else:
        st.error(f"theme in THEME_FILES {theme}, {THEME_FILES}")

    if theme in THEME_FILES:
        combined_content = ""
        for file_name in THEME_FILES[theme]:
            file_path = os.path.join(MD_DIR, file_name)
            try:
                with open(file_path, "r", encoding='utf-8') as file:
                    combined_content += file.read() + "\n\n"
            except FileNotFoundError:
                st.error(f"FileNotFoundError: {file_name}")
                continue
            except Exception as e:
                st.error(f"{file_name} - {str(e)}")
                continue

        cache_dir = f"./.cache/embeddings/{theme}"
        cache_store = LocalFileStore(cache_dir)
        embeddings = OllamaEmbeddings()
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, cache_store)
        docs = [Document(page_content=combined_content)]
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        print(1)
        theme_retrievers[theme] = vectorstore.as_retriever()
        print(2)
        print(theme_retrievers)

        st.session_state.selected_theme = theme
        st.success(f"{theme} selected!")

        selected_theme = st.session_state.get(
            'selected_theme')
        print(selected_theme)
        print(theme_retrievers)
        if selected_theme not in theme_retrievers:
            st.error(
                f"selected_theme not in tehem_retrievers {selected_theme}, {theme_retrievers}")
        else:
            retriever = theme_retrievers[selected_theme]
            try:
                chain = (
                    {
                        "context": retriever | RunnableLambda(format_docs),
                        "question": RunnablePassthrough(),
                    }
                    | prompt
                    | llm
                )
                response = chain.invoke(question)
                if isinstance(response, dict):
                    st.success(response)
                else:
                    st.success(str(response.content))
            except Exception as e:
                st.error(f"체인 try except 부분, {str(e)}")

    else:
        st.error(f"theme in THEME_FILES, {theme}, {THEME_FILES}")
