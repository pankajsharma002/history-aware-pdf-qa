import streamlit as st
import os
from dotenv import load_dotenv

# LangChain imports
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# 🔥 RERANKER IMPORT
from sentence_transformers import CrossEncoder

# ---------------- Load ENV ----------------
load_dotenv()

# ---------------- Streamlit UI ----------------
st.title("Conversational RAG with PDF upload and chat history (With Reranker)")
st.write("Upload PDFs and chat with their content")

api_key = st.text_input("Enter the Groq API key:", type="password")

# ---------------- CACHED EMBEDDINGS ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ---------------- CACHED LLM ----------------
@st.cache_resource
def load_llm(api_key):
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant"
    )

# ---------------- CACHED RERANKER ----------------
@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------- RERANK FUNCTION ----------------
def rerank_documents(query, docs, top_k=3):
    reranker = load_reranker()

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))
    scored_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in scored_docs[:top_k]]

# ---------------- CACHED VECTORSTORE ----------------
@st.cache_resource
def create_vectorstore(file_bytes):
    temp_pdf = "temp.pdf"

    with open(temp_pdf, "wb") as f:
        f.write(file_bytes)

    loader = PyPDFLoader(temp_pdf)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(documents)

    embeddings = load_embeddings()

    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )

    os.remove(temp_pdf)
    return vectorstore

# ---------------- CHAT HISTORY ----------------
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

# ---------------- MAIN APP ----------------
if api_key:
    llm = load_llm(api_key)

    session_id = st.text_input("Session ID", value="default_session")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        accept_multiple_files=False
    )

    if uploaded_file is not None:

        vectorstore = create_vectorstore(uploaded_file.getvalue())

        # 🔥 Retrieve more docs for reranking
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

        # ---------------- HISTORY-AWARE RETRIEVER ----------------
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, "
            "formulate a standalone question. Do NOT answer it."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriever,
            contextualize_q_prompt
        )

        # ---------------- QA PROMPT ----------------
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following retrieved context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Use a maximum of three sentences.\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt
        )

        # ---------------- CUSTOM RAG WITH RERANKER ----------------
        def custom_rag_chain(query, session_id):
            chat_history = get_session_history(session_id).messages

            # Step 1: Reformulate query
            standalone_query = llm.invoke(
                contextualize_q_prompt.format_messages(
                    input=query,
                    chat_history=chat_history
                )
            ).content

            # Step 2: Retrieve docs
            docs = retriever.get_relevant_documents(standalone_query)

            # Step 3: 🔥 Rerank
            top_docs = rerank_documents(standalone_query, docs, top_k=3)

            # Step 4: Generate answer
            response = llm.invoke(
                qa_prompt.format_messages(
                    input=query,
                    chat_history=chat_history,
                    context="\n\n".join([doc.page_content for doc in top_docs])
                )
            )

            # Step 5: Save history
            history = get_session_history(session_id)
            history.add_user_message(query)
            history.add_ai_message(response.content)

            return response.content

        # ---------------- USER INPUT ----------------
        user_input = st.text_input("Your question:")

        if user_input:
            answer = custom_rag_chain(user_input, session_id)
            st.success(f"Assistant: {answer}")

            st.write("Chat History:", get_session_history(session_id).messages)

else:
    st.warning("Please enter your Groq API key")