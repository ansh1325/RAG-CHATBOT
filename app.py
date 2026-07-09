import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import os
import tempfile
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

st.set_page_config(page_title="Washing Machine Assistant", page_icon="🤖")
st.title("🤖 Washing Machine RAG Chatbot")

# Automatically pull key from secrets.toml
if "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    st.error("Please add OPENAI_API_KEY to your .streamlit/secrets.toml file.")
    st.stop()

with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload the manual (HTML file)", type=["html", "htm"])
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Build pipeline if file uploaded
if uploaded_file and st.session_state.rag_chain is None:
    with st.spinner("Processing document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        try:
            loader = UnstructuredHTMLLoader(file_path=tmp_file_path)
            docs = loader.load()
            splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
            vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))
            
            prompt = ChatPromptTemplate.from_template(
                "You are an assistant for question-answering tasks.\n"
                "Use the retrieved context to answer the question. If you don't know, say so.\n"
                "Max 3 sentences.\n\nQuestion: {question}\nContext: {context}\nAnswer:"
            )
            st.session_state.rag_chain = (
                {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()} | prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0)
            )
            st.success("Ready! Ask away.")
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

# UI Elements
if not uploaded_file:
    st.info("Please drop your HTML manual into the sidebar to start.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask a question about your manual..."):
    if not st.session_state.rag_chain:
        st.warning("Please upload a document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke(user_query).content
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})