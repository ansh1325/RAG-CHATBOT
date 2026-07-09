import streamlit as st
import os
import datetime
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Samsung Wash Code RAG",
    page_icon="🧼",
    layout="centered", # Centered is better for chat interfaces
    initial_sidebar_state="expanded"
)

MANUAL_FILE_PATH = "washing_machine_manual.html"

# ==========================================
# 2. Cached RAG Engine Initialization
# ==========================================
@st.cache_resource(show_spinner="Initializing Manual Database...")
def init_rag_engine(api_key: str):
    """Loads, splits, and embeds the document only once."""
    if not os.path.exists(MANUAL_FILE_PATH):
        raise FileNotFoundError(f"Manual file '{MANUAL_FILE_PATH}' not found.")

    # Load and split
    loader = UnstructuredHTMLLoader(file_path=MANUAL_FILE_PATH)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # Embed and store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    # Create RAG Chain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    prompt = ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question} 
Context: {context} 
Answer:""")
    
    rag_chain = (
        {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return rag_chain

# ==========================================
# 3. Sidebar UI
# ==========================================
with st.sidebar:
    st.markdown("## 🧺 Samsung RAG Chatbot")
    st.markdown("Context-aware assistant for your washing machine.")
    
    # API Key Input
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if api_key:
        st.success("🔒 API Key loaded from Secrets.")
    else:
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-proj-...")
        if not api_key:
            st.warning("🔑 Please enter your OpenAI API Key to continue.")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.caption("Powered by Streamlit • LangChain • ChromaDB • OpenAI")

# ==========================================
# 4. Main Chat Interface
# ==========================================
st.title("Washing Machine Assistant")
st.write("Ask questions about manual codes, cleaning cycles, daily washes, and warning signs.")

# Initialize chat history & prompt state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_prompt" not in st.session_state:
    st.session_state.active_prompt = None

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Suggested Questions UI (Only show if chat is empty)
if not st.session_state.messages:
    st.markdown("### Suggested Questions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("What does the Drum Clean cycle do?", use_container_width=True):
            st.session_state.active_prompt = "What does the Drum Clean cycle do?"
        if st.button("What does error code 4C mean?", use_container_width=True):
            st.session_state.active_prompt = "What does error code 4C mean?"
            
    with col2:
        if st.button("Which cycle is best for cotton clothes?", use_container_width=True):
            st.session_state.active_prompt = "Which cycle is best for cotton clothes?"
        if st.button("How do I start Eco Bubble mode?", use_container_width=True):
            st.session_state.active_prompt = "How do I start Eco Bubble mode?"

# Native Chat Input
user_input = st.chat_input("Ask a question about your washing machine...")

# Handle input (either from chat box or suggested buttons)
if user_input:
    st.session_state.active_prompt = user_input

if st.session_state.active_prompt:
    prompt = st.session_state.active_prompt
    st.session_state.active_prompt = None # Reset state after capturing
    
    if not api_key:
        st.error("Please enter your OpenAI API Key in the sidebar.")
    else:
        # 1. Show user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 2. Generate and show assistant response
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching manual..."):
                    rag_chain = init_rag_engine(api_key)
                    response = rag_chain.invoke(prompt).content
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            except FileNotFoundError as e:
                st.error(f"📂 Missing manual asset: {e}")
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")