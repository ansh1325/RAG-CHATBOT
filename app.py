import streamlit as st
import os
import datetime
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Set Streamlit page configuration with a premium icon and layout
st.set_page_config(
    page_title="Samsung Washing Machine Manual RAG Chatbot",
    page_icon="🧼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-quality, premium visual appearance
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Premium Theme Variables */
    :root {
        --bg-main: #0B0F19;
        --bg-sidebar: #111827;
        --bg-card: #1F2937;
        --border-color: #374151;
        --text-primary: #F3F4F6;
        --text-secondary: #9CA3AF;
        --accent-blue: #3B82F6;
        --accent-blue-hover: #60A5FA;
        --accent-blue-glow: rgba(59, 130, 246, 0.4);
    }

    /* Main Container & Background Color (no height or overflow modification) */
    [data-testid="stAppViewContainer"] {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Header styling (transparent background with blur) */
    [data-testid="stHeader"] {
        background-color: rgba(11, 15, 25, 0.8) !important;
        backdrop-filter: blur(10px) !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    /* Streamlit Main Page Container Spacing */
    .block-container {
        padding-top: 6rem !important; /* Proper padding so the title is fully visible and below header */
        padding-bottom: 4rem !important;
        max-width: 1200px !important;
    }

    /* Premium card wrapper styles */
    .premium-card {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 18px !important;
        margin-bottom: 15px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }

    .premium-card h4 {
        margin-top: 0 !important;
        margin-bottom: 8px !important;
        color: var(--accent-blue-hover) !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }

    .premium-card p, .premium-card ol, .premium-card li {
        font-size: 0.88rem !important;
        line-height: 1.4 !important;
        color: var(--text-primary) !important;
        margin-bottom: 0 !important;
    }

    /* Sidebar Logo & Title styling */
    .sidebar-logo-container {
        text-align: center;
        margin-top: 10px;
        margin-bottom: 25px;
        padding-bottom: 15px;
        border-bottom: 1px solid var(--border-color);
    }
    
    .logo-box {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
        transition: all 0.3s ease;
    }
    
    .logo-box.circular {
        width: 90px;
        height: 90px;
        border-radius: 50%;
    }
    
    .logo-box.rectangular {
        padding: 10px 15px;
        border-radius: 12px;
    }
    
    .logo-box img {
        max-width: 120px;
        height: auto;
        border-radius: 6px;
    }

    .sidebar-title-main {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        color: var(--text-primary) !important;
        line-height: 1.3 !important;
    }
    
    .sidebar-title-sub {
        color: var(--accent-blue-hover) !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }

    /* Titles and Subtitles */
    .main-title {
        font-family: 'Outfit', sans-serif !important;
        font-size: calc(1.8rem + 1.2vw) !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 50%, #93C5FD 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 0.5rem !important;
        line-height: 1.2 !important;
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif !important;
        font-size: calc(0.9rem + 0.15vw) !important;
        color: var(--text-secondary) !important;
        margin-bottom: 2rem !important;
        font-weight: 400 !important;
    }

    /* Input area - Form override */
    [data-testid="stForm"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3) !important;
        margin-bottom: 15px !important;
    }

    /* Custom styles for Streamlit Inputs */
    .stTextInput input {
        background-color: #111827 !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stTextInput input:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 1px var(--accent-blue) !important;
    }

    /* Button Consistency & Styling (No negative margins or translateY) */
    div.stFormSubmitButton > button {
        background: linear-gradient(135deg, var(--accent-blue) 0%, #1D4ED8 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
        padding: 0.6rem 1.2rem !important;
        box-shadow: 0 4px 6px var(--accent-blue-glow) !important;
        width: 100% !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    div.stFormSubmitButton > button:hover {
        background: linear-gradient(135deg, var(--accent-blue-hover) 0%, var(--accent-blue) 100%) !important;
        box-shadow: 0 6px 12px var(--accent-blue-glow) !important;
    }

    /* Clear Button & Suggestions */
    div.stButton > button {
        background: var(--bg-card) !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.4rem 1rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }
    
    div.stButton > button:hover {
        color: var(--text-primary) !important;
        border-color: #4B5563 !important;
        background: #374151 !important;
    }

    /* Suggested questions styling */
    .suggestion-title {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        margin-top: 15px !important;
        margin-bottom: 8px !important;
        font-family: 'Outfit', sans-serif !important;
    }

    /* Custom Chat history elements */
    .chat-timestamp {
        font-size: 0.72rem !important;
        color: var(--text-secondary) !important;
        text-align: right !important;
        margin-top: 4px !important;
        display: block !important;
    }
    
    [data-testid="chatAvatarIcon-user"], [data-testid="chatAvatarIcon-assistant"] {
        background-color: var(--accent-blue) !important;
    }

    /* Footer styling */
    .footer {
        text-align: center !important;
        padding: 20px 0 !important;
        font-size: 0.8rem !important;
        color: var(--text-secondary) !important;
        border-top: 1px solid var(--border-color) !important;
        margin-top: 40px !important;
        font-family: 'Inter', sans-serif !important;
    }
    .footer strong {
        color: var(--accent-blue-hover) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Define local relative path to manual
MANUAL_FILE_PATH = "washing_machine_manual.html"

# ==========================================
# 1. Document Loading
# ==========================================
@st.cache_data(show_spinner=False)
def load_documents(file_path: str):
    """
    Loads document from the specified local HTML file path.
    Cached using @st.cache_data to prevent re-reading the file on every interaction.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Manual file '{file_path}' not found in the project root directory.")
    
    # Utilizing UnstructuredHTMLLoader to load the plain text content of the HTML manual
    loader = UnstructuredHTMLLoader(file_path=file_path)
    return loader.load()

# ==========================================
# 2. Vectorstore Creation
# ==========================================
@st.cache_resource(show_spinner=False)
def create_vectorstore(_documents, openai_api_key: str):
    """
    Splits loaded documents and initializes an in-memory Chroma vectorstore.
    Cached using @st.cache_resource to prevent re-indexing the manual on every interaction.
    """
    # Initialize character splitter with appropriate chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(_documents)
    
    # Initialize OpenAIEmbeddings using the user provided API key
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    
    # Load document splits into Chroma vectorstore (runs in-memory)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

# ==========================================
# 3. RAG Chain Configuration
# ==========================================
def create_rag_chain(retriever, openai_api_key: str):
    """
    Sets up the LLM, ChatPromptTemplate, and builds the RAG chain using LCEL.
    """
    # Initialize chat model with temperature = 0 for deterministic and focused answers
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
    
    # Define system instructions and context-aware prompt structure
    prompt = ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question} 
Context: {context} 
Answer:""")
    
    # Chain components: context retriever and question passthrough to prompt to LLM
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return rag_chain

# ==========================================
# 4. Query Invocation
# ==========================================
def get_response(chain, question: str) -> str:
    """
    Invokes the built RAG chain with the user question and returns the text response.
    """
    response = chain.invoke(question)
    return response.content

# ==========================================
# 5. Main Execution
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def check_logo_exists(path_or_url: str) -> bool:
    """
    Checks if a local file exists or if a remote URL is reachable.
    Cached for 1 hour to prevent blocking Streamlit rerun performance.
    """
    if not path_or_url:
        return False
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        try:
            import urllib.request
            req = urllib.request.Request(path_or_url, method="HEAD")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                return resp.status == 200
        except Exception:
            return False
    return os.path.exists(path_or_url)

def main():
    # ------------------ SIDEBAR ------------------
    # 1. Logo
    logo_path = "samsung_logo.png"
    logo_url = "https://upload.wikimedia.org/wikipedia/commons/2/24/Samsung_Logo.svg"
    
    logo_exists = False
    active_logo = ""
    
    if os.path.exists(logo_path):
        logo_exists = True
        active_logo = logo_path
    else:
        # Check cached URL availability
        if check_logo_exists(logo_url):
            logo_exists = True
            active_logo = logo_url
            
    if logo_exists:
        st.sidebar.markdown(
            f"""
            <div class="sidebar-logo-container">
                <div class="logo-box rectangular">
                    <img src="{active_logo}" />
                </div>
                <div class="sidebar-title-main">
                    Samsung Washing Machine<br/>
                    <span class="sidebar-title-sub">RAG Chatbot</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            """
            <div class="sidebar-logo-container">
                <div class="logo-box circular">
                    <span style="font-size: 2.8rem;">🧺</span>
                </div>
                <div class="sidebar-title-main">
                    Samsung Washing Machine<br/>
                    <span class="sidebar-title-sub">RAG Chatbot</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # 2. OpenAI API Key Input
    secrets_api_key = st.secrets.get("OPENAI_API_KEY", "")
    
    if secrets_api_key:
        st.sidebar.success("🔒 API Key loaded from Secrets.")
        api_key = secrets_api_key
    else:
        api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-proj-...",
            help="Your key is processed locally and never stored on our servers."
        )
        if not api_key:
            st.sidebar.info("🔑 Please enter your OpenAI API Key above to unlock the chatbot.")
            
    # 3. About Card
    st.sidebar.markdown(
        """
        <div class="premium-card">
            <h4>About the Project</h4>
            <p>This context-aware RAG Chatbot integrates the Samsung Washing Machine manual to provide immediate operational guidance, warning message explanation, and mode instructions.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 4. Instructions Card
    st.sidebar.markdown(
        """
        <div class="premium-card">
            <h4>Instructions</h4>
            <ol>
                <li>Enter your OpenAI API Key.</li>
                <li>Enter your question or select one of the suggested questions.</li>
                <li>Click <strong>Ask</strong> to get context-aware answers.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # ------------------ MAIN CONTAINER ------------------
    # Main Dashboard UI
    st.markdown("<div class='main-title'>Samsung Washing Machine Manual RAG Chatbot</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Ask questions about manual codes, cleaning cycles, daily washes, and warning signs.</div>", unsafe_allow_html=True)
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.markdown(f"<span class='chat-timestamp'>{message['timestamp']}</span>", unsafe_allow_html=True)
            
    # Form for user input to handle submission and Clear button flow
    with st.form(key="qa_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_question = st.text_input(
                "Enter your question:",
                placeholder="e.g., What is the cycle for DRUM CLEAN?",
                label_visibility="collapsed"
            )
        with col_btn:
            submit_button = st.form_submit_button("Ask", use_container_width=True)

    # Clickable Suggested Questions
    st.markdown("<div class='suggestion-title'>Suggested Questions:</div>", unsafe_allow_html=True)
    
    col_sug1, col_sug2 = st.columns(2)
    clicked_question = ""
    
    with col_sug1:
        if st.button("What does the Drum Clean cycle do?", key="sug_1", use_container_width=True):
            clicked_question = "What does the Drum Clean cycle do?"
        if st.button("How do I clean the washing machine?", key="sug_2", use_container_width=True):
            clicked_question = "How do I clean the washing machine?"
        if st.button("What does error code 4C mean?", key="sug_3", use_container_width=True):
            clicked_question = "What does error code 4C mean?"
            
    with col_sug2:
        if st.button("Which cycle is best for cotton clothes?", key="sug_4", use_container_width=True):
            clicked_question = "Which cycle is best for cotton clothes?"
        if st.button("How do I start Eco Bubble mode?", key="sug_5", use_container_width=True):
            clicked_question = "How do I start Eco Bubble mode?"

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

    # Layout for Clear button
    col_clear, _ = st.columns([1, 4])
    with col_clear:
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Determine query to process
    active_query = ""
    if submit_button and user_question:
        active_query = user_question
    elif clicked_question:
        active_query = clicked_question

    # Process submission
    if active_query:
        # Check if the API key is provided
        if not api_key:
            st.error("🔑 Please enter a valid OpenAI API Key in the sidebar or configure it in your application settings.")
            return
            
        # Display user message in chat
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        st.session_state.messages.append({
            "role": "user",
            "content": active_query,
            "timestamp": current_time
        })
            
        # Spinner while loading resources, embeddings, and fetching answers
        with st.spinner("Analyzing manual and generating response..."):
            try:
                # 1. Load documents locally
                documents = load_documents(MANUAL_FILE_PATH)
                
                # 2. Setup vectorstore & embeddings
                vectorstore = create_vectorstore(documents, api_key)
                retriever = vectorstore.as_retriever()
                
                # 3. Create RAG chain
                chain = create_rag_chain(retriever, api_key)
                
                # 4. Generate answer
                answer = get_response(chain, active_query)
                
                # Show assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "timestamp": datetime.datetime.now().strftime("%I:%M %p")
                })
                    
            except FileNotFoundError as fnf_error:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"📂 Missing manual asset: {fnf_error}",
                    "timestamp": datetime.datetime.now().strftime("%I:%M %p")
                })
            except Exception as e:
                # Provide user-friendly alert for any configuration, network, or API errors
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ An error occurred: {e}",
                    "timestamp": datetime.datetime.now().strftime("%I:%M %p")
                })
        st.rerun()
                
    # Premium styled footer
    st.markdown(
        """
        <div class='footer'>
            Powered by <strong>Streamlit</strong> • <strong>LangChain</strong> • <strong>ChromaDB</strong> • <strong>OpenAI</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
