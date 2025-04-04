import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.vectorstores import Chroma

# Load environment variables
#load_dotenv()

# Get the API key from the environment
api_key = st.secrets["GOOGLE_API_KEY"]

# Custom CSS for orange and white theme with forced light mode
orange_white_theme = """
<style>
    /* Force light mode by overriding dark mode styles */
    [data-theme="dark"] {
        --background-color: white !important;
        --text-color: #31333F !important;
        --font: "Source Sans Pro", sans-serif !important;
    }
    
    /* Main background and text colors */
    .stApp {
        background-color: white !important;
        color: #31333F !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #FF6600 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #FF6600 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
    }
    
    .stButton>button:hover {
        background-color: #E65C00 !important;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-163ttbj, .css-k1vhr4 {
        background-color: #FFF3E0 !important;
    }
    
    /* Override any dark mode elements */
    .stTextInput>div>div>input {
        background-color: white !important;
        color: #31333F !important;
        border: 1px solid #FFA366 !important;
    }
    
    .stTextInput>label {
        color: #31333F !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #FFF3E0 !important;
        color: #FF6600 !important;
    }
    
    /* Success messages */
    .element-container div[data-testid="stImage"] {
        border: 2px solid #FF6600 !important;
    }
    
    /* File uploader */
    .uploadedFile {
        border: 1px solid #FF6600 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #FF6600 !important;
    }
    
    /* Card-like elements */
    .element-container {
        border-radius: 10px !important;
    }
    
    /* Links */
    a {
        color: #FF6600 !important;
    }
    
    /* Make sure all text is visible in light mode */
    p, span, div {
        color: #31333F !important;
    }
    
    /* Override any remaining dark mode elements */
    [data-baseweb="select"] {
        background-color: white !important;
    }
    
    [data-baseweb="base-input"] {
        background-color: white !important;
    }
</style>
"""

# Page configuration with forced light theme
st.set_page_config(
    page_title="Swiggy Employee BOT",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🍊",
    # Force light theme at the configuration level
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Swiggy Employee Assistant"
    }
)

# Apply custom theme with light mode enforcement
st.markdown(orange_white_theme, unsafe_allow_html=True)

# Add JavaScript to force light theme
st.markdown("""
<script>
    // Force light mode by overriding user preferences
    localStorage.setItem('theme', 'light');
    
    // Apply light mode immediately and prevent changes
    document.querySelector('body').dataset.theme = 'light';
    
    // Prevent theme changes by intercepting the relevant events
    const observer = new MutationObserver(() => {
        document.querySelector('body').dataset.theme = 'light';
    });
    
    observer.observe(document.querySelector('body'), { 
        attributes: true, 
        attributeFilter: ['data-theme'] 
    });
</script>
""", unsafe_allow_html=True)

# Initialize session state
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "web_links" not in st.session_state:
    st.session_state.web_links = []
if "show_main_info" not in st.session_state:
    st.session_state.show_main_info = False
if "show_sidebar_info" not in st.session_state:
    st.session_state.show_sidebar_info = False

# Add toggle functions for the info sections
def toggle_main_info():
    st.session_state.show_main_info = not st.session_state.show_main_info
    
def toggle_sidebar_info():
    st.session_state.show_sidebar_info = not st.session_state.show_sidebar_info

# App title with orange styling
col1, col2 = st.columns([4, 1])

with col1:
    st.markdown("<h1 style='color: #FF6600;'>Swiggy Employee BOT</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #333333;'>Upload PDF documents, add web links, and ask questions about their content.</p>", unsafe_allow_html=True)

# Sidebar for document upload with custom styling
with st.sidebar:
    # Add Important Info button at the top of sidebar as well
    if st.button("Imp Info", key="sidebar_imp_info", on_click=toggle_sidebar_info):
        pass  # The actual toggle happens in the on_click function
    
    # Display info when show_sidebar_info is True
    if st.session_state.show_sidebar_info:
        st.markdown("""
        <div style="padding: 10px; border-radius: 8px; border: 2px solid #FF6600; background-color: #FFF9F5;">
            <h4 style="color: #FF6600; margin-top: 0;">File Size Limitations:</h4>
            <p>Please note that PDFs and web content are limited to approximately 8,000 tokens (roughly 6-7 pages of text). 
            Larger documents may be truncated or fail to process correctly.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='color: #FF6600;'>Document Upload</h3>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
    
    # Web links input
    st.markdown("<h3 style='color: #FF6600;'>Web Links</h3>", unsafe_allow_html=True)
    web_url = st.text_input("Enter a web URL:")
    if web_url and st.button("Add URL"):
        if web_url not in st.session_state.web_links:
            st.session_state.web_links.append(web_url)
            st.success(f"Added URL: {web_url}")
        else:
            st.warning("URL already added.")
    
    # Display added web links
    if st.session_state.web_links:
        st.markdown("<h4 style='color: #FF6600;'>Added URLs:</h4>", unsafe_allow_html=True)
        for i, url in enumerate(st.session_state.web_links):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"{i+1}. {url}")
            with col2:
                if st.button("Remove", key=f"remove_{i}"):
                    st.session_state.web_links.pop(i)
                    st.rerun()
    
    # Process button with orange styling
    if (uploaded_files or st.session_state.web_links):
        process_button = st.button("Process Documents", use_container_width=True)
        if process_button:
            with st.spinner("Processing documents..."):
                try:
                    # Initialize models
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
                    
                    # Create temporary directory to save uploaded files
                    temp_dir = tempfile.TemporaryDirectory()
                    all_pages = []
                    
                    # Process each uploaded file
                    if uploaded_files:
                        for uploaded_file in uploaded_files:
                            temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
                            with open(temp_file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # Load and split the PDF
                            loader = PyPDFLoader(temp_file_path)
                            pages = loader.load_and_split()
                            all_pages.extend(pages)
                            st.write(f"Processed PDF: {uploaded_file.name}")
                    
                    # Process web links
                    if st.session_state.web_links:
                        for url in st.session_state.web_links:
                            try:
                                loader = WebBaseLoader(url)
                                web_pages = loader.load_and_split()
                                all_pages.extend(web_pages)
                                st.write(f"Processed URL: {url}")
                            except Exception as e:
                                st.error(f"Error processing URL {url}: {str(e)}")
                    
                    if all_pages:
                        st.session_state.vectordb = Chroma.from_documents(all_pages, embeddings)
                        st.session_state.retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 3})
                        st.session_state.documents_processed = True
                        
                        st.success(f"Successfully processed {len(uploaded_files) if uploaded_files else 0} PDFs and {len(st.session_state.web_links)} web links.")
                    else:
                        st.error("No documents or web links were successfully processed.")
                        st.session_state.documents_processed = False
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
                    st.session_state.documents_processed = False

# Main area for Q&A with custom styling
st.markdown("<h2 style='color: #FF6600;'>Ask Questions</h2>", unsafe_allow_html=True)

if st.session_state.documents_processed:
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro-exp-03-25", temperature=0.2)
    
    template = """
    You are a helpful AI assistant that answers questions based only on the provided context.
    If the answer is not in the context, say "I don't have sufficient information to answer this question."
    Do not use prior knowledge beyond what's in the context.
    
    Context: {context}
    
    Question: {input}
    
    Answer:
    """
    
    prompt = PromptTemplate.from_template(template)
    
    # Create document chain
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(st.session_state.retriever, combine_docs_chain)
    
    # User question input
    user_question = st.text_input("Enter your question:", key="question_input")
    
    # Get answer button with custom styling
    if user_question:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            answer_button = st.button("Get Answer", use_container_width=True)
        
        if answer_button:
            with st.spinner("Generating answer..."):
                try:
                    # Get response
                    response = retrieval_chain.invoke({"input": user_question})
                    
                    # Display answer with custom styling
                    st.markdown("<h3 style='color: #FF6600;'>Answer:</h3>", unsafe_allow_html=True)
                    
                    # Create an orange-bordered container for the answer
                    st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; border: 1px solid #FF6600; background-color: #FFF9F5;">
                        {response["answer"]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display source documents with custom styling
                    with st.expander("Source Documents"):
                        for i, doc in enumerate(response["context"]):
                            st.markdown(f"<b style='color: #FF6600;'>Document {i+1}:</b>", unsafe_allow_html=True)
                            st.write(doc.page_content[:300] + "...")
                            # If the document has a source attribute (for web links), display it
                            if hasattr(doc.metadata, 'source') and doc.metadata.source:
                                st.markdown(f"<b>Source:</b> <a href='{doc.metadata.source}'>{doc.metadata.source}</a>", unsafe_allow_html=True)
                            st.markdown("<hr style='border-top: 1px solid #FFCCA5;'>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
else:
    # Display info message with custom styling
    st.markdown("""
    <div style="padding: 20px; border-radius: 10px; border: 1px solid #FF6600; background-color: #FFF9F5;">
        <p>Please upload documents/add web links and click 'Process Documents' in the sidebar to begin.</p>
    </div>
    """, unsafe_allow_html=True)

# Add footer with instructions, token limit note, and orange styling
st.markdown("<hr style='border-top: 1px solid #FFCCA5;'>", unsafe_allow_html=True)
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: flex-start;">
    <p style="color: #888888; font-size: 0.8em;">Instructions: Upload PDF files, add web links, process them, and then ask questions about their content.</p>
    <p style="color: #FF6600; font-size: 0.8em; font-weight: bold;">Note: Maximum document size: 8k tokens</p>
</div>
""", unsafe_allow_html=True)

