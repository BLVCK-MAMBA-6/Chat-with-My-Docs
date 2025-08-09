"""
A free document Q&A application using:
- Groq API for fast LLM inference (free tier)
- HuggingFace for local embeddings (completely free)
- Streamlit for web interface (free)
- FAISS for vector search (free, local)
- LangChain for document processing (free)

Total cost: $0.00 - No credit card required!
"""
import random
import streamlit as st 
from groq import Groq 
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import faiss 
import numpy as np 
import tempfile 
import os

# =============================================================================
# GROQ API INTEGRATION WITH CONVERSATION MEMORY
# =============================================================================

def initialize_groq(api_key):
    """
    Initialize the Groq client with API key
    
    Args:
        api_key (str): Groq API key from user input
        
    Returns:
        Groq: Initialized Groq client object
    """
    return Groq(api_key=api_key)

def get_groq_response_with_memory(client, context, question, conversation_history, model_name="llama-3.1-8b-instant"):
    """
    Get response from Groq API using RAG pattern with conversation memory
    
    This function:
    1. Uses system prompts for better conversation awareness
    2. Includes previous Q&A pairs as context
    3. Handles references like "that", "it", "the topic we discussed"
    4. Maintains document grounding while being conversational
    
    Args:
        client (Groq): Initialized Groq client
        context (str): Relevant document chunks as context
        question (str): Current user question
        conversation_history (list): Previous Q&A pairs
        model_name (str): Groq model to use
        
    Returns:
        str: Generated answer with conversation awareness
    """
    
    # Build conversation messages for better context management
    messages = [
        {
            "role": "system",
            "content": """You are a document analysis assistant with conversation memory. Your capabilities:

1. DOCUMENT GROUNDING: Always base answers on the provided document context
2. CONVERSATION AWARENESS: Remember and reference previous exchanges when relevant
3. REFERENCE RESOLUTION: When users say "that", "it", "the topic", understand what they're referring to
4. CLARITY: If a reference is ambiguous, ask for clarification
5. ACCURACY: Never make up information not in the document or conversation

You maintain context across the conversation while staying grounded in the document."""
        }
    ]
    
    # Add recent conversation history (last 5 exchanges to manage tokens)
    for prev_q, prev_a in conversation_history[-5:]:
        messages.append({"role": "user", "content": prev_q})
        messages.append({"role": "assistant", "content": prev_a})
    
    # Add current question with document context
    current_message = f"""Document Context:
{context}

Current Question: {question}"""
    
    messages.append({"role": "user", "content": current_message})
    
    try:
        # Make API call to Groq with conversation context
        response = client.chat.completions.create(
            messages=messages,
            model=model_name,  # Using Llama 3.1 8B for speed and quality
            temperature=0.1,   # Low temperature for factual, consistent answers
            max_tokens=1000    # Reasonable response length
        )
        return response.choices[0].message.content
    except Exception as e:
        # Return user-friendly error message
        return f"Error getting response: {str(e)}"

# =============================================================================
# LOCAL EMBEDDING MODEL
# =============================================================================

@st.cache_resource
def load_embedding_model():
    """
    Load sentence transformer model for creating embeddings locally
    
    Uses Streamlit's cache_resource decorator to load the model only once
    and reuse it across sessions for better performance
    
    Returns:
        SentenceTransformer: Loaded embedding model
    """
    # all-MiniLM-L6-v2 is a good balance of speed, size, and quality
    # It supports 100+ languages and creates 384-dimensional embeddings
    return SentenceTransformer('all-MiniLM-L6-v2')

# =============================================================================
# LOCAL VECTOR STORE CLASS
# =============================================================================

class LocalVectorStore:
    """
    A local vector store using FAISS for similarity search
    
    This class:
    1. Stores document chunks and their embeddings
    2. Creates a FAISS index for fast similarity search
    3. Provides methods to add documents and search for similar content
    """
    
    def __init__(self, embedding_model):
        """
        Initialize the vector store
        
        Args:
            embedding_model: SentenceTransformer model for creating embeddings
        """
        self.embedding_model = embedding_model
        self.chunks = []           # Store original text chunks
        self.embeddings = None     # Store embedding vectors
        self.index = None          # FAISS search index
    
    def add_documents(self, documents):
        """
        Add documents to the vector store and create embeddings
        
        This method:
        1. Extracts text content from document objects
        2. Creates embeddings for each chunk using the local model
        3. Builds a FAISS index for fast similarity search
        
        Args:
            documents (list): List of LangChain document objects
        """
        # Extract text content from LangChain document objects
        self.chunks = [doc.page_content for doc in documents]
        
        # Create embeddings locally (no API calls!)
        embeddings = self.embedding_model.encode(self.chunks)
        self.embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index for fast similarity search
        # IndexFlatL2 uses L2 (Euclidean) distance for similarity
        dimension = self.embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
    
    def similarity_search(self, query, k=4):
        """
        Find the most similar chunks to a query
        
        Args:
            query (str): User's question
            k (int): Number of similar chunks to return
            
        Returns:
            list: List of most similar text chunks
        """
        if self.index is None:
            return []
        
        # Create embedding for the query
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search for similar chunks
        distances, indices = self.index.search(query_embedding, k)
        
        # Return the actual text chunks
        results = []
        for i in indices[0]:
            if i < len(self.chunks):
                results.append(self.chunks[i])
        
        return results

# =============================================================================
# DOCUMENT PROCESSING
# =============================================================================

def load_and_split_pdfs(uploaded_files):
    """
    Load multiple PDF files and split them into chunks for processing.
    
    This function:
    1. Saves each uploaded file temporarily
    2. Uses LangChain to extract text from all PDFs
    3. Combines the text from all documents
    4. Splits the combined text into manageable chunks with overlap
    5. Cleans up temporary files
    
    Uses Streamlit's cache_data decorator to avoid reprocessing the same files.
    
    Args:
        uploaded_files: A list of Streamlit uploaded file objects
        
    Returns:
        list: List of text chunks as LangChain document objects
    """
    if not uploaded_files:
        return []

    temp_paths = []
    all_documents = []

    try:
        # Loop through each uploaded file
        for uploaded_file in uploaded_files:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
                temp_paths.append(tmp_path)
            
            # Load PDF using LangChain's PyPDFLoader
            loader = PyPDFLoader(tmp_path)
            all_documents.extend(loader.load())
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,     # Each chunk ~1000 characters
            chunk_overlap=200,   # 200 character overlap to preserve context
            separators=["\n\n", "\n", " ", ""] # Split on paragraphs, then lines, then words
        )
        chunks = text_splitter.split_documents(all_documents)
        
        return chunks
    finally:
        # Clean up all temporary files
        for tmp_path in temp_paths:
            os.unlink(tmp_path)


# =============================================================================
# CONVERSATION MANAGEMENT
# =============================================================================

def manage_conversation_context(conversation_history, max_exchanges=10):
    """
    Manage conversation history to prevent token overflow
    
    Args:
        conversation_history (list): List of (question, answer) tuples
        max_exchanges (int): Maximum number of exchanges to keep
        
    Returns:
        list: Trimmed conversation history
    """
    if len(conversation_history) > max_exchanges:
        return conversation_history[-max_exchanges:]
    return conversation_history

# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_document(uploaded_file, groq_client, embedding_model):
    """
    Main document processing pipeline with conversation memory
    
    This function orchestrates the entire RAG pipeline:
    1. Loads and splits the PDF
    2. Creates embeddings and vector store
    3. Sets up the conversational Q&A interface
    4. Handles user questions with conversation context
    
    Args:
        uploaded_file: Streamlit uploaded file object
        groq_client: Initialized Groq API client
        embedding_model: Loaded sentence transformer model
    """
    st.write("📄 Processing your document...")
    
    # Step 1: Load and split PDF
    with st.spinner("📖 Reading PDF..."):
        chunks = load_and_split_pdfs(uploaded_file)
    
    if not chunks:
        st.error("❌ Could not extract text from PDF")
        return
    
    st.success(f"✅ Document loaded! Found {len(chunks)} chunks")
    
    # Step 2: Create vector store with embeddings
    with st.spinner("🧮 Creating embeddings (running locally)..."):
        vector_store = LocalVectorStore(embedding_model)
        vector_store.add_documents(chunks)
    
    st.success("✅ Document ready for questions!")
    
    # Step 3: Initialize conversation history if not exists
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Step 4: Store everything in session state for persistence
    st.session_state.vector_store = vector_store
    st.session_state.groq_client = groq_client
    st.session_state.ready = True

    # Step 5: Conversational Q&A Interface
    if st.session_state.get('ready', False):
        st.header("💬 Ask Your Questions")
        
        # Show conversation status
        if st.session_state.conversation_history:
            st.info(f"💭 Conversation memory: {len(st.session_state.conversation_history)} exchanges")
        
        # Provide example questions to help users get started
        st.write("**Try asking:**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 What is this document about?"):
                st.session_state.question = "What is this document about?"
            if st.button("👥 Who are the main authors or people mentioned?"):
                st.session_state.question = "Who are the main authors or people mentioned?"
        with col2:
            if st.button("🔍 What are the key findings or conclusions?"):
                st.session_state.question = "What are the key findings or conclusions?"
            if st.button("📊 Can you elaborate on that?"):
                st.session_state.question = "Can you elaborate on that?"
        
        # Clear conversation button
        if st.session_state.conversation_history:
            if st.button("🗑️ Clear Conversation History"):
                st.session_state.conversation_history = []
                st.success("Conversation history cleared!")
                st.rerun()
        
        # Main question input
        question = st.text_input(
            "Your question:", 
            value=st.session_state.get('question', ''),
            key="user_question",
            placeholder="Ask anything about the document... I remember our conversation!"
        )
        
        # Process question when user enters one
        if question:
            try:
                with st.spinner("🤔 Thinking... (using conversation context + Groq's lightning-fast API)"):
                    # Step 5a: Find relevant chunks using similarity search
                    relevant_chunks = st.session_state.vector_store.similarity_search(question, k=4)
                    
                    if not relevant_chunks:
                        st.warning("🤷 No relevant information found. Try rephrasing your question.")
                        return
                    
                    # Step 5b: Combine chunks into context
                    context = "\n\n".join(relevant_chunks)
                    
                    # Step 5c: Get conversation history
                    conversation_history = manage_conversation_context(
                        st.session_state.conversation_history, 
                        max_exchanges=10
                    )
                    
                    # Step 5d: Get response with conversation memory
                    answer = get_groq_response_with_memory(
                        st.session_state.groq_client, 
                        context, 
                        question, 
                        conversation_history,
                        st.session_state.get('selected_model', 'llama-3.1-8b-instant')
                    )
                    
                    # Step 5e: Store this Q&A in conversation history
                    st.session_state.conversation_history.append((question, answer))
                
                # Step 5f: Display results
                st.write("**🎯 Answer:**")
                st.write(answer)
                
                # Show performance info
                st.success("⚡ Powered by Groq's blazing-fast inference + conversation memory!")
                
                # Show conversation history
                if len(st.session_state.conversation_history) > 1:
                    with st.expander("💬 Conversation History"):
                        for i, (q, a) in enumerate(st.session_state.conversation_history[:-1]):  # Exclude current
                            st.write(f"**Q{i+1}:** {q}")
                            display_answer = a[:200] + "..." if len(a) > 200 else a
                            st.write(f"**A{i+1}:** {display_answer}")
                            st.write("---")
                
                # Show source chunks for transparency and debugging
                with st.expander("📚 View source chunks"):
                    for i, chunk in enumerate(relevant_chunks):
                        st.write(f"**Chunk {i+1}:**")
                        # Truncate long chunks for readability
                        display_chunk = chunk[:400] + "..." if len(chunk) > 400 else chunk
                        st.write(display_chunk)
                        st.write("---")
                
            except Exception as e:
                # Handle different types of errors gracefully
                if "rate limit" in str(e).lower():
                    st.error("🕐 Rate limit reached. Please wait a moment and try again.")
                    st.info("💡 Free tier limits are generous but not unlimited!")
                elif "context_length" in str(e).lower():
                    st.error("📏 Conversation too long. Clearing older messages...")
                    st.session_state.conversation_history = st.session_state.conversation_history[-5:]
                    st.info("💡 Try asking your question again!")
                else:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Try simplifying your question or check your API key.")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """
    Main Streamlit application with conversation memory
    
    This function:
    1. Sets up the page configuration
    2. Creates the sidebar for API key and model selection
    3. Handles file upload
    4. Orchestrates the entire application flow with conversation context
    """
    # Configure the Streamlit page
    st.set_page_config(
        page_title="Chat with My Docs 📚💬", 
        page_icon="📚",
        layout="wide"
    )
    
    st.markdown("""
    <style>
    /* Import a font from Google Fonts. */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    /* General body and text styling with a more robust selector */
    html, body, [class*="st-"], div, p, label, .st-b, .st-c {
        font-family: 'Poppins', sans-serif;
        color: #f0f2f6; /* Set a default light text color */
    }
    
    /* Main Content Area (Larger Area) */
    html, body {
        background-color: #0e1117; /* Very dark grey / black */
    }

    /* Sidebar (Lesser Area) */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e; /* A slightly lighter grey for the sidebar */
        color: #f0f2f6; /* Ensure text is visible on the darker background */
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); /* A subtle shadow for depth */
    }
    
    /* Main title styling */
    h1 {
        color: #2e8b57;
        text-align: center;
    }

    /* Button styling for suggestions, etc. */
    div.stButton > button {
        margin-right: 5px;
        margin-bottom: 5px;
        border-radius: 20px;
        background-color: #3b5998;
        color: white;
        border: none;
        padding: 10px 15px;
        font-weight: bold;
        cursor: pointer;
    }

    div.stButton > button:hover {
        background-color: #4b6ead;
    }
    
    /* Input widgets styling */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stSlider>div>div {
        border-radius: 10px;
    }

    /* Main chat input styling */
    .st-emotion-cache-1629p8f {
        background-color: #1a1a2e;
        border-radius: 10px;
        border: 1px solid #2e8b57;
    }
    </style>
     """, unsafe_allow_html=True)

    
    st.title("Chat with My Docs 📚💬")
    st.write("100% free APIs - Upload a PDF and have a conversation about it!")
    
    # Sidebar for configuration
    st.sidebar.header("🔧 Setup (Free!)")
    st.sidebar.write("Get your free Groq API key at: https://console.groq.com")
    
    # API key input
    groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
    
    # Model selection dropdown
    model_options = {
        "llama-3.1-8b-instant": "Llama 3.1 8B (Fast & Smart) - 131k context",
        "llama-3.3-70b-versatile": "Llama 3.3 70B (Most Capable) - 131k context",
        "gemma2-9b-it": "Gemma2 9B (Balanced) - 8k context"
    }
    
    selected_model = st.sidebar.selectbox(
        "Choose Model:",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0  # Default to llama-3.1-8b-instant
    )
    
    # Conversation settings
    st.sidebar.header("💬 Conversation Settings")
    max_history = st.sidebar.slider(
        "Max conversation exchanges to remember:",
        min_value=3,
        max_value=20,
        value=10,
        help="Higher values provide more context but use more tokens"
    )
    
    # Show helpful info if no API key
    if not groq_api_key:
        st.warning("⚠️ Get your free Groq API key at https://console.groq.com")
        st.info("💡 No credit card required - just sign up and start building!")
        
        # Show demo info
        st.markdown("""
        ### ✨ Key Features
        - **Supports Multiple Documents**: Upload and chat with one or more PDF files.
        - **Intelligent Q&A**: Ask questions and get accurate, context-aware answers from your documents.
        - **Conversational Memory**: The AI remembers your previous questions and answers, allowing for a natural conversation flow.
        - **Lightning-Fast Responses**: Get your answers in under a second, powered by Groq's APIs.
        - **Absolute Privacy**: Your documents are processed locally, and only relevant snippets are sent to the API, ensuring your data remains private.
        - **Completely Free**: Use this application without any hidden costs or credit card requirements.
        """)
        st.stop()
    
    # Initialize clients
    groq_client = initialize_groq(groq_api_key)
    embedding_model = load_embedding_model()
    
    # Store selected model and settings in session state
    st.session_state.selected_model = selected_model
    st.session_state.max_history = max_history
    
    # Modified File upload widget to accept multiple files
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF documents to start asking questions about them"
    )
    
    # Corrected flow: all subsequent logic is now inside this block
    if uploaded_files:
        # Load and split documents, which returns a list of chunks
        chunks = load_and_split_pdfs(uploaded_files) 
        
        if chunks:
            # Process the documents and enable the chat interface
            process_document(uploaded_files, groq_client, embedding_model)
    else:
        # Show instructions when no file is uploaded
        st.markdown("""
        ### 🚀 Getting Started
        1. **Get your free Groq API key** at https://console.groq.com
        2. **Enter your API key** in the sidebar
        3. **Upload a PDF document** using the file uploader above
        4. **Start asking questions** - the AI remembers your conversation!
        
        ### 💡 Example Questions to Try
        - "What is this document about?"
        - "Who are the main authors?"
        - "Can you elaborate on that?" (references previous answer)
        - "How does it compare to what we discussed earlier?"
        """)

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()

