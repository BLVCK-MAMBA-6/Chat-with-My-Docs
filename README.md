# Chat with My Docs 📚💬

A completely free document Q&A application with conversation memory, built during Week 1 of AI Summer of Code Season 2.

## 🎯 What This App Does

Upload one or more PDF documents and have an intelligent conversation about them! The AI remembers your previous questions and can reference them naturally.

**Example conversation:**
* You: "What is this research paper about?"
* AI: "This paper discusses machine learning applications in healthcare..."
* You: "What methodology did they use for that?"
* AI: "For the machine learning applications in healthcare that I mentioned, they used..."
* You: "How does it compare to previous work?"
* AI: "Compared to the previous work mentioned in the paper..."

## 💰 Cost Breakdown

* **Total cost:** $0.00
* **✅ Groq API:** Free tier (thousands of requests/day)
* **✅ HuggingFace Embeddings:** Run locally (completely free)
* **✅ Streamlit:** Free framework
* **✅ FAISS Vector Search:** Free, runs locally
* **✅ Streamlit Cloud Hosting:** Free

No credit card required. No hidden fees. Actually free.

## 🛠️ Technology Stack

* **Groq:** Lightning-fast LLM inference (< 1 second response times) - [https://groq.com/](https://groq.com/)
* **HuggingFace:** Local embeddings with `all-MiniLM-L6-v2` - [https://huggingface.co/](https://huggingface.co/)
* **Streamlit:** Web framework for the UI - [https://streamlit.io/](https://streamlit.io/)
* **LangChain:** Document processing and text splitting - [https://langchain.com/](https://langchain.com/)
* **FAISS:** Fast similarity search for vectors - [https://faiss.ai/](https://faiss.ai/)

## 🏃‍♂️ Quick Start

1. **Clone the Repository**

    ```
    git clone [https://github.com/aisummerofcode/aisoc-season-2.git](https://github.com/aisummerofcode/aisoc-season-2.git)
    cd "aisoc-season-2/Applied AI/src/week_1/day_3_first_llm"
    ```

2. **Install Dependencies**

    ```
    pip install -r requirements.txt
    ```

3. **Get Your Free Groq API Key**

    * Go to [console.groq.com](https://console.groq.com)
    * Sign up (no credit card required)
    * Create an API key

4. **Run the Application**

    ```
    streamlit run app.py
    ```

5. **Start Chatting!**

    * Enter your Groq API key in the sidebar
    * Upload a PDF document
    * Ask questions and have a conversation!

## 🧠 How It Works (The RAG Pattern)

This app implements RAG (Retrieval-Augmented Generation) with conversation memory:
* **Document Processing:** PDF is split into chunks with overlap
* **Local Embeddings:** Each chunk gets an embedding vector (runs on your computer)
* **Vector Storage:** FAISS index for fast similarity search
* **Question Processing:** Find relevant chunks for each question
* **Conversation Context:** Include previous Q&A pairs as context
* **LLM Generation:** Groq generates answers using document context + conversation history

## 🎛️ Available Models

* `llama-3.1-8b-instant` (Default): Fast and smart, 131k context window
* `llama-3.3-70b-versatile`: Most capable, 131k context window
* `gemma2-9b-it`: Balanced option, 8k context window

## 🔧 Configuration Options

### Chunk Settings
* **Chunk Size:** 1000 characters (adjustable)
* **Overlap:** 200 characters (prevents information splitting)
* **Retrieval:** Top 4 most relevant chunks per question

### Conversation Memory
* **History Length:** Configurable (3-20 exchanges)
* **Token Management:** Automatic trimming to prevent overflow
* **Context Awareness:** References previous answers naturally

### Model Parameters
* **Temperature:** 0.1 (factual, consistent answers)
* **Max Tokens:** 1000 (reasonable response length)
* **System Prompt:** Optimized for document grounding + conversation

## 🔍 Troubleshooting

### Common Issues
* **"No relevant information found"**
    * Document might be scanned (needs OCR)
    * Try rephrasing your question
    * Check if PDF text extracted properly
* **Rate limiting errors**
    * Free tier has generous limits but not unlimited
    * Wait a moment and try again
    * Consider upgrading for heavy usage
* **Wrong or incomplete answers**
    * Check the "View source chunks" section
    * Information might be split across chunks
    * Try adjusting chunk size or asking more specific questions
* **Conversation context issues**
    * Clear conversation history if it gets too long
    * Reduce max history length in sidebar
    * Be specific when referencing previous answers

