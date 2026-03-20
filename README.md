# 🔋 WattMonk RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot powered by ChromaDB, semantic search, and Google Gemini AI. Intelligently answers questions about NEC electrical codes and Wattmonk solar installation services.

## Features

✨ **Dual Knowledge Bases**
- ⚡ **NEC** - National Electrical Code standards (Article 690)
- 🌞 **Wattmonk** - Solar installation and services
- 📖 **General** - General knowledge synthesis

🤖 **Smart Intent Classification**
- Automatically routes queries to the most relevant knowledge base
- Keyword-based intent detection (expandable)

📚 **Vector Search with ChromaDB**
- Semantic similarity search on PDF content
- Metadata filtering by source (NEC/Wattmonk)
- 192+ document chunks for precise retrieval

🔄 **Retry Logic with Rate Limiting**
- Automatic 60-second retry on API quota limits
- Up to 3 retry attempts before fallback
- Graceful error handling

💬 **Interactive Streamlit UI**
- Real-time chat interface with history
- Source attribution for each response
- Clear chat button for conversation reset
- Sidebar with knowledge base info

## Project Structure

```
wattmonk-RAG/
├── data/                          # PDF documents
│   ├── Article-690-Photovoltaic-PV-System.pdf
│   ├── Wattmonk (1) (1) (1).pdf
│   └── Wattmonk Information (1).pdf
├── chroma_db/                     # Vector database (auto-created)
├── ingest.py                      # PDF ingestion & embedding pipeline
├── retriever.py                   # Vector search testing
├── rag_pipeline.py                # Core RAG logic (intent, retrieval, generation)
├── app.py                         # Streamlit web interface
├── requirements.txt               # Python dependencies
├── .env                           # API keys (not versioned)
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kaushlendu-Narayan-Parashar/wattmonk-RAG.git
   cd wattmonk-RAG
   ```

2. **Create virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Create `.env` file in the project root
   - Add your Google Gemini API key:
     ```
     GOOGLE_API_KEY=your_key_here
     ```
   - Get a free key from [Google AI Studio](https://ai.google.dev)

## Quick Start

### 1. Ingest Documents
```bash
python ingest.py
```
This will:
- Load 3 PDFs from `data/` folder
- Tag with metadata (source="NEC" or source="Wattmonk")
- Create embeddings using HuggingFace
- Store in ChromaDB at `chroma_db/`

### 2. Test Retriever
```bash
python retriever.py
```
Shows similarity search results with source metadata.

### 3. Run the Web App
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

## Usage Examples

### Via Python
```python
from rag_pipeline import chat

# Ask about NEC codes
result = chat("What are NEC requirements for photovoltaic systems?")
print(result['response'])
print(f"Source: {result['source']}")
print(f"Intent: {result['intent']}")
```

### Via Streamlit
1. Start the app with `streamlit run app.py`
2. Type your question in the chat input
3. View the response with source attribution
4. Click "Clear Chat" to reset history

## How It Works

### 1. Intent Classification
Detects if query relates to:
- **NEC**: Keywords like "code", "article", "electrical", "wire", "circuit"
- **Wattmonk**: Keywords like "service", "permit", "planset", "solar", "PTO"
- **General**: default fallback

### 2. Context Retrieval
- Creates vector embeddings of text
- Searches ChromaDB using cosine similarity
- Filters results by source if intent is domain-specific
- Returns top 3 most relevant chunks

### 3. Response Generation
- Passes context + query to Google Gemini API
- Uses `gemini-2.0-flash` for fast, accurate responses
- Implements retry logic for rate limits
- Falls back gracefully on API errors

## Dependencies

| Package | Purpose |
|---------|---------|
| `chromadb` | Vector database |
| `google-generativeai` | Gemini API client |
| `langchain` | LLM framework |
| `langchain-community` | Community integrations |
| `langchain-huggingface` | HuggingFace embeddings |
| `sentence-transformers` | Embedding models |
| `pypdf` | PDF loading |
| `python-dotenv` | Environment variables |
| `streamlit` | Web UI |
| `fastapi` | (optional) API server |
| `uvicorn` | (optional) ASGI server |

## Configuration

### Models Used
- **Embedding**: `all-MiniLM-L6-v2` (HuggingFace)
- **LLM**: `gemini-2.0-flash` (Google)
- **Vector Store**: ChromaDB (in-memory + persistent)

### Tunable Parameters
Edit `rag_pipeline.py`:
- `chunk_size=500` - Text chunk size for embeddings
- `chunk_overlap=50` - Overlap between chunks
- `k=3` - Number of retrieval results
- `max_retries=3` - API retry attempts

## API Rate Limits

**Free Tier (Google Gemini)**
- 1,500 requests/day
- 15 requests/minute

The app automatically handles rate limits by:
1. Waiting 60 seconds on 429 errors
2. Retrying up to 3 times
3. Returning user-friendly message if all retries fail

## File Descriptions

### `ingest.py`
- Loads PDFs from `data/` folder
- Tags with metadata based on filename
- Chunks text using `RecursiveCharacterTextSplitter`
- Embeds with HuggingFace embeddings
- Stores in ChromaDB

### `retriever.py`
- Tests similarity search functionality
- Shows source and page information
- Verifies ChromaDB initialization

### `rag_pipeline.py`
- **classify_intent()** - Intent classification
- **get_context()** - ChromaDB retrieval with filtering
- **generate_response()** - Gemini API with retry logic
- **chat()** - Main orchestration function

### `app.py`
- Streamlit web interface
- Session state management  
- Chat history persistence
- Source attribution displays

## Troubleshooting

### "API Key not loaded"
- Check `.env` file exists in project root
- Verify `GOOGLE_API_KEY=...` is set
- Run: `python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('GOOGLE_API_KEY')[:10])"`

### "429 RESOURCE_EXHAUSTED" errors
- Free tier rate limit hit
- Wait 60 seconds (app auto-retries)
- Consider using paid Google Cloud account
- Or set up local LLM with Ollama

### "No module named 'chromadb'"
- Run: `pip install -r requirements.txt`

### Streamlit won't start
- Check port 8501 is not in use
- Run: `streamlit run app.py --server.port=8502`

## Future Enhancements

- [ ] Web search integration
- [ ] Multi-language support
- [ ] Document upload via UI
- [ ] Long-term memory with database
- [ ] Response rating & feedback loop
- [ ] API endpoint with FastAPI
- [ ] Citation with page numbers
- [ ] Streaming response display

## License

MIT License - feel free to use this project!

## Author

**Kaushlendu Narayan Parashar**  
📧 parasarkaushlendu@gmail.com  
🔗 GitHub: [@Kaushlendu-Narayan-Parashar](https://github.com/Kaushlendu-Narayan-Parashar)

---

**Happy chatting! 🚀**
