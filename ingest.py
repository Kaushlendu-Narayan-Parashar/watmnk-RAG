import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

def ingest_documents():
    """Ingest PDFs from data folder into ChromaDB."""
    
    # Define paths
    data_dir = Path("data")
    chroma_db_path = "chroma_db"
    
    # Collection of documents with metadata
    all_documents = []
    
    # PDF files mapping
    pdf_mapping = {
        "Wattmonk (1) (1) (1).pdf": "Wattmonk",
        "Wattmonk Information (1).pdf": "Wattmonk",
        "Article-690-Photovoltaic-PV-System.pdf": "NEC"
    }
    
    print("Loading PDFs...")
    for pdf_file, source in pdf_mapping.items():
        pdf_path = data_dir / pdf_file
        
        if pdf_path.exists():
            print(f"  Loading {pdf_file}...")
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            
            # Add source metadata to all documents from this PDF
            for doc in documents:
                doc.metadata["source"] = source
            
            all_documents.extend(documents)
            print(f"    Loaded {len(documents)} pages from {pdf_file}")
        else:
            print(f"  Warning: {pdf_file} not found at {pdf_path}")
    
    print(f"\nTotal documents loaded: {len(all_documents)}")
    
    # Chunk documents
    print("\nChunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"Total chunks created: {len(chunks)}")
    
    # Initialize embeddings
    print("\nInitializing embeddings with HuggingFace...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create and persist ChromaDB
    print(f"\nCreating ChromaDB at {chroma_db_path}...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_db_path,
        collection_name="wattmonk_rag"
    )
    
    print("✓ Documents successfully ingested and stored in ChromaDB!")
    print(f"✓ Vector database persisted at: {chroma_db_path}")
    
    return vectorstore

if __name__ == "__main__":
    try:
        vectorstore = ingest_documents()
        print("\nIngestion complete!")
    except Exception as e:
        print(f"Error during ingestion: {e}")
        raise
