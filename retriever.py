import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

def test_retriever():
    """Test the ChromaDB retriever with a sample query."""
    
    chroma_db_path = "chroma_db"
    
    # Check if ChromaDB exists
    if not os.path.exists(chroma_db_path):
        print(f"Error: {chroma_db_path} not found!")
        print("Please run ingest.py first to create the vector database.")
        return
    
    print("Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print(f"Loading ChromaDB from {chroma_db_path}...")
    vectorstore = Chroma(
        persist_directory=chroma_db_path,
        embedding_function=embeddings,
        collection_name="wattmonk_rag"
    )
    
    # Test query
    test_query = "photovoltaic system installation requirements"
    
    print(f"\n{'='*60}")
    print(f"Testing similarity search with query:")
    print(f'"{test_query}"')
    print(f"{'='*60}\n")
    
    # Perform similarity search
    results = vectorstore.similarity_search(test_query, k=3)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found {len(results)} relevant documents:\n")
    
    for i, result in enumerate(results, 1):
        source = result.metadata.get("source", "Unknown")
        page = result.metadata.get("page", "N/A")
        
        print(f"Result {i}:")
        print(f"  Source: {source}")
        print(f"  Page: {page}")
        print(f"  Content Preview: {result.page_content[:150]}...")
        print()
    
    print(f"{'='*60}")
    print("✓ Retriever test completed successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        test_retriever()
    except Exception as e:
        print(f"Error during retriever test: {e}")
        raise
