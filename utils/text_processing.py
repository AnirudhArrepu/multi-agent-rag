import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import SpacyTextSplitter
from sentence_transformers import SentenceTransformer

class SplitEmbedDB:
    def __init__(self):
        self.chroma_client = chromadb.Client()

        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        self.splitter = SpacyTextSplitter(
            separator="\n\n",
            chunk_size=100,
            chunk_overlap=20,
            length_function=len
        )

    def split_text(self, text):
        return self.splitter.split_text(text)
    
    def embed_text(self, texts):
        return self.embedder.encode(texts, convert_to_tensor=False).tolist()
    
    def store_embeddings(self, text, project_id):
        chunks = self.split_text(text)
        embeddings = self.embed_text(chunks)
        
        collection = self.chroma_client.get_or_create_collection(name=project_id)

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[f"{project_id}_doc_{i}"]
            )
        
        return f"Stored {len(chunks)} chunks under project '{project_id}'"

    def query_project(self, project_id, query_text, n_results=5):
        collection = self.chroma_client.get_collection(name=project_id)

        query_embedding = self.embed_text([query_text])[0]

        # Query Chroma
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        return results['documents']
