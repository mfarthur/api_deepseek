# services/vector_store_service.py
import chromadb
from sentence_transformers import SentenceTransformer
from core.config import settings

class VectorStoreService:
    def __init__(self):
        # Inicializa o cliente do ChromaDB com persistência no disco
        self.client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
        # Carrega o modelo de embedding
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        # Obtém ou cria a coleção (tabela) na base vetorial
        self.collection = self.client.get_or_create_collection(
            name=settings.VECTOR_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"} # Usar distância de cosseno para similaridade
        )

    def add_documents(self, documents: list[str], ids: list[str]):
        """ Gera embeddings e adiciona documentos à coleção. """
        embeddings = self.embedding_model.encode(documents).tolist()
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=ids
        )
        print(f"Adicionados {len(documents)} documentos à coleção.")

    def search(self, query: str, n_results: int = 3) -> list[str]:
        """ Busca por documentos similares a uma consulta. """
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results['documents'][0] if results['documents'] else []

# Instância única do serviço para ser usada em toda a aplicação
vector_store_service = VectorStoreService()