# services/vector_store_service.py
import chromadb
from sentence_transformers import SentenceTransformer
from core.config import settings
from typing import List, Dict, Any, Optional, Tuple
# --- MELHORIA: Importação explícita dos tipos para tipagem correta ---
from chromadb.api.types import Metadata, IDs

class VectorStoreService:
    """
    Serviço para gerenciar a interação com a base de dados vetorial (ChromaDB).
    Esta versão inclui suporte a metadatos, filtragem e processamento em lotes.
    """
    def __init__(self):
        """
        Inicializa o cliente do ChromaDB e carrega o modelo de embedding.
        """
        print("Iniciando o serviço de base vetorial...")
        self.client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
        
        print(f"Carregando o modelo de embedding: '{settings.EMBEDDING_MODEL_NAME}'...")
        # Carrega o modelo de embedding a partir da configuração
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        print("Modelo de embedding carregado com sucesso.")
        
        # Obtém ou cria a coleção (tabela) na base vetorial
        self.collection = self.client.get_or_create_collection(
            name=settings.VECTOR_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"} # Usar distância de cosseno para similaridade
        )
        print(f"Coleção '{settings.VECTOR_COLLECTION_NAME}' carregada/criada com sucesso.")

    # --- CORREÇÃO: Tipagem de 'ids' e 'metadatas' ajustada para corresponder ao ChromaDB ---
    def add_documents(self, documents: List[str], ids: IDs, metadatas: Optional[List[Metadata]] = None):
        """
        Gera embeddings e adiciona documentos, com seus metadados, à coleção.
        O processo é feito em lotes para otimizar o uso de memória.

        Args:
            documents (List[str]): A lista de textos (chunks) a serem adicionados.
            ids (IDs): Uma lista de IDs únicos para cada documento.
            metadatas (Optional[List[Metadata]]): Uma lista de dicionários de metadados, um para cada documento.
        """
        if not documents:
            print("Nenhum documento para adicionar.")
            return

        batch_size = 100  # Define um tamanho de lote razoável
        for i in range(0, len(documents), batch_size):
            # Cria os lotes para cada um dos parâmetros
            doc_batch = documents[i:i + batch_size]
            id_batch = ids[i:i + batch_size]
            meta_batch = metadatas[i:i + batch_size] if metadatas else None

            print(f"Processando lote {i//batch_size + 1}: adicionando {len(doc_batch)} documentos...")
            
            try:
                # Gera os embeddings para o lote atual de documentos
                embeddings = self.embedding_model.encode(doc_batch).tolist()
                
                # Adiciona o lote à coleção do ChromaDB
                self.collection.add(
                    embeddings=embeddings,
                    documents=doc_batch,
                    metadatas=meta_batch,
                    ids=id_batch
                )
            except Exception as e:
                print(f"❌ Erro ao processar o lote {i//batch_size + 1}: {e}")
                # Continua para o próximo lote se houver um erro
                continue
        
        print(f"\n✅ Total de {len(documents)} documentos adicionados à coleção.")

    # --- CORREÇÃO: Tipo de retorno ajustado para ser mais preciso ---
    def search(self, query: str, n_results: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Metadata]]:
        """
        Busca por documentos similares a uma consulta, com a opção de aplicar um filtro de metadados.

        Args:
            query (str): O texto da pergunta do usuário.
            n_results (int): O número de resultados a serem retornados.
            filter (Optional[Dict[str, Any]]): Um dicionário para filtrar a busca. Ex: {"autor": "Kant"}

        Returns:
            List[Tuple[str, Metadata]]: Uma lista de tuplas, onde cada tupla contém
                                         o texto do chunk e seus metadados.
        """
        try:
            # Gera o embedding para a consulta do usuário
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Realiza a busca na coleção, aplicando o filtro se ele for fornecido
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter
            )
            
            # Lógica de extração de resultados mais segura e explícita
            documents_list = results.get('documents')
            metadatas_list = results.get('metadatas')

            # Verifica se as listas não são nulas ou vazias antes de prosseguir
            if not documents_list or not metadatas_list:
                return []

            # ChromaDB retorna uma lista contendo uma lista de resultados para uma única consulta
            retrieved_docs = documents_list[0]
            retrieved_metas = metadatas_list[0]

            if not retrieved_docs or not retrieved_metas:
                return []

            # Retorna uma lista de tuplas (documento, metadados)
            return list(zip(retrieved_docs, retrieved_metas))
        
        except Exception as e:
            print(f"❌ Erro durante a busca na base vetorial: {e}")
            return []

# Instância única do serviço para ser usada em toda a aplicação
vector_store_service = VectorStoreService()
