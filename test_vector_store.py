# test_vector_store.py

import sys
from services.vector_store_service import vector_store_service

def main():
    """
    Este script testa a busca vetorial diretamente na base de dados ChromaDB.
    """
    if len(sys.argv) < 2:
        print("❌ Erro: Por favor, forneça uma pergunta para o teste.")
        print("   Exemplo de uso: python test_vector_store.py \"Qual o tema principal?\"")
        return

    query = " ".join(sys.argv[1:])

    print("-" * 50)
    print(f"🔍 Buscando na base de dados pela pergunta: \"{query}\"")
    print("-" * 50)

    try:
        retrieved_chunks = vector_store_service.search(query, n_results=5)

        if not retrieved_chunks:
            print("🚫 Nenhum chunk relevante foi encontrado para esta pergunta.")
            return

        print(f"✅ Sucesso! Foram encontrados {len(retrieved_chunks)} chunks relevantes:\n")

        for i, chunk in enumerate(retrieved_chunks):
            print(f"--- Chunk Relevante #{i + 1} ---")
            print(chunk)
            print("\n" + "="*40 + "\n")

    except Exception as e:
        print(f"🚨 Ocorreu um erro durante o teste: {e}")

if __name__ == "__main__":
    main()