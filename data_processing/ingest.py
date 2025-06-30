# data_processing/ingest.py
import os
import re  # ⬅️ 1. Importamos a biblioteca de expressões regulares
from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.vector_store_service import vector_store_service

def load_documents_from_directory(directory_path: str) -> list[str]:
    """ Carrega e extrai texto de arquivos .txt em um diretório. """
    texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            try:
                with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
            except Exception as e:
                print(f"Erro ao ler o arquivo {filename}: {e}")
    return texts

def clean_text(text: str) -> str:
    """
    Aplica uma série de regras de limpeza ao texto.
    """
    print("Iniciando a limpeza do texto...")
    
    # Remove linhas que são sumários ou índices (ex: "Capítulo I .......... 5")
    # Este padrão busca por linhas que contêm 5 ou mais pontos seguidos,
    # opcionalmente seguidos por números de página.
    text = re.sub(r'.*\.{5,}.*\d+\s*$', '', text, flags=re.MULTILINE)

    # Remove linhas que parecem ser apenas números de página ou cabeçalhos/rodapés numéricos
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Remove múltiplas quebras de linha, deixando no máximo duas
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove espaços em branco extras no meio das linhas
    text = re.sub(r' {2,}', ' ', text)
    
    print("Limpeza do texto concluída.")
    return text.strip()


def main():
    print("Iniciando o processo de ingestão de dados...")

    # Carregar documentos
    documents_path = "documents/"
    raw_texts = load_documents_from_directory(documents_path)
    if not raw_texts:
        print("Nenhum documento encontrado na pasta 'documents/'. Encerrando.")
        return

    full_text = "\n".join(raw_texts)

    # ⬇️ 2. APLICAMOS A NOSSA NOVA FUNÇÃO DE LIMPEZA AQUI!
    cleaned_text = clean_text(full_text)

    # Dividir em chunks usando o texto já limpo
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(cleaned_text)
    print(f"Texto dividido em {len(chunks)} chunks.")

    # Gerar IDs únicos para cada chunk
    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

    # Adicionar à base vetorial
    print("Adicionando chunks à base vetorial...")
    vector_store_service.add_documents(documents=chunks, ids=chunk_ids)
    print("Ingestão de dados concluída com sucesso!")


if __name__ == "__main__":
    main()