# data_processing/ingest.py
import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.vector_store_service import vector_store_service

def load_documents_from_directory(directory_path: str) -> list[str]:
    """
    Carrega e extrai texto de arquivos .txt em um diretório.
    """
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
    Aplica uma série de regras de limpeza robustas, focadas em textos
    estruturados como livros (ex: Crítica da Razão Pura).
    """
    print("Iniciando a limpeza robusta do texto...")

    # 1. Isolar o corpo principal do texto, removendo cabeçalho e notas de rodapé
    # Tenta encontrar o início do conteúdo principal (INTRODUÇÃO) e o fim (NOTAS)
    start_match = re.search(r'INTRODUÇÃO', text, re.IGNORECASE)
    end_match = re.search(r'NOTAS', text, re.IGNORECASE)

    # Verifica se ambos os marcadores foram encontrados antes de cortar o texto
    if start_match and end_match:
        start_index = start_match.start()
        end_index = end_match.start()
        print(f"Marcadores 'INTRODUÇÃO' e 'NOTAS' encontrados. Extraindo o corpo principal do texto.")
        text = text[start_index:end_index]
    else:
        print("Aviso: Não foi possível encontrar os marcadores 'INTRODUÇÃO' ou 'NOTAS'. A limpeza prosseguirá no texto completo.")

    # 2. Remover linhas de sumário/índice com muitos pontos
    text = re.sub(r'^[IVXLCDM\s—–-]+\s?.*\.{5,}.*$', '', text, flags=re.MULTILINE)

    # 3. Remover linhas que são títulos de seções/capítulos
    text = re.sub(r'^\s*(PRIMEIRA SEÇÃO|SEGUNDA SEÇÃO|TERCEIRA SEÇÃO)\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^\s*(CAPITULO|LIVRO)\s+[IVXLCDM\dº]+.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^\s*[IVXLCDM]+\s*—.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*Analítica dos conceitos\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

    # 4. Remover linhas que parecem ser tabelas de contagem de livros/versos
    text = re.sub(r'^\s*”?\s*Livro\s+[IVXLCDM\dº]+\.?.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*”\s*\d+.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*Total\s+.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Quantos versos tem o original.*$', '', text, flags=re.MULTILINE)

    # 5. Remover marcadores de notas numéricos como (1), (2), (15)
    text = re.sub(r'\(\d+\)', '', text)

    # 6. Remover linhas que são apenas números (possíveis números de página)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # 7. Normalizar quebras de linha e espaços
    # Substituir 3 ou mais quebras de linha por apenas duas (para separar parágrafos)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remover espaços extras no início/fim de cada linha
    text = re.sub(r'^[ \t]+|[ \t]+$', '', text, flags=re.MULTILINE)

    print("Limpeza do texto concluída.")
    return text.strip()


def main():
    """
    Função principal que orquestra o carregamento, limpeza e ingestão dos dados.
    """
    print("Iniciando o processo de ingestão de dados...")
    documents_path = "documents/"
    raw_texts = load_documents_from_directory(documents_path)
    if not raw_texts:
        print("Nenhum documento encontrado na pasta 'documents/'. Encerrando.")
        return

    full_text = "\n".join(raw_texts)
    
    # Aplica a nova função de limpeza
    cleaned_text = clean_text(full_text)

    # Divide o texto limpo em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Mantém um bom tamanho para contexto
        chunk_overlap=200,   # Sobreposição para não perder o contexto entre chunks
        length_function=len
    )
    chunks = text_splitter.split_text(cleaned_text)
    print(f"Texto dividido em {len(chunks)} chunks após a limpeza.")

    # Gera IDs únicos para cada chunk
    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

    # Adiciona os chunks limpos à base vetorial
    print("Adicionando chunks limpos à base vetorial...")
    vector_store_service.add_documents(documents=chunks, ids=chunk_ids)
    print("Ingestão de dados concluída com sucesso!")


if __name__ == "__main__":
    main()
