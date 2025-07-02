# data_processing/ingest.py

import sys
import os

# ✅ Garante que o diretório raiz do projeto esteja no sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.vector_store_service import vector_store_service
from core.pdf_utils import ler_pdf  # <- utilitário para leitura de PDF

def load_documents_from_directory(directory_path: str) -> list[str]:
    """
    Carrega e extrai texto de arquivos .pdf em um diretório, mostrando quais foram carregados.
    """
    print(f"\nCarregando documentos PDF do diretório: '{directory_path}'...")
    texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            try:
                caminho = os.path.join(directory_path, filename)
                texto_pdf = ler_pdf(caminho)
                texts.append(texto_pdf)
                print(f"  - Arquivo PDF carregado: {filename}")
            except Exception as e:
                print(f"❌ Erro ao ler o arquivo {filename}: {e}")
    
    print(f"Total de {len(texts)} arquivos PDF carregados.")
    return texts


def clean_text(text: str) -> str:
    """
    Aplica uma série de regras de limpeza robustas, com tratamento aprimorado
    de quebras de linha para preservar a estrutura de parágrafos.
    """
    print("\nIniciando a limpeza robusta do texto...")

    start_match = re.search(r'^\s*INTRODUÇÃO\s*$', text, re.MULTILINE | re.IGNORECASE)
    end_match = re.search(r'^\s*NOTAS\s*$', text, re.MULTILINE | re.IGNORECASE)

    text_body = text
    if start_match:
        start_index = start_match.start()
        text_body = text[start_index:]
        print("  - Marcador 'INTRODUÇÃO' encontrado. Processando a partir deste ponto.")
        
        end_match_in_body = re.search(r'^\s*NOTAS\s*$', text_body, re.MULTILINE | re.IGNORECASE)
        if end_match_in_body:
            end_index = end_match_in_body.start()
            text_body = text_body[:end_index]
            print("  - Marcador 'NOTAS' encontrado. O corpo principal do texto foi isolado.")

    lines = text_body.splitlines()
    cleaned_lines = []
    for line in lines:
        if re.search(r'\.{5,}', line): continue
        if re.search(r'^\s*(PRIMEIRA SEÇÃO|SEGUNDA SEÇÃO|TERCEIRA SEÇÃO|CAPITULO|LIVRO)\s*[IVXLCDM\dº]*.*$', line, re.IGNORECASE): continue
        if re.search(r'^\s*”?\s*Livro\s+[IVXLCDM\dº]+\.?.*$', line): continue
        if re.search(r'^\s*\d+\s*$', line): continue
        cleaned_line = re.sub(r'\s*\(\d+\)\s*', ' ', line).strip()
        if cleaned_line:
            cleaned_lines.append(cleaned_line)

    text_with_proper_lines = "\n".join(cleaned_lines)

    # Junta linhas que não são parágrafos reais
    processed_text = re.sub(r'\n{2,}', '_PARAGRAPH_BREAK_', text_with_proper_lines)
    processed_text = re.sub(r'\n', ' ', processed_text)
    final_text = processed_text.replace('_PARAGRAPH_BREAK_', '\n\n')

    print("Limpeza do texto concluída.")
    return final_text.strip()


def main():
    """
    Função principal que orquestra o carregamento, limpeza e ingestão dos dados.
    """
    print("Iniciando o processo de ingestão de dados...")
    documents_path = "documents/"
    raw_texts = load_documents_from_directory(documents_path)
    if not raw_texts:
        print("\nNenhum documento encontrado na pasta 'documents/'. Encerrando.")
        return

    full_text = "\n".join(raw_texts)
    cleaned_text = clean_text(full_text)

    if not cleaned_text or cleaned_text.isspace():
        print("\n❌ ERRO CRÍTICO: Após a limpeza, o texto ficou vazio.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(cleaned_text)
    print(f"\nTexto dividido em {len(chunks)} chunks após a limpeza.")

    if not chunks:
        print("\n❌ ERRO CRÍTICO: Nenhum chunk foi gerado a partir do texto limpo.")
        return

    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

    print("\nAdicionando chunks limpos à base vetorial...")
    vector_store_service.add_documents(documents=chunks, ids=chunk_ids, metadatas=None)
    print("\n✅ Ingestão de dados concluída com sucesso!")


if __name__ == "__main__":
    main()
