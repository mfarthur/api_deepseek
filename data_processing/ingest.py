# data_processing/ingest.py
import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.vector_store_service import vector_store_service

def load_documents_from_directory(directory_path: str) -> list[str]:
    """
    Carrega e extrai texto de arquivos .txt em um diretório, mostrando quais foram carregados.
    """
    print(f"\nCarregando documentos do diretório: '{directory_path}'...")
    texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            try:
                with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                    print(f"  - Arquivo carregado: {filename}")
            except Exception as e:
                print(f"❌ Erro ao ler o arquivo {filename}: {e}")
    
    print(f"Total de {len(texts)} arquivos .txt carregados.")
    return texts

def clean_text(text: str) -> str:
    """
    Aplica uma série de regras de limpeza robustas, com tratamento aprimorado
    de quebras de linha para preservar a estrutura de parágrafos.
    """
    print("\nIniciando a limpeza robusta do texto...")
    
    # 1. Isola o corpo principal do texto (se os marcadores existirem)
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
    
    # 2. Remove padrões de ruído específicos (títulos, sumários, etc.)
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
    
    # 3. --- MELHORIA: Tratamento inteligente de quebras de linha ---
    # Preserva parágrafos (duas ou mais quebras de linha) usando um marcador temporário
    processed_text = re.sub(r'\n{2,}', '_PARAGRAPH_BREAK_', text_with_proper_lines)
    # Junta linhas que foram quebradas no meio de uma frase
    processed_text = re.sub(r'\n', ' ', processed_text)
    # Restaura os parágrafos
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
        print("   Verifique o texto de entrada e as regras no script 'data_processing/ingest.py'.")
        print("   A ingestão de dados foi interrompida para evitar erros.\n")
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
        print("   A ingestão de dados foi interrompida.\n")
        return

    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

    print("\nAdicionando chunks limpos à base vetorial...")
    # Assumindo que o `vector_store_service` foi atualizado para lidar com metadados
    # Por agora, passamos None para metadados para manter a compatibilidade.
    vector_store_service.add_documents(documents=chunks, ids=chunk_ids, metadatas=None)
    print("\n✅ Ingestão de dados concluída com sucesso!")


if __name__ == "__main__":
    main()
