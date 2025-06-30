🤖 RAG com LLM Local e Base VetorialEste projeto é uma implementação de um sistema de RAG (Retrieval-Augmented Generation) utilizando um modelo de linguagem grande (LLM) rodando localmente com Ollama e uma base de dados vetorial com ChromaDB. A API é construída com FastAPI.O sistema é capaz de receber perguntas, buscar informações relevantes em uma base de documentos customizada e gerar respostas coesas e contextuais.✨ FuncionalidadesIngestão de Documentos: Processa arquivos de texto (.txt) para alimentar a base de conhecimento.Limpeza de Texto: Rotinas de pré-processamento para remover ruídos e formatações indesejadas dos textos.Busca Vetorial: Utiliza sentence-transformers e ChromaDB para encontrar os trechos de texto mais relevantes para uma dada pergunta.Geração de Resposta com LLM Local: Integra-se com qualquer modelo suportado pelo Ollama (neste caso, deepseek-v2) para gerar respostas.API Robusta: Uma API feita com FastAPI para servir o modelo e facilitar a integração com frontends.🛠️ Tecnologias UtilizadasBackend: Python 3.8+API: FastAPILLM Local: Ollama (com o modelo deepseek-v2)Base Vetorial: ChromaDBEmbeddings: Sentence TransformersProcessamento de Texto: LangChain (para TextSplitter)🚀 Como Rodar o ProjetoSiga estes passos na ordem correta para configurar e executar a aplicação em seu ambiente local.1. Pré-requisitosAntes de começar, garanta que você tenha os seguintes softwares instalados:Python 3.8+Ollama: É fundamental para rodar o modelo de linguagem localmente. Se ainda não o tiver, baixe e instale a partir do site oficial do Ollama.2. Configuração InicialEstes comandos preparam o ambiente do projeto. Execute-os na raiz do seu projeto.# 1. Clone este repositório (se estiver no GitHub)
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio

# Instale todas as bibliotecas necessárias:
pip install -r requirements.txt

# 3. Configure o ambiente
# Crie um arquivo chamado .env na raiz do projeto e cole o seguinte conteúdo dentro dele:
# LLM_API_KEY="ollama"
# LLM_API_URL="http://localhost:11434/v1/chat/completions"
# (Você pode criar o arquivo com o comando abaixo no Linux/macOS)
echo 'LLM_API_KEY="ollama"\nLLM_API_URL="http://localhost:11434/v1/chat/completions"' > .env

# 4. Adicione seus documentos
# Coloque todos os seus arquivos de texto (.txt) na pasta /documents.

# 5. Baixe o modelo de linguagem local via Ollama
# Este comando fará o download do modelo (pode demorar um pouco) e o deixará pronto para uso.
# Rode este comando uma vez para garantir que o modelo está disponível.
ollama pull deepseek-v2
3. Executando a AplicaçãoCom o ambiente configurado, siga esta sequência para colocar a aplicação no ar.# PASSO 1: Ingestão dos Dados
# Este script lê, limpa e vetoriza seus documentos para a base de dados.
# Rode-o uma vez ou sempre que alterar os documentos.
python -m data_processing.ingest

# PASSO 2: Iniciar o Servidor do Modelo (Terminal 1)
# Este comando ativa o LLM. Este terminal precisa ficar aberto.
ollama run deepseek-v2

# PASSO 3: Iniciar a API Backend (Terminal 2)
# Em um NOVO terminal (deixe o terminal do Ollama rodando), inicie o servidor FastAPI.
# Este também precisa ficar aberto.
uvicorn api.main:app --reload
Após o Passo 3, sua API estará rodando e acessível em http://127.0.0.1:8000.🔬 Como TestarExistem duas formas principais de testar o sistema.Teste da Aplicação Completa (com LLM)Com os servidores do Ollama e do Uvicorn rodando, abra seu navegador e acesse a documentação interativa da API:http://127.0.0.1:8000/docsExpanda o endpoint POST /api/v1/query, clique em "Try it out" e envie uma pergunta no formato JSON.Teste Específico da Base de Dados (sem LLM)Este teste verifica se a busca vetorial está retornando os chunks de texto corretos para uma pergunta. Não é necessário ter o Ollama ou o Uvicorn rodando.# Execute o script de teste passando uma pergunta entre aspas
python test_vector_store.py "Qual é a sua pergunta de teste?"
📂 Estrutura do Projeto/
├── api/
│   ├── main.py             # Lógica da API FastAPI e endpoint principal
│   └── models.py           # Modelos Pydantic para validação de dados
├── core/
│   └── config.py           # Carregamento e gerenciamento de configurações
├── data_processing/
│   └── ingest.py           # Script para ingestão, limpeza e vetorização de dados
├── documents/
│   └── (coloque seus .txt aqui)
├── services/
│   ├── llm_service.py      # Serviço para se comunicar com o LLM via Ollama
│   └── vector_store_service.py # Serviço para interagir com o ChromaDB
├── .env                    # Arquivo de configuração local (NÃO ENVIAR PARA O GIT)
├── requirements.txt        # Lista de dependências Python
├── test_vector_store.py    # Script para testar a busca vetorial
└── README.md               # Este arquivo
Este README foi gerado em 30 de junho de 2025.
