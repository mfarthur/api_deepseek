# interface_streamlit.py

import streamlit as st
import sys
import os
import asyncio
import traceback  # Para depuração de erros

# Garante que o caminho do projeto esteja no sys.path para encontrar os módulos
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Importa os serviços necessários
from services.vector_store_service import vector_store_service
from services.llm_service import llm_service

# --- Configuração da Página do Streamlit ---
st.set_page_config(
    page_title="Chat Filosófico",
    page_icon="🏛️",
    layout="wide"
)

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.title("Painel de Controlo")
    st.markdown("Ajuste os parâmetros do seu assistente filosófico.")
    
    st.divider()

    # Slider para controlar o número de chunks de contexto
    num_resultados = st.slider(
        "Número de trechos de contexto (Relevância):",
        min_value=1,
        max_value=10,
        value=4,
        help="Selecione quantos trechos relevantes dos textos devem ser usados para formular a resposta."
    )
    
    st.divider()
    
    st.subheader("🏛️ Sobre a Ferramenta")
    st.info(
        "Este é um chatbot que utiliza a técnica RAG (Retrieval-Augmented Generation) "
        "para responder a perguntas com base numa vasta coleção de textos filosóficos e literários."
    )

# --- Lógica do Chat ---
st.title("🏛️ Chat Filosófico")
st.markdown("### Converse com os grandes pensadores da história.")

# Inicializa o histórico do chat na sessão do Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Como posso ajudá-lo a explorar estes textos filosóficos hoje?"}]

# Exibe as mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Captura a nova pergunta do utilizador
if prompt := st.chat_input("Faça a sua pergunta..."):
    # Adiciona a mensagem do utilizador ao histórico e exibe-a
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Inicia o fluxo RAG para gerar a resposta do assistente ---
    with st.chat_message("assistant"):
        with st.spinner("A consultar os textos e a formular uma resposta..."):
            try:
                # 1. BUSCA: Encontra os chunks relevantes
                retrieved_results = vector_store_service.search(prompt, n_results=num_resultados)

                if not retrieved_results:
                    response = "Desculpe, não consegui encontrar informações relevantes sobre este tópico nos documentos disponíveis."
                else:
                    # Extrai os textos dos chunks como contexto
                    context_texts = [doc for doc, meta in retrieved_results]

                    # 2. GERAÇÃO: Envia o contexto e a pergunta para o LLM
                    final_answer = asyncio.run(
                        llm_service.get_response(query=prompt, context=context_texts)
                    )

                    # Apenas retorna a resposta (sem exibir fontes)
                    response = final_answer

                # Exibe a resposta final
                st.markdown(response)

                # Adiciona a resposta ao histórico
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                print("--- OCORREU UM ERRO ---")
                traceback.print_exc()
                print("------------------------")
                
                error_message = f"Ocorreu um erro inesperado do tipo `{type(e).__name__}`. Por favor, verifique o terminal para mais detalhes."
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
