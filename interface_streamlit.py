# interface_streamlit.py

import streamlit as st
import sys
import os
import re
import asyncio
import traceback # <-- Adicionado para depuração detalhada de erros

# Garante que o caminho do projeto esteja no sys.path para encontrar os módulos
# Isto é crucial para que o Streamlit encontre a pasta 'services'
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
        # Mostra um spinner enquanto a resposta está a ser gerada
        with st.spinner("A consultar os textos e a formular uma resposta..."):
            try:
                # 1. BUSCA (Retrieval): Encontra os chunks relevantes na base de dados
                retrieved_results = vector_store_service.search(prompt, n_results=num_resultados)
                
                if not retrieved_results:
                    response = "Desculpe, não consegui encontrar informações relevantes sobre este tópico nos documentos disponíveis."
                else:
                    # Extrai apenas o texto dos chunks para usar como contexto
                    context_texts = [doc for doc, meta in retrieved_results]
                    
                    # 2. GERAÇÃO (Generation): Envia o contexto e a pergunta para o LLM
                    # Como o nosso llm_service é assíncrono, usamos asyncio.run()
                    final_answer = asyncio.run(
                        llm_service.get_response(query=prompt, context=context_texts)
                    )
                    
                    # Constrói a resposta final com as fontes
                    fontes_texto = "\n\n---\n**Fontes Consultadas:**\n"
                    fontes_unicas = set()
                    for doc, meta in retrieved_results:
                        fonte = meta.get('fonte', 'Desconhecida')
                        if fonte not in fontes_unicas:
                            fontes_unicas.add(fonte)
                            fontes_texto += f"- *{fonte}*\n"
                    
                    response = final_answer + fontes_texto

                # Exibe a resposta final
                st.markdown(response)
                
                # Adiciona a resposta do assistente ao histórico
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                # --- MELHORIA: Depuração de Erros ---
                # Imprime o erro completo no terminal para diagnóstico
                print("--- OCORREU UM ERRO ---")
                traceback.print_exc()
                print("-------------------------")
                
                # Exibe uma mensagem de erro mais informativa na interface
                error_message = f"Ocorreu um erro inesperado do tipo `{type(e).__name__}`. Por favor, verifique o terminal para mais detalhes."
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
