# api/main.py

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import json
from typing import List, Dict, Any, Optional, Tuple

# Importa os nossos serviços
from services.vector_store_service import vector_store_service
# O llm_service original não será usado diretamente aqui, pois precisamos de streaming

# --- Modelos Pydantic ---

# Modelo para a nossa API de teste original
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    context: list[str]

# Modelos para a integração com Open WebUI
class OpenWebUIMessage(BaseModel):
    role: str
    content: str

class OpenWebUIRequest(BaseModel):
    model: str
    messages: List[OpenWebUIMessage]
    stream: bool = True

# --- Configuração da Aplicação FastAPI ---
app = FastAPI(
    title="RAG Backend API",
    description="API para buscar informações em documentos e gerar respostas com LLM, com suporte para Open WebUI.",
    version="2.0.0"
)

# --- Endpoint de Teste Original (mantido para depuração) ---
@app.post("/api/v1/query", response_model=QueryResponse, tags=["Teste RAG (Não-Streaming)"])
async def handle_query(request: QueryRequest):
    """
    Recebe uma pergunta, busca contexto e gera uma resposta (não-streaming).
    Útil para testes rápidos.
    """
    try:
        retrieved_results = vector_store_service.search(request.question, n_results=3)
        context_texts = [doc for doc, meta in retrieved_results]

        if not context_texts:
            return QueryResponse(answer="Não encontrei informações relevantes...", context=[])

        # Para uma resposta não-streaming, podemos usar uma lógica mais simples
        # (Esta parte pode ser adaptada do llm_service original se necessário)
        prompt_for_llm = f"Contexto: {context_texts}\n\nPergunta: {request.question}\n\nResposta:"
        # ... (lógica para chamar o LLM e obter uma resposta completa) ...
        answer = "Resposta de teste (implementação de não-streaming necessária)."

        return QueryResponse(answer=answer, context=context_texts)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- NOVO Endpoint para Integração com Open WebUI ---
@app.post("/api/v1/rag", tags=["Integração RAG (Open WebUI)"])
async def rag_handler(request: OpenWebUIRequest):
    """
    Endpoint compatível com a funcionalidade RAG do Open WebUI.
    Recebe o histórico da conversa, adiciona o contexto e faz o stream da resposta do LLM.
    """
    # 1. Extrair a pergunta mais recente do utilizador
    user_query = ""
    if request.messages:
        user_query = request.messages[-1].content
    if not user_query:
        raise HTTPException(status_code=400, detail="Nenhuma pergunta de utilizador encontrada.")

    # 2. Fazer a busca vetorial para obter o contexto relevante
    print(f"Buscando contexto para a pergunta: '{user_query}'")
    retrieved_results = vector_store_service.search(user_query, n_results=4)
    context_texts = [doc for doc, meta in retrieved_results]
    context_str = "\n\n---\n\n".join(context_texts)

    # 3. Construir o prompt final para o LLM, injetando o contexto
    # Mantemos o histórico da conversa para dar contexto ao LLM
    prompt_for_llm = f"""Você é um assistente de IA especialista. Use o contexto fornecido abaixo para responder à pergunta do usuário de forma completa e coesa.
Se a resposta não estiver no contexto, diga que você não tem informações sobre isso nos documentos fornecidos.

### Contexto Relevante ###
{context_str}
---
### Pergunta do Utilizador ###
{user_query}

### Resposta ###
"""
    # Substituímos a última mensagem do utilizador pelo nosso prompt enriquecido
    request.messages[-1].content = prompt_for_llm

    # 4. Fazer o stream da resposta do Ollama de volta para o Open WebUI
    async def stream_generator():
        async with httpx.AsyncClient() as client:
            ollama_url = "http://localhost:11434/api/chat"
            payload = {
                "model": request.model,
                "messages": [msg.dict() for msg in request.messages],
                "stream": True
            }
            
            try:
                async with client.stream("POST", ollama_url, json=payload, timeout=None) as response:
                    response.raise_for_status() # Lança um erro para respostas 4xx/5xx
                    async for chunk in response.aiter_bytes():
                        # O Ollama faz o stream de objetos JSON separados por novas linhas.
                        # Nós reencaminhamos estes chunks diretamente para o Open WebUI.
                        yield chunk
            except httpx.HTTPStatusError as e:
                error_body = await e.response.aread()
                print(f"Erro na chamada ao Ollama: {e.response.status_code} - {error_body.decode()}")
                error_message = json.dumps({"error": f"Erro ao comunicar com o Ollama: {error_body.decode()}"})
                yield error_message.encode('utf-8')
            except Exception as e:
                print(f"Erro inesperado no stream: {e}")
                error_message = json.dumps({"error": f"Erro inesperado no servidor: {str(e)}"})
                yield error_message.encode('utf-8')

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

