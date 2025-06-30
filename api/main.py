# api/main.py
from fastapi import FastAPI, HTTPException
from .models import QueryRequest, QueryResponse
from services.vector_store_service import vector_store_service
from services.llm_service import llm_service

app = FastAPI(
    title="RAG Backend API",
    description="API para buscar informações em documentos e gerar respostas com LLM.",
    version="1.0.0"
)

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok"}

@app.post("/api/v1/query", response_model=QueryResponse, tags=["RAG"])
async def handle_query(request: QueryRequest):
    """
    Recebe uma pergunta, busca contexto relevante e gera uma resposta.
    """
    try:
        # 1. Buscar contexto relevante na base vetorial
        context = vector_store_service.search(request.question, n_results=3)

        if not context:
            return QueryResponse(
                answer="Não encontrei informações relevantes em meus documentos para responder a essa pergunta.",
                context=[]
            )

        # 2. Gerar uma resposta usando o LLM com o contexto
        answer = await llm_service.get_response(query=request.question, context=context)

        return QueryResponse(answer=answer, context=context)

    except Exception as e:
        print(f"Erro no endpoint /query: {e}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno no servidor.")