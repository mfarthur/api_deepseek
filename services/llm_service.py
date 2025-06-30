# services/llm_service.py
import httpx
from core.config import settings

class LLMService:
    def __init__(self):
        self.api_key = settings.LLM_API_KEY
        self.api_url = settings.LLM_API_URL
        # Usar httpx.AsyncClient para chamadas assíncronas
        self.client = httpx.AsyncClient(headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def _build_prompt(self, query: str, context: list[str]) -> str:
        """ Monta o prompt final para o LLM. """
        context_str = "\n\n".join(context)
        
        prompt = f"""
        Você é um assistente de IA especialista. Use o contexto fornecido abaixo para responder à pergunta do usuário.
        Se a resposta não estiver no contexto, diga que você não tem informações sobre isso. Não invente informações.
        Responda em português.

        Contexto:
        ---
        {context_str}
        ---

        Pergunta do Usuário: {query}

        Resposta:
        """
        return prompt.strip()

    async def get_response(self, query: str, context: list[str]) -> str:
        """ Envia o prompt para o LLM e retorna a resposta. """
        prompt = self._build_prompt(query, context)
        
        payload = {
            "model": "deepseek-v2",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2048,
            "temperature": 0.2
        }

        try:
            response = await self.client.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()  # Lança exceção para status 4xx/5xx
            data = response.json()
            return data['choices'][0]['message']['content']
        except httpx.HTTPStatusError as e:
            print(f"Erro na chamada da API do LLM: {e.response.text}")
            return "Desculpe, ocorreu um erro ao me comunicar com o modelo de linguagem."
        except Exception as e:
            print(f"Um erro inesperado ocorreu: {e}")
            return "Desculpe, ocorreu um erro inesperado."

# Instância única do serviço
llm_service = LLMService()