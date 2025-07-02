import fitz  # PyMuPDF

def ler_pdf(caminho: str) -> str:
    texto = ""
    with fitz.open(caminho) as doc:
        for pagina in doc:
            texto += pagina.get_text()
    return texto
