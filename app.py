import os
import numpy as np
import re
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from pypdf import PdfReader
from google import genai
import google.genai.types as types

load_dotenv()

app = Flask(__name__, static_folder="static")

# ===================== CONFIGURAÇÃO GOOGLE GEMINI (2026) =====================
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Modo ultra econômico de tokens
EMBEDDING_MODEL = "models/embedding-001"
GENERATION_MODEL = "gemini-2.5-flash-lite"
FALLBACK_GENERATION_MODELS = [
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-pro",
]
MAX_CONTEXT_CHARS = 700
MAX_OUTPUT_TOKENS = 90


def _normalizar_tokens(texto):
    return set(re.findall(r"\w+", texto.lower(), flags=re.UNICODE))


def _limpar_texto_pdf(texto):
    texto = texto.replace("\n", " ")
    texto = re.sub(r"[\u4e00-\u9fff]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def extrair_contexto_relevante(pergunta, texto_pdf, limite_chars=MAX_CONTEXT_CHARS):
    texto_limpo = _limpar_texto_pdf(texto_pdf)[:3000]
    sentencas = re.split(r"(?<=[\.!?])\s+", texto_limpo)
    termos_pergunta = _normalizar_tokens(pergunta)

    ranqueadas = []
    for sentenca in sentencas:
        s = sentenca.strip()
        if len(s) < 50:
            continue
        score = len(termos_pergunta.intersection(_normalizar_tokens(s)))
        if score > 0:
            ranqueadas.append((score, s))

    if not ranqueadas:
        return texto_limpo[:limite_chars]

    ranqueadas.sort(key=lambda item: item[0], reverse=True)
    escolhidas = []
    tamanho_atual = 0
    for _, trecho in ranqueadas:
        if trecho in escolhidas:
            continue
        novo_tamanho = tamanho_atual + len(trecho) + 1
        if novo_tamanho > limite_chars:
            continue
        escolhidas.append(trecho)
        tamanho_atual = novo_tamanho
        if len(escolhidas) >= 2:
            break
        if tamanho_atual >= limite_chars:
            break

    if not escolhidas:
        return texto_limpo[:limite_chars]

    return "\n".join(escolhidas)


def gerar_resposta_local(pergunta, docs):
    termos_pergunta = _normalizar_tokens(pergunta)

    if not docs:
        return (
            "Gemini indisponível por limite de cota no momento. Não encontrei trecho claro no PDF para responder "
            "com segurança. Tente reformular sua pergunta com termos mais específicos."
        )

    linhas = []
    fontes = []
    for doc in docs[:2]:
        nome = doc["nome"]
        texto_pdf = doc["texto"]
        fontes.append(nome)

        texto_limpo = _limpar_texto_pdf(texto_pdf)
        sentencas = re.split(r"(?<=[\.!?])\s+", texto_limpo)
        melhor_trecho = ""
        melhor_score = -1

        for sentenca in sentencas:
            s = sentenca.strip()
            if len(s) < 50:
                continue
            score = len(termos_pergunta.intersection(_normalizar_tokens(s)))
            if score > melhor_score:
                melhor_score = score
                melhor_trecho = s

        if not melhor_trecho and sentencas:
            melhor_trecho = sentencas[0].strip()

        melhor_trecho = melhor_trecho[:160].rstrip() + ("..." if len(melhor_trecho) > 160 else "")
        linhas.append(f"- {nome}: {melhor_trecho}")

    return (
        "Gemini indisponível por limite de cota (429).\n"
        "Resumo local com base no PDF:\n"
        + "\n".join(linhas)
        + f"\n\nFontes: {', '.join(fontes)}\n"
        "Para voltar às respostas completas do Gemini: aumente cota/billing da chave ou troque para outra chave com saldo."
    )


def gerar_resposta_gemini(prompt):
    modelos_tentativa = []
    modelo_env = os.getenv("GEMINI_MODEL", "").strip()
    if modelo_env:
        modelos_tentativa.append(modelo_env)
    modelos_tentativa.extend([GENERATION_MODEL, *FALLBACK_GENERATION_MODELS])

    modelos_unicos = []
    for modelo in modelos_tentativa:
        if modelo and modelo not in modelos_unicos:
            modelos_unicos.append(modelo)

    ultimo_erro = None
    for modelo in modelos_unicos:
        try:
            response = client.models.generate_content(
                model=modelo,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    top_p=0.7,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                ),
            )
            resposta = (response.text or "Não foi possível gerar resposta no momento.").strip()
            return resposta, modelo
        except Exception as e:
            ultimo_erro = e
            erro_txt = str(e)
            if "429" in erro_txt or "RESOURCE_EXHAUSTED" in erro_txt:
                raise RuntimeError("GEMINI_QUOTA_EXCEEDED") from e
            if "404" in erro_txt or "NOT_FOUND" in erro_txt:
                continue
            raise

    raise RuntimeError(
        f"Nenhum modelo Gemini disponível para generate_content. Último erro: {ultimo_erro}"
    )

# ===================== CARREGAMENTO DOS PDFs =====================
pdfs = {}  # nome_do_pdf: {"text": texto_completo, "embedding": array}


def selecionar_pdfs_relevantes(pergunta, limite=2):
    termos = _normalizar_tokens(pergunta)
    if "peic" in termos and "2025" in termos and "2026" in termos:
        selecionados = []
        for nome_pdf, meta in pdfs.items():
            caminho = meta.get("path", "").lower().replace("\\", "/")
            nome_base = os.path.basename(nome_pdf).lower()
            if "/peic_2025/" in caminho and "peic" in nome_base:
                selecionados.append(nome_pdf)
                break
        for nome_pdf, meta in pdfs.items():
            caminho = meta.get("path", "").lower().replace("\\", "/")
            nome_base = os.path.basename(nome_pdf).lower()
            if "/peic_2026/" in caminho and "peic" in nome_base and nome_pdf not in selecionados:
                selecionados.append(nome_pdf)
                break
        if selecionados:
            return selecionados[:limite]

    candidatos = []

    for nome_pdf, meta in pdfs.items():
        caminho = meta.get("path", "")
        contexto_arquivo = f"{nome_pdf} {caminho}".lower()
        score = 0

        for termo in termos:
            if termo in contexto_arquivo:
                score += 3

        if "peic" in termos and "peic" in contexto_arquivo:
            score += 6
        if "2025" in termos and "2025" in contexto_arquivo:
            score += 5
        if "2026" in termos and "2026" in contexto_arquivo:
            score += 5

        if score > 0:
            candidatos.append((score, nome_pdf))

    if not candidatos:
        return list(pdfs.keys())[:limite]

    candidatos.sort(key=lambda item: item[0], reverse=True)
    selecionados = []
    for _, nome_pdf in candidatos:
        if nome_pdf not in selecionados:
            selecionados.append(nome_pdf)
        if len(selecionados) >= limite:
            break

    if "2025" in termos and "2026" in termos:
        faltantes = []
        if not any("2025" in pdfs[n]["path"].lower() for n in selecionados):
            faltantes.append("2025")
        if not any("2026" in pdfs[n]["path"].lower() for n in selecionados):
            faltantes.append("2026")

        for ano in faltantes:
            for nome_pdf, meta in pdfs.items():
                caminho = meta.get("path", "").lower()
                if ano in caminho and nome_pdf not in selecionados:
                    selecionados.append(nome_pdf)
                    break

    return selecionados[:limite]

def carregar_pdfs():
    global pdfs
    pdfs = {}
    pasta = "pdfs"
    if not os.path.exists(pasta):
        os.makedirs(pasta)
    
    print("🔄 Carregando lista de PDFs...")
    for root, dirs, files in os.walk(pasta):
        for arquivo in files:
            if arquivo.lower().endswith(".pdf"):
                caminho = os.path.join(root, arquivo)
                chave_pdf = os.path.relpath(caminho, pasta).replace("\\", "/")
                print(f"   ✓ {chave_pdf}")
                
                # Apenas armazena o caminho (não extrai texto agora)
                pdfs[chave_pdf] = {"path": caminho, "text": None}
    
    print(f"✅ {len(pdfs)} PDFs carregados e prontos!")

carregar_pdfs()

# ===================== ROTAS =====================
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    pergunta = data.get("pergunta", "").strip()
    
    if not pergunta or not pdfs:
        return jsonify({"resposta": "Nenhum PDF encontrado ou pergunta vazia."})
    
    pdfs_relevantes = selecionar_pdfs_relevantes(pergunta, limite=2)
    docs = []

    for nome_pdf in pdfs_relevantes:
        if pdfs[nome_pdf]["text"] is None:
            try:
                reader = PdfReader(pdfs[nome_pdf]["path"])
                texto = " ".join([page.extract_text() or "" for page in reader.pages])
                pdfs[nome_pdf]["text"] = texto[:3000]
            except Exception as e:
                return jsonify({"resposta": f"Erro ao processar PDF: {e}"})

        docs.append({
            "nome": nome_pdf,
            "texto": pdfs[nome_pdf]["text"],
        })

    melhor_pdf = docs[0]["nome"] if docs else list(pdfs.keys())[0]
    contexto_relevante = "\n\n".join(
        [f"[{d['nome']}]\n{extrair_contexto_relevante(pergunta, d['texto'])}" for d in docs]
    )
    melhor_score = 100.0

    if not os.getenv("GOOGLE_API_KEY"):
        return jsonify({
            "resposta": "A chave GOOGLE_API_KEY não foi encontrada. Configure o arquivo .env para usar o Gemini.",
            "pdf_usado": melhor_pdf,
            "confiança": melhor_score
        }), 500

    try:
        prompt = (
            "PT-BR. Resposta curta e inteligente em no máximo 3 linhas, usando apenas o CONTEXTO. "
            "Se não houver dado suficiente, diga isso em 1 linha.\n\n"
            f"PERGUNTA: {pergunta}\n\n"
            f"CONTEXTO ({melhor_pdf}):\n{contexto_relevante}"
        )

        resposta, modelo_usado = gerar_resposta_gemini(prompt)
    except Exception as e:
        if "GEMINI_QUOTA_EXCEEDED" in str(e):
            resposta_local = gerar_resposta_local(pergunta, docs)
            return jsonify({
                "resposta": resposta_local,
                "pdf_usado": melhor_pdf,
                "confiança": melhor_score,
                "modelo_usado": "fallback-local",
                "aviso": "Quota do Gemini excedida (429)."
            })
        return jsonify({
            "resposta": f"Erro ao consultar o Gemini: {e}",
            "pdf_usado": melhor_pdf,
            "confiança": melhor_score
        }), 500

    return jsonify({
        "resposta": resposta,
        "pdf_usado": melhor_pdf,
        "confiança": melhor_score,
        "modelo_usado": modelo_usado
    })

# ===================== INICIALIZAÇÃO =====================
if __name__ == "__main__":
    print("🚀 Chatbot PDF com Gemini iniciado!")
    print("   Acesse: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)