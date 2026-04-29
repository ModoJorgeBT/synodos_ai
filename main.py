import os
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END
#from langchain_ollama import ChatOllama Nota: no voy a usar OLLAMA
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
# Importamos los prompts desde el archivo agents.py
from agents import ELIADE_SYSTEM_PROMPT, CIORAN_SYSTEM_PROMPT, MODERATOR_SYSTEM_PROMPT
from sentence_transformers import CrossEncoder
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import sys
import math

# ==========================================
# 1. CONFIGURACIÃ“N DE COMPONENTES LOCALES
# ==========================================

# Cargamos el modelo de embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Reranker de pasajes para mejorar precisión contextual (mitiga "lost in the middle")
RERANK_TOP_K = 20
RERANK_KEEP_N = 4
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = None
try:
    reranker = CrossEncoder(RERANK_MODEL)
    print(f"Reranker activo: {RERANK_MODEL}")
except Exception as e:
    print(f"Advertencia: no se pudo cargar reranker ({RERANK_MODEL}): {e}")
    print("Se usara el ranking original de Chroma hasta resolver cache/espacio.")

# Conectamos con la base de datos vectorial local
if os.path.exists("./db_synodos_local"):
    vector_db = Chroma(persist_directory="./db_synodos_local", embedding_function=embeddings)
else:
    print("Alerta: No se encuentra db_synodos_local. Ejecuta ingest_local.py primero.")
    vector_db = None

# Configuracion optimizada para Ryzen 7 3700U
#llm = ChatOllama(
#    model="gemma2:2b", 
#  temperature=0.7,
#    top_k=20,
#    top_p=0.80,
#    repeat_penalty=1.2,
#    num_ctx=1024,
#    low_vram=True,
#    timeout=120,
#    num_thread=4
#)

# Cargar variables de entorno desde un archivo .env por seguridad
load_dotenv()

# Configuracion HÃ­brida: Razonamiento en Azure, Datos en Local
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), # El nombre que pusiste en AI Foundry
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    
    # ParÃ¡metros de personalizaciÃ³n (TraducciÃ³n de tu Ollama)
    temperature=0.7, #Creatividad
    top_p=0.8,       # Diversidad lexica
    max_tokens=1024, # Equivalente a num_ctx // MÃ¡ximo de respuesta
    frequency_penalty=0.4, # Equivalente a repeat_penalty repeticiones
)

# --- FUNCION DE METRICA MATEMATICA ---
def calcular_similitud_vectores(texto1, texto2):
    try:
        # Limpiamos los prefijos de nombre para comparar solo la esencia del pensamiento
        t1 = re.sub(r"^(ELIADE|CIORAN|MODERADOR):\s*", "", texto1, flags=re.IGNORECASE).strip()
        t2 = re.sub(r"^(ELIADE|CIORAN|MODERADOR):\s*", "", texto2, flags=re.IGNORECASE).strip()
        
        vec1 = embeddings.embed_query(t1)
        vec2 = embeddings.embed_query(t2)
        
        sim = cosine_similarity([vec1], [vec2])[0][0]
        return round(float(sim), 4)
    except Exception as e:
        print(f"Error en similitud: {e}")
        return 0.0

# ==========================================
# 2. DEFINICION DEL ESTADO (AgentState)
# ==========================================

class AgentState(TypedDict):
    messages: Annotated[List[tuple], "Historial de mensajes"]
    turn: int
    topic: str
    memory_summary: str
    eval_history: List[dict]
    correction_log: List[str]

# ==========================================
# 3. LOGICA DE NODOS CON FILTRADO RAG
# ==========================================

def call_agent_with_rag(state: AgentState, system_prompt: str, name: str, autor_filtro: str = None):
    # La consulta al RAG es el último mensaje o el tema inicial
    query = state["messages"][-1][1] if state["messages"] else state["topic"]

    contexto = ""
    contexts_for_eval = []
    if vector_db:
        search_kwargs = {"k": RERANK_TOP_K}
        if autor_filtro:
            search_kwargs["filter"] = {"autor": autor_filtro}
        docs = vector_db.similarity_search(query, **search_kwargs)
        docs = rerank_docs(query, docs, keep_n=RERANK_KEEP_N)
        contexts_for_eval = [d.page_content for d in docs]
        contexto = f"\n\n[CONTEXTO RAG DE {name.upper()}]:\n" + "\n".join([f"- {d.page_content}" for d in docs])

    full_prompt = (
        f"{system_prompt}{contexto}\n\n"
        f"Instrucción: Responde SOLO como {name}. Sé breve y directo. "
        f"No escribas el prefijo '{name}:' al inicio."
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", full_prompt)
    ] + build_memory_messages(state))

    # Bloquea que el agente imite a otros, sin bloquearse a sí mismo.
    stop_tokens = [f"{other}:" for other in ["ELIADE", "CIORAN", "MODERADOR"] if other != name]
    response = llm.invoke(prompt_template.format_messages(), stop=stop_tokens)

    content = (response.content or "").strip()
    if not content:
        # Reintento de respaldo para evitar turnos vacíos por truncado agresivo.
        retry_messages = prompt_template.format_messages() + [
            ("user", f"Continúa el debate como {name} en 3-5 frases, sin prefijo de nombre.")
        ]
        retry_response = llm.invoke(retry_messages)
        content = (retry_response.content or "").strip() or "(Sin respuesta generada)"

    content = re.sub(rf"^\s*{re.escape(name)}:\s*", "", content, flags=re.IGNORECASE).strip()
    ragas_scores = evaluate_ragas_turn(
        user_input=query,
        response=content,
        retrieved_contexts=contexts_for_eval,
        reference=state.get("topic", query)
    )
    new_message = ("assistant", f"{name}: {content}")
    eval_history = state.get("eval_history", [])
    eval_history = eval_history + [{
        "turn": state["turn"] + 1,
        "agent": name,
        **ragas_scores
    }]
    return {
        "messages": state["messages"] + [new_message],
        "turn": state["turn"] + 1,
        "memory_summary": state.get("memory_summary", ""),
        "eval_history": eval_history
    }

def rerank_docs(query: str, docs, keep_n: int = 4):
    if not docs:
        return docs
    if reranker is None:
        return docs[:keep_n]
    try:
        pairs = [[query, d.page_content] for d in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: float(x[0]), reverse=True)
        return [doc for _, doc in ranked[:keep_n]]
    except Exception as e:
        print(f"Advertencia reranker: {e}. Usando ranking original de Chroma.")
        return docs[:keep_n]

def build_memory_messages(state: AgentState):
    # Ventana dinámica: resumen acumulado + 2 últimas intervenciones literales.
    summary = (state.get("memory_summary") or "").strip()
    recent = state["messages"][-2:] if state["messages"] else []
    if summary:
        return [("system", f"[MEMORIA RESUMIDA DEL DEBATE]\n{summary}")] + recent
    return recent

def update_memory_summary(state: AgentState):
    previous_summary = (state.get("memory_summary") or "").strip()
    latest_messages = state["messages"][-3:] if len(state["messages"]) >= 3 else state["messages"]
    block = "\n".join([f"- {m[1]}" for m in latest_messages])

    prompt = (
        "Eres un analista de debate. Actualiza una memoria resumida acumulada.\n"
        "Objetivo: capturar puntos de fricción, acuerdos parciales y evolución.\n"
        "Reglas:\n"
        "1) Máximo 6 viñetas.\n"
        "2) Enfócate en diferencias conceptuales, no estilo.\n"
        "3) Incluye estado actual del desacuerdo en 1 línea final.\n"
        "4) No inventes datos.\n\n"
        f"RESUMEN PREVIO:\n{previous_summary if previous_summary else '(vacío)'}\n\n"
        f"NUEVAS INTERVENCIONES:\n{block}\n\n"
        "Devuelve solo el resumen actualizado."
    )

    try:
        resp = llm.invoke([("system", prompt)])
        return (resp.content or previous_summary or "").strip()
    except Exception as e:
        print(f"Advertencia memoria: {e}. Se mantiene resumen previo.")
        return previous_summary

def evaluate_ragas_turn(user_input: str, response: str, retrieved_contexts: List[str], reference: str):
    # RAGAS single-turn evaluation; values in [0, 1].
    if not retrieved_contexts:
        return {"faithfulness": 0.0, "answer_relevance": 0.0, "context_precision": 0.0}
    try:
        ds = Dataset.from_dict({
            "user_input": [user_input],
            "response": [response],
            "retrieved_contexts": [retrieved_contexts],
            "reference": [reference],
        })
        result = evaluate(
            dataset=ds,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=llm,
            embeddings=embeddings,
            show_progress=False
        )
        row = result.to_pandas().iloc[0].to_dict()
        def safe_score(v):
            try:
                x = float(v)
                return 0.0 if math.isnan(x) or math.isinf(x) else x
            except Exception:
                return 0.0
        return {
            "faithfulness": safe_score(row.get("faithfulness", 0.0)),
            "answer_relevance": safe_score(row.get("answer_relevancy", 0.0)),
            "context_precision": safe_score(row.get("context_precision", 0.0)),
        }
    except Exception as e:
        print(f"Advertencia RAGAS: {e}")
        return {"faithfulness": 0.0, "answer_relevance": 0.0, "context_precision": 0.0}

def eliade_node(state: AgentState):
    return call_agent_with_rag(state, ELIADE_SYSTEM_PROMPT, "ELIADE", autor_filtro="eliade")

def cioran_node(state: AgentState):
    return call_agent_with_rag(state, CIORAN_SYSTEM_PROMPT, "CIORAN", autor_filtro="cioran")

def keyword_score(text: str, keywords: List[str]) -> int:
    t = text.lower()
    return sum(1 for k in keywords if k in t)

def critique_and_rewrite(author_name: str, text: str):
    # Heurística de fidelidad conceptual por autor
    eliade_kw = ["hierofanía", "hierofania", "arquetipo", "illud tempus", "axis mundi", "sagrado", "mito"]
    cioran_kw = ["vacío", "vacio", "nada", "fatiga del ser", "inconveniente de haber nacido", "nihilismo", "desesperación", "desesperacion"]

    score_eliade = keyword_score(text, eliade_kw)
    score_cioran = keyword_score(text, cioran_kw)

    looks_swapped = (
        (author_name == "ELIADE" and score_cioran > score_eliade + 1) or
        (author_name == "CIORAN" and score_eliade > score_cioran + 1)
    )

    if not looks_swapped:
        return text, False, "sin corrección"

    correction_prompt = (
        f"Eres un editor de fidelidad de autor para un debate filosófico.\n"
        f"Autor objetivo: {author_name}\n"
        "Tarea: Reescribe el texto para que sea fiel al autor objetivo, "
        "manteniendo la misma postura argumental central y longitud aproximada.\n"
        "Reglas:\n"
        "1) No cambies el tema del debate.\n"
        "2) No menciones que estás corrigiendo.\n"
        "3) Conserva el idioma español.\n"
        "4) Devuelve solo el texto corregido.\n\n"
        f"Texto original:\n{text}"
    )
    try:
        corrected = (llm.invoke([("system", correction_prompt)]).content or "").strip()
        if corrected:
            return corrected, True, f"corregido por fuga de estilo ({author_name})"
    except Exception as e:
        print(f"Advertencia critica_fuentes: {e}")
    return text, False, "fallo de corrección"

def source_critic_node(state: AgentState):
    messages = list(state["messages"])
    correction_log = list(state.get("correction_log", []))

    # Revisa las dos últimas intervenciones de autores antes del moderador.
    for idx in range(max(0, len(messages) - 2), len(messages)):
        role, content = messages[idx]
        if role != "assistant":
            continue
        if content.startswith("ELIADE:"):
            body = re.sub(r"^ELIADE:\s*", "", content).strip()
            corrected, changed, note = critique_and_rewrite("ELIADE", body)
            if changed:
                messages[idx] = ("assistant", f"ELIADE: {corrected}")
            correction_log.append(f"Turno {state['turn']} ELIADE -> {note}")
        elif content.startswith("CIORAN:"):
            body = re.sub(r"^CIORAN:\s*", "", content).strip()
            corrected, changed, note = critique_and_rewrite("CIORAN", body)
            if changed:
                messages[idx] = ("assistant", f"CIORAN: {corrected}")
            correction_log.append(f"Turno {state['turn']} CIORAN -> {note}")

    return {
        "messages": messages,
        "turn": state["turn"],
        "memory_summary": state.get("memory_summary", ""),
        "eval_history": state.get("eval_history", []),
        "correction_log": correction_log,
    }

def moderator_node(state: AgentState):
    # Filtramos para obtener solo las respuestas de los autores (ignoramos al usuario y al propio moderador previo)
    mensajes_autores = [m[1] for m in state["messages"] if any(prefix in m[1] for prefix in ["ELIADE:", "CIORAN:"])]
    
    score_coseno = 0.0
    if len(mensajes_autores) >= 2:
        # Comparamos la Ãºltima respuesta de Eliade con la Ãºltima de Cioran
        score_coseno = calcular_similitud_vectores(mensajes_autores[-1], mensajes_autores[-2])
    
    res = call_agent_with_rag(state, MODERATOR_SYSTEM_PROMPT, "MODERADOR")
    
    # Inyectamos métricas al final del texto del moderador
    texto_mod = res["messages"][-1][1]
    eval_history = res.get("eval_history", [])
    last_e = next((x for x in reversed(eval_history) if x.get("agent") == "ELIADE"), None)
    last_c = next((x for x in reversed(eval_history) if x.get("agent") == "CIORAN"), None)
    last_m = next((x for x in reversed(eval_history) if x.get("agent") == "MODERADOR"), None)

    fidelidad_autor = 0.0
    if last_e and last_c:
        fidelidad_autor = ((last_e.get("faithfulness", 0.0) + last_c.get("faithfulness", 0.0)) / 2.0) * 100.0

    ragas_line = ""
    if last_m:
        ragas_line = (
            f"[RAGAS] Faithfulness: {last_m.get('faithfulness', 0.0):.4f} | "
            f"Answer Relevance: {last_m.get('answer_relevance', 0.0):.4f} | "
            f"Context Precision: {last_m.get('context_precision', 0.0):.4f}\n"
        )

    res["messages"][-1] = (
        "assistant",
        f"{texto_mod}\n\n"
        f"[METRICA MATEMATICA] Similitud Coseno: {score_coseno}\n"
        f"{ragas_line}"
        f"[FIDELIDAD AUTOR] {fidelidad_autor:.2f}/100"
    )
    res["memory_summary"] = update_memory_summary(res)

    return res

# ==========================================
# 4. CONSTRUCCION DEL GRAFO
# ==========================================

workflow = StateGraph(AgentState)

workflow.add_node("Eliade", eliade_node)
workflow.add_node("Cioran", cioran_node)
workflow.add_node("CriticaFuentes", source_critic_node)
workflow.add_node("Moderador", moderator_node)

workflow.set_entry_point("Eliade")
workflow.add_edge("Eliade", "Cioran")
workflow.add_edge("Cioran", "CriticaFuentes")
workflow.add_edge("CriticaFuentes", "Moderador")

def should_continue(state: AgentState):
    # Terminamos tras 6 pasos (2 rondas completas de E-C-M)
    if state["turn"] >= 6:
        return END
    return "Eliade"

workflow.add_conditional_edges("Moderador", should_continue)

app = workflow.compile()

# ==========================================
# 5. EJECUCION (Modo Consola)
# ==========================================

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("\nSYNODOS AI: SISTEMA MULTI-AGENTE ACTIVADO")
    dilema = input("Introduce el tema para el debate: ")
    
    initial_input = {
        "messages": [("user", f"Dilema inicial: {dilema}")],
        "turn": 0,
        "topic": dilema,
        "memory_summary": "",
        "eval_history": [],
        "correction_log": []
    }

    for output in app.stream(initial_input):
        for node_name, state_update in output.items():
            if node_name == "CriticaFuentes":
                continue
            print(f"\nTURNO DE: {node_name.upper()}\n{'-'*30}")
            texto_turno = state_update["messages"][-1][1]
            print(texto_turno.encode("cp1252", errors="replace").decode("cp1252"))



