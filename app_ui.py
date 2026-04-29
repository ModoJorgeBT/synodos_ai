import io
import re

import chainlit as cl
import matplotlib.pyplot as plt

from main import app  # Grafo de LangGraph compilado


@cl.on_chat_start
async def start():
    # Historial de métricas por ronda
    cl.user_session.set("history_conv", [])
    cl.user_session.set("history_cos", [])
    cl.user_session.set("history_fid", [])

    await cl.Message(
        content="**Synodos AI: Sistema Multi-Agente Activado**\n"
                "Debate Filosofico: *Mircea Eliade vs Emil Cioran*\n"
                "---"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    graph = app
    h_conv = cl.user_session.get("history_conv")
    h_cos = cl.user_session.get("history_cos")
    h_fid = cl.user_session.get("history_fid")

    initial_input = {
        "messages": [("user", f"Dilema inicial: {message.content}")],
        "turn": 0,
        "topic": message.content,
        "memory_summary": "",
        "eval_history": [],
        "correction_log": [],
    }

    async for output in graph.astream(initial_input):
        for node_name, state_update in output.items():
            content = state_update["messages"][-1][1]

            if node_name.lower() == "moderador":
                match_conv = re.search(r"(\d+)/100", content)
                if match_conv:
                    h_conv.append(int(match_conv.group(1)))

                match_cos = re.search(r"Similitud Coseno: ([\d\.]+)", content)
                if match_cos:
                    h_cos.append(float(match_cos.group(1)) * 100)

                match_fid = re.search(r"\[FIDELIDAD AUTOR\]\s*([\d\.]+)/100", content)
                if match_fid:
                    h_fid.append(float(match_fid.group(1)))

            if node_name.lower() != "criticafuentes":
                await cl.Message(content=f"**{node_name.upper()}**:\n{content}").send()

    if h_conv:
        plt.figure(figsize=(10, 5))

        plt.plot(
            range(1, len(h_conv) + 1),
            h_conv,
            marker="o",
            linestyle="-",
            color="#1f77b4",
            label="Convergencia (Evaluacion IA)",
        )

        if h_cos:
            plt.plot(
                range(1, len(h_cos) + 1),
                h_cos,
                marker="s",
                linestyle="--",
                color="#d62728",
                label="Similitud Coseno (Vectorial)",
            )

        if h_fid:
            plt.plot(
                range(1, len(h_fid) + 1),
                h_fid,
                marker="^",
                linestyle="-.",
                color="#2ca02c",
                label="Fidelidad al Autor (RAGAS)",
            )

        plt.title("Analisis de Proximidad Dialectica: Eliade vs Cioran")
        plt.xlabel("Rondas de Debate")
        plt.ylabel("Nivel de Acuerdo / Similitud (0-100)")
        plt.ylim(0, 105)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)

        image = cl.Image(content=buf.read(), name="metricas_debate", display="inline")
        await cl.Message(
            content="### Reporte Analitico Final\n"
                    "Comparativa entre convergencia, similitud vectorial y fidelidad al autor.",
            elements=[image],
        ).send()


# Ejecutar: chainlit run app_ui.py -w
