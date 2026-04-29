# Synodos AI: El Vacío Dialéctico 🧠🌑

**Resolución de conflictos filosóficos mediante arquitecturas RAG Multi-Agente.**

Este proyecto implementa un sistema de inteligencia artificial híbrido diseñado para computar la contradicción dialéctica entre **Mircea Eliade** (Lo Sagrado, El Orden) y **Emil Cioran** (La Nada, El Nihilismo). Utiliza un flujo de trabajo de agentes coordinados para explorar si la creación humana es un acto de trascendencia o un gesto de desesperación.

---

## 🚀 Arquitectura del Sistema

El sistema utiliza una **Arquitectura Híbrida** para maximizar la soberanía de los datos y la potencia de razonamiento:

- **Orquestación:** [LangGraph](https://www.langchain.com/langgraph) para la gestión de estados y turnos entre agentes.
- **LLM (Razonamiento):** Inferencia en la nube mediante **Azure AI Foundry** (GPT-4o / Phi-3.5).
- **RAG (Datos):** Base de datos vectorial local **ChromaDB** con embeddings de HuggingFace.
- **Optimización de Recuperación:** Implementación de **Cross-Encoders (Reranking)** para asegurar relevancia semántica superior.

## 📚 Estrategia de Datos (Corpus)

Se ha procesado un corpus bibliográfico masivo para garantizar que los agentes no alucinen y mantengan la fidelidad a sus autores:
- **6 Libros Fundamentales** indexados.
- **4,661 Fragmentos (chunks)** bibliográficos.
- **Metadatos de Autoría:** Filtros rígidos que previenen la contaminación cruzada en el espacio latente.

## 📊 Métricas de Evaluación Científica

A diferencia de los chatbots convencionales, Synodos AI mide el debate mediante métricas híbridas:

1. **Similitud Coseno (Cuantitativa):** Mide la proximidad vectorial de las respuestas.
2. **Índice de Convergencia (Cualitativa):** Evaluación realizada por un agente Moderador sobre el acuerdo lógico.
3. **Framework RAGAS:**
   - **Faithfulness (Fidelidad):** Validación de que las respuestas provienen del corpus.
   - **Answer Relevance:** Pertinencia de la respuesta al dilema planteado.

> **Hallazgo Clave:** El proyecto documenta la *Paradoja Vectorial*, donde una alta similitud matemática no implica necesariamente un acuerdo filosófico, demostrando cómo los autores utilizan un lenguaje similar para defender tesis opuestas.

## 🛠️ Configuración Local

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/ModoJorgeBT/synodos_ai.git](https://github.com/ModoJorgeBT/synodos_ai.git)
   cd synodos_ai
