Synodos AI: El Vacío Dialéctico
Synodos AI es un proyecto de Inteligencia Artificial desarrollado para el curso de Aprendizaje Automático de la Universidad CENFOTEC
. Consiste en un sistema de debate automatizado que enfrenta dos visiones del mundo radicalmente opuestas: "Lo Sagrado" (orden y trascendencia) de Mircea Eliade frente a "La Nada" (nihilismo y vacío) de Emil Cioran
.
🧠 El Problema Filosófico
El sistema utiliza técnicas avanzadas de procesamiento de lenguaje natural para mediar en el conflicto dialéctico entre la búsqueda del mito eterno y la aceptación del cinismo existencial
.
🛠️ Arquitectura Técnica
El proyecto evolucionó de una infraestructura local hacia una Arquitectura Híbrida para maximizar el razonamiento lógico
:
Inferencia: Uso de Azure AI Foundry (GPT-4o) para el procesamiento de alto nivel
.
Orquestación: Implementación de LangGraph (StateGraph) para la gestión de turnos, memoria de debate y estados entre agentes
.
Estrategia de Datos:
Corpus de 6 libros fundamentales (3 de Eliade y 3 de Cioran) procesados en 4,661 fragmentos bibliográficos
.
Embeddings: Modelo all-miniLM-l6-v2 de HuggingFace
.
Base de Datos Vectorial: ChromaDB local para garantizar la soberanía de los datos
.
Optimizaciones: Uso de Reranking (Cross-Encoders) y memoria resumida para mitigar alucinaciones por exceso de contexto
,
.
📚 Bibliografía del Corpus
El conocimiento del sistema se extrae directamente de las siguientes obras:
Mircea Eliade: El mito del eterno retorno, Lo sagrado y lo profano y Tratado de historia de las religiones.
Emil Cioran: Breviario de podredumbre, Del inconveniente de haber nacido y La tentación de existir.
📊 Evaluación y Resultados (Métricas RAGAS)
Para validar la precisión científica del sistema, se integró el framework RAGAS, midiendo tres indicadores críticos
,
:
Faithfulness: Valida que las respuestas de los agentes provengan estrictamente de los libros
.
Answer Relevance: Mide la pertinencia de la respuesta al dilema planteado
.
Context Precision: Evalúa la utilidad de los fragmentos recuperados de ChromaDB
.
Hallazgo clave: Se implementó un Agente de Crítica de Fuentes, un nodo de auto-corrección que verifica la pureza conceptual de cada autor antes de la intervención del moderador
,
.
