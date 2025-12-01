import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate


CHROMA_DIR = "chroma_db"


@st.cache_resource
def load_qa_chain():
    # Cargar embeddings y base vectorial
    embeddings = OllamaEmbeddings(model="embeddinggemma:300m")

    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # nº de chunks a recuperar
    )

    llm = ChatOllama(
        model="llama3",   # modelo que tengas en Ollama
        temperature=0.1
    )

    template = """
Eres un asistente que responde usando EXCLUSIVAMENTE el contexto proporcionado.
Si la respuesta no está en el contexto, di claramente que no aparece en los documentos.

---------------- CONTEXTO ----------------
{context}
-----------------------------------------

Pregunta del usuario: {question}

Respuesta en español, clara y concisa:
"""
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return qa_chain


def main():
    st.set_page_config(page_title="RAG PDFs + Ollama", page_icon="📚", layout="wide")

    st.title("📚 RAG con PDFs + Ollama")
    st.caption("Pregunta sobre el contenido de tus PDFs indexados en Chroma.")

    # Cargar cadena de QA
    with st.spinner("Cargando modelo y base vectorial..."):
        qa_chain = load_qa_chain()

    # Sidebar
    st.sidebar.header("Configuración")
    k = st.sidebar.slider("Número de fragmentos a recuperar (k)", 1, 10, 4)
    temperatura = st.sidebar.slider("Temperatura del modelo", 0.0, 1.0, 0.1, 0.1)

    # Opcional: actualizar retriever / llm con parámetros de sidebar
    # (simple: solo mostramos k en pantalla y mantenemos fijos en load_qa_chain)
    st.sidebar.markdown(
        f"""
        **Info actual:**
        - k = {k} (solo mostrado, el retriever está configurado en 4 en el código)
        - temperatura = {temperatura} (config fija en 0.1 en el código)
        """
    )
    st.sidebar.info(
        "Si quieres que k y temperatura cambien realmente, se puede ajustar para recrear la cadena con estos valores."
    )

    # Simular chat simple
    if "history" not in st.session_state:
        st.session_state.history = []

    st.subheader("❓ Haz tu pregunta")

    pregunta = st.text_area(
        "Escribe tu pregunta sobre los PDFs:",
        placeholder="Ejemplo: ¿Cuál es la metodología utilizada en el documento X?",
        height=100,
    )

    if st.button("Enviar pregunta"):
        if not pregunta.strip():
            st.warning("Por favor escribe una pregunta.")
        else:
            with st.spinner("Generando respuesta con RAG..."):
                result = qa_chain({"query": pregunta})
                answer = result["result"]
                sources = result.get("source_documents", [])

            # Guardar en historial
            st.session_state.history.append(
                {
                    "question": pregunta,
                    "answer": answer,
                    "sources": [
                        {
                            "file": doc.metadata.get("source_file", "desconocido"),
                            "page": doc.metadata.get("page", "¿?"),
                        }
                        for doc in sources
                    ],
                }
            )

    # Mostrar historial
    if st.session_state.history:
        st.subheader("📜 Historial de preguntas")
        for i, item in enumerate(reversed(st.session_state.history), start=1):
            st.markdown(f"### Pregunta {len(st.session_state.history) - i + 1}")
            st.markdown(f"**❓ Pregunta:** {item['question']}")
            st.markdown(f"**💬 Respuesta:** {item['answer']}")

            with st.expander("📚 Ver fuentes utilizadas"):
                if not item["sources"]:
                    st.write("No se devolvieron fuentes.")
                else:
                    for j, src in enumerate(item["sources"], start=1):
                        st.write(f"**[{j}]** {src['file']} (página {src['page']})")

            st.markdown("---")
    else:
        st.info("Aún no has hecho ninguna pregunta. Escribe tu primera pregunta arriba.")


if __name__ == "__main__":
    main()