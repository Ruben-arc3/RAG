# RAG
Practica de Retrieval Augmented Generation (RAG) 

# Proyecto RAG con LangChain, Ollama y Streamlit

Este proyecto implementa un sistema RAG (Retrieval-Augmented Generation) que permite hacer preguntas a documentos PDF usando modelos locales con Ollama, LangChain y una base de datos vectorial Chroma, a través de una interfaz web creada con Streamlit.

---

Características:
- Carga de archivos PDF
- Procesamiento y división de texto en chunks
- Generación de embeddings con Ollama
- Almacenamiento en ChromaDB
- Recuperación de información con RAG
- Interfaz web con Streamlit
- Respuestas generadas con llama3

---

Tecnologías utilizadas:
- Python 3.10+
- LangChain
- Ollama
- ChromaDB
- Streamlit
- PyPDF

---

Estructura del proyecto:

RAG/
  chroma_db/
  pdfs/
  venv/
  ingest.py
  chat.py
  pregunta.rtf
  requirements.txt
  README.md

---

Instalación:

1. Clonar el repositorio
git clone https://github.com/Ruben-arc3/RAG.git
cd RAG

2. Crear entorno virtual
python -m venv venv

3. Activar entorno virtual

Windows:
venv\Scripts\activate

Linux / Mac:
source venv/bin/activate

4. Instalar dependencias
pip install -r requirements.txt

---

Instalación de Ollama y modelos:

Descargar Ollama desde:
https://ollama.com

Luego ejecutar:
ollama pull llama3
ollama pull embeddinggemma:300m

Verificar modelos instalados:
ollama list

---

Crear la base vectorial:

Colocar los archivos PDF dentro de la carpeta "pdfs" y ejecutar:
python ingest.py

Se creará la base de datos vectorial en la carpeta "chroma_db".

---

Ejecutar la aplicación:

streamlit run chat.py

Luego abrir en el navegador:
http://localhost:8501



Autor:
Rubén Mora
Estudiante de Ingeniería de Sistemas

Proyecto desarrollado con fines académicos.

