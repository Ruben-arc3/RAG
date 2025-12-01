import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

PDF_DIR = "pdfs"
CHROMA_DIR = "chroma_db"

def load_pdfs(pdf_dir: str):
    docs = []
    for fname in os.listdir(pdf_dir):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(pdf_dir, fname)
            print(f"[INGESTA] Leyendo: {path}")
            loader = PyPDFLoader(path)
            file_docs = loader.load()
            # Etiquetar el nombre del archivo como metadata
            for d in file_docs:
                d.metadata["source_file"] = fname
            docs.extend(file_docs)
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    return splitter.split_documents(docs)

def main():
    if not os.path.exists(PDF_DIR):
        raise FileNotFoundError(f"No existe la carpeta '{PDF_DIR}'. Créala y pon tus PDFs allí.")

    docs = load_pdfs(PDF_DIR)
    print(f"[INGESTA] Documentos cargados: {len(docs)}")

    chunks = split_docs(docs)
    print(f"[INGESTA] Chunks generados: {len(chunks)}")

    # Embeddings con Ollama
    embeddings = OllamaEmbeddings(
        model="embeddinggemma:300m"   # puedes cambiar por el modelo que tengas en Ollama
    )

    # Crear / sobreescribir la base Chroma
    print("[INGESTA] Creando base vectorial Chroma...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    vectordb.persist()
    print(f"[INGESTA] Listo. Base vectorial guardada en '{CHROMA_DIR}'.")

if __name__ == "__main__":
    main()