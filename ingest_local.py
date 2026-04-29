import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def crear_base_datos_local():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    pdf_folder = "./data"
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
        print("Carpeta /data creada.")
        return

    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)

    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            print(f"Cargando: {file}")
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents = loader.load()
            
            # --- NUEVA LÓGICA DE METADATOS ---
            # Identificamos al autor según el nombre del archivo
            if "Cioran" in file:
                autor_tag = "cioran"
            elif "Eliade" in file:
                autor_tag = "eliade"
            else:
                autor_tag = "desconocido"

            # Dividimos y asignamos el metadato a cada fragmento
            file_chunks = text_splitter.split_documents(documents)
            for chunk in file_chunks:
                chunk.metadata["autor"] = autor_tag
            
            all_chunks.extend(file_chunks)

    if not all_chunks:
        print("No hay documentos para procesar.")
        return

    print(f"Procesando {len(all_chunks)} fragmentos con metadatos...")
    
    # IMPORTANTE: Esto sobreescribirá la carpeta anterior para limpiar datos viejos
    vector_db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory="./db_synodos_local"
    )
    print("¡Base de datos vectorial con METADATOS creada con éxito!")

if __name__ == "__main__":
    crear_base_datos_local()