import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings, OpenAIEmbeddings
from constants import CHROMA_SETTINGS

load_dotenv()

def load_documents_from_memory():
    loaders = []  # List to store loaders
    documents = []  # List to store loaded documents

    for root, dirs, files in os.walk("source_documents"):
        for file in files:
            if file.endswith(".txt"):
                loader = TextLoader(os.path.join(root, file), encoding="utf8")
            elif file.endswith(".pdf"):
                loader = PDFMinerLoader(os.path.join(root, file))
            elif file.endswith(".csv"):
                loader = CSVLoader(os.path.join(root, file))

            loaders.append(loader)  # Store the loader for each file

    # Load documents using each loader
    for loader in loaders:
        loaded_docs = loader.load()
        documents.extend(loaded_docs)  # Append the loaded documents to the list

    return documents

def main():

    llama_embeddings_model = os.environ.get('LLAMA_EMBEDDINGS_MODEL')
    persist_directory = os.environ.get('PERSIST_DIRECTORY')
    model_n_ctx = os.environ.get('MODEL_N_CTX')
    model_type = os.environ.get('MODEL_TYPE')
    documents = load_documents_from_memory()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(texts)
    # Create embeddings
    # llama = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)
    match model_type:
        case "OpenAI":
            embedding_func = OpenAIEmbeddings()
        case _default:
            embedding_func = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)
    # Create and store locally vectorstore
    db = Chroma.from_documents(texts, embedding_func, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

if __name__ == "__main__":
    main()
