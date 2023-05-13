from dotenv import load_dotenv
from langchain.chains import RetrievalQA, ConversationChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import LlamaCppEmbeddings, OpenAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI, LlamaCpp, GPT4All
import os

load_dotenv()

llama_embeddings_model = os.environ.get("LLAMA_EMBEDDINGS_MODEL")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')

from constants import CHROMA_SETTINGS

def main():
    match model_type:
        case "OpenAI":
            embedding_func = OpenAIEmbeddings()
        case _default:
            embedding_func = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)

    db = Chroma(persist_directory=persist_directory, embedding_function=embedding_func, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    # Prepare the LLM
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = None
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        case 'OpenAI':
            llm = OpenAI()
        case _default:
            print(f"Model {model_type} not supported!")
            exit

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # qa = ConversationalRetrievalChain.from_llm(llm=OpenAI(temperature=0), retriever=retriever, memory=memory)

    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        
        # Get the answer from the chain
        res = qa(query)
        answer = res['result']
        # query = "What did the president say about Ketanji Brown Jackson"
        # res = qa({"question": query})
        # answer = res['answer']


        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)
        
        # Print the relevant sources used for the answer
        # for document in docs:
        #     print("\n> " + document.metadata["source"] + ":")
        #     print(document.page_content)

if __name__ == "__main__":
    main()
