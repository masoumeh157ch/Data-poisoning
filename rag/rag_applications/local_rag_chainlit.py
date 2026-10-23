import os
# import requests
import numpy as np
from langchain.schema import Document 
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
import ocrmypdf


class EnRag():
    def __init__(self) -> None:
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        self.PROMPT_TEMPLATE =  \
        """
        Answer the question based only on the following context and do not use any external information or knowledge:

        {context}

        ---

        Answer the question based on the above context: {question}
        """
        self.llm = ChatOllama(
        model="llama3.1",
        temperature=0,
        )
        
    def parse_file(self, filepath:str):
        print(f"Processing file: {filepath}")
        # if filepath.endswith(".pdf"):
        #     ocrmypdf.ocr(filepath,
        #         f"{filepath.rstrip('.pdf')}_temp.pdf", 
        #         output_type= "pdf",
        #         deskew=False, 
        #         force_ocr=True ,
        #         sidecar=f"{filepath.rstrip('.pdf')}.txt" ,
        #         progress_bar=True)
        # with open(f"{filepath.rstrip('.pdf')}.txt", 'r') as file:
        #     text = file.read()
        
        # remove /home/shadmehr/llmXP/EngRAG/ from filepath 
        file_path_relative = "/".join(filepath.split("/")[6:])
        loader = DirectoryLoader('.', glob="eltex.txt")
        documents = loader.load()
        
        # print(documents)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        
        def score_normalizer(val: float) -> float:
            return 1 - 1 / (1 + np.exp(val)) 
        self.chunks = text_splitter.split_documents(documents)
        self.vector_store = FAISS.from_documents(self.chunks, self.embedding_model ,  relevance_score_fn=score_normalizer )
        print(f"Processing file: {filepath} done.")
    def query(self, query_text:str):
        def score_normalizer(val: float) -> float:
            return 1 - 1 / (1 + np.exp(val))
        results= self.vector_store.similarity_search_with_relevance_scores(query_text , k=5 , relevance_score_fn=score_normalizer)
        
        if len(results) == 0 or results[0][1] < 0.6:
            print(f"Unable to find matching results.")

        prompt_template = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt = prompt_template.format(context=context_text, question=query_text)

        
        messages = \
        [
        ("system", "You are a helpful assistant. answer questions based on the given context. don't use external information. and don't make assupmtions."),
        ("human", prompt),
        ]
        result = self.llm.invoke(messages)
        return result.content



import chainlit as cl
import requests
from chainlit.input_widget import Select

rag_app = EnRag()
rag_app.parse_file("eltex.txt")
@cl.on_chat_start
async def on_chat_start():

    uid = cl.user_session.get("id")
    await cl.Message(
        content=f"Hello and welcome {uid}" ,
        disable_feedback=True
    ).send()

    # files = None
    # # Wait for the user to upload a file
    # while files == None:
    #     files = await cl.AskFileMessage(
    #         content=f"Please upload a text or pdf file to begin! ",
    #         accept=["text/plain", "application/pdf"],
    #         max_size_mb=20,
    #         timeout=280,
    #     ).send()

    # file = files[0]
    # msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True )
    # await msg.send()
    # def req():
    #     return rag_app.parse_file(file.path)
    # response = await cl.make_async(req)()
    # msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    # await msg.update()
    
    
@cl.on_message
async def main(message: cl.Message):
    def answer():
        return rag_app.query(message.content)
    response = await cl.make_async(answer)()
    await cl.Message(content=response).send()