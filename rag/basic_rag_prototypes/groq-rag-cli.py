import os
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate

# 1. Initialize Groq Client
api_key = "key"
client = Groq(api_key=api_key)

# 2. Load your knowledge base with specified encoding
loader = TextLoader("test.txt", encoding="utf-8")
docs = loader.load()

# 3. Create Embeddings and Vector Store for Retrieval
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# 4. Define the RAG Chatbot
retriever = vectorstore.as_retriever()

# Prompt engineering
prompt_template = """
You are an intelligent assistant. Use the provided context to answer the query.

Context: {context}
Query: {query}

Response:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])

# 5. Define the Language Model (LLaMA3)
def llama3_completion(prompt):
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content

# 6. Define the Chat Function
def chat(query):
    # Step 1: Retrieve relevant context
    docs = retriever.get_relevant_documents(query)[:3]
    context = " ".join([doc.page_content[:1000] for doc in docs])

    # Step 2: Generate a response with the language model
    final_prompt = prompt.format(context=context, query=query)
    response = llama3_completion(final_prompt)

    return response

# Example Interaction
if __name__ == "__main__":
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        bot_response = chat(user_input)
        print(f"Bot: {bot_response}")
