import os
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain.prompts import PromptTemplate
from langdetect import detect  # For language detection
import streamlit as st  # Streamlit library for UI

# API keys
api_keys = [
    "Replace with your second API key",
    "Replace with your second API key",  # Replace with your second API key
]

# Groq client initialization with dynamic key management
class GroqClientManager:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.client = Groq(api_key=self.api_keys[self.current_key_index])

    def switch_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.client = Groq(api_key=self.api_keys[self.current_key_index])

    def get_client(self):
        return self.client

groq_manager = GroqClientManager(api_keys)

# 2. Load your knowledge base with JSON data
jq_schema = ".[] | {content: (.adv_texts | join(\" \"))}"
loader = JSONLoader(file_path="nq.json", jq_schema=jq_schema, content_key="content")
docs = loader.load()

# 3. Create Embeddings and Vector Store for Retrieval
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# 4. Define the RAG Chatbot
retriever = vectorstore.as_retriever()

# Prompt engineering
prompt_template_en = """
You are an intelligent assistant. Use the provided context to answer the query in English.

Context: {context}
Query: {query}

Response:
"""

prompt_template_fa = """
شما یک دستیار هوشمند هستید. از اطلاعات ارائه‌شده استفاده کنید و به پرسش به زبان فارسی پاسخ دهید.

متن: {context}
پرسش: {query}

پاسخ:
"""

# Define prompts for both languages
prompt_en = PromptTemplate(template=prompt_template_en, input_variables=["context", "query"])
prompt_fa = PromptTemplate(template=prompt_template_fa, input_variables=["context", "query"])

# 5. Define the Language Model (LLaMA3)
def llama3_completion(prompt):
    retries = len(api_keys)
    while retries > 0:
        try:
            client = groq_manager.get_client()
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            if "quota" in str(e).lower():
                groq_manager.switch_key()
                retries -= 1
            else:
                raise e
    raise RuntimeError("All API keys have exceeded their limits.")

# 6. Define the Chat Function
def chat(query):
    # Step 1: Detect language
    language = detect(query)  # Detect the query's language (e.g., 'en' or 'fa')

    # Step 2: Retrieve relevant context
    docs = retriever.get_relevant_documents(query)[:3]
    context = " ".join([doc.page_content[:1000] for doc in docs])

    # Step 3: Select the appropriate prompt template
    if language == "fa":
        final_prompt = prompt_fa.format(context=context, query=query)
    else:
        final_prompt = prompt_en.format(context=context, query=query)

    # Step 4: Generate a response with the language model
    response = llama3_completion(final_prompt)

    return response

# Streamlit App
st.title("Multilingual RAG Chatbot")
st.write("Ask your questions in English or Persian, and the chatbot will respond accordingly!")

# Input field for the user query
user_query = st.text_input("Enter your question:")

if user_query:
    # Get the chatbot's response
    with st.spinner("Generating response..."):
        bot_response = chat(user_query)

    # Display the response
    st.markdown(f"**Bot:** {bot_response}")
