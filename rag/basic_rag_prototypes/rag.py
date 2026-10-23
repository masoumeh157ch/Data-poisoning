from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from llama_api import LLaMAAPI

class RAG:
    def __init__(self, api_key, api_url, knowledge_base):
        self.api_key = api_key
        self.api_url = api_url
        self.knowledge_base = knowledge_base
        self.llama_api = LLaMAAPI(api_key, api_url)

    def retrieve(self, query):
        # Simple retrieval method: find the first matching line in the knowledge base
        for line in self.knowledge_base.splitlines():
            if query.lower() in line.lower():
                return line
        return None

    def generate(self, query, max_length=2048):
        retrieved_text = self.retrieve(query)
        if retrieved_text:
            prompt = f"Based on the following text: {retrieved_text}\n{query}"
        else:
            prompt = query
        return self.llama_api.generate(prompt, max_length=max_length)