from rag import RAG

def main():
    api_key = "key"
    api_url = "https://api.groq.com/v1/llama-3.0-70b/generate"
    knowledge_base = open("yourfile.txt", "r", encoding="utf-8", errors="ignore").read()   
    rag = RAG(api_key, api_url, knowledge_base)

    while True:
        query = input("User: ")
        response = rag.generate(query)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()