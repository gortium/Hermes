import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Load environment variables from .env file
load_dotenv()
ollama_url = os.getenv('OLLAMA_URL')
llm_model = os.getenv('MODEL')

llm = ChatOllama(model=llm_model, base_url=ollama_url, temperature=0)

if __name__ == "__main__":
    response = llm.invoke([
        ("system", "You are a helpful translator. Translate the user sentence to French."),
        ("human", "I love programming."),
        ])
    
    print(response.content)