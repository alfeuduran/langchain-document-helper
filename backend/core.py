from langchain.vectorstores.azuresearch import AzureSearch
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import os
import logging
from typing import List, Dict, Any


logger = logging.getLogger(__name__)

vector_store_address: str = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
vector_store_password: str = os.getenv("AZURE_SEARCH_ADMIN_KEY")
index_name: str = os.getenv("INDEX_NAME")


def run_llm(query: str, chat_history: List[Dict[str, Any]]) -> any:
    embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        deployment="embedding",
        chunk_size=200,
        openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_type="azure",
        model="text-embedding-ada-002",
    )
    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=vector_store_address,
        azure_search_key=vector_store_password,
        index_name=index_name,
        embedding_function=embeddings.embed_query,
        search_type="similarity"
    )

    

    chat = AzureChatOpenAI(
        verbose=True,
        temperature=0,
        openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        deployment_name="chat",
    )
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=vector_store.as_retriever(),return_source_documents=True 
    )

    
    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm("What are the use cases for langchain"))
