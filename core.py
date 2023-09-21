from langchain.vectorstores.azuresearch import AzureSearch
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import os
import logging



logger = logging.getLogger(__name__)

model: str = "text-embedding-ada-002"
vector_store_address: str = "https://gptkb-rahzjehw7exti.search.windows.net"
vector_store_password: str = "KB7fjV9u8X4WWuF94enaPwmaCFCDgqgZeKZoEsz0WKAzSeCEt1Nl"
index_name = "langchain-index-helper-final"


embeddings: OpenAIEmbeddings = OpenAIEmbeddings(openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),deployment="embedding", chunk_size=200, openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"), openai_api_type="azure",model="text-embedding-ada-002" )


# podemos criar um método aqui chamado run_llm onde ele recebe a query e retorna o resultado pelo chat

index_name = "langchain-index-helper-final"
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query, 
)

chat = AzureChatOpenAI(verbose=True,temperature=0,openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"))
qa = RetrievalQA.from_chain_type(llm=chat,chain_type="stuff",retriever=vector_store.as_retriever(), return_source_documents=True)


print(qa("What is module in Langchain?"))



# def runllm(query: str) -> any:
#     #  embeddings new
#     # add vectorstore connection again? 
