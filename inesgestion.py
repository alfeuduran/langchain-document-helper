import logging
import os
import re
from bs4 import BeautifulSoup, SoupStrainer, Tag
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Weaviate
from langchain.vectorstores.azuresearch import AzureSearch

logger = logging.getLogger(__name__)

vector_store_address: str = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
vector_store_password: str = os.getenv("AZURE_SEARCH_ADMIN_KEY")
index_name : str = os.getenv("INDEX_NAME")


# !pip install nest-asyncio
import nest_asyncio
nest_asyncio.apply()


def _get_text(tag):
    new_line_elements = {"h1", "h2", "h3", "h4", "code", "p", "li"}
    code_elements = {"code"}
    skip_elements = {"button"}
    for child in tag.children:
        if isinstance(child, Tag):
            # if the tag is a block type tag then yield new lines before after
            is_code_element = child.name in code_elements
            is_block_element = is_code_element and "codeBlockLines_e6Vv" in child.get(
                "class", ""
            )
            if is_block_element:
                yield "\n```python\n"
            elif is_code_element:
                yield "`"
            elif child.name in new_line_elements:
                yield "\n"
            if child.name == "br":
                yield from ["\n"]
            elif child.name not in skip_elements:
                yield from _get_text(child)

            if is_block_element:
                yield "```\n"
            elif is_code_element:
                yield "`"
        else:
            yield child.text

def _doc_extractor(html):
    soup = BeautifulSoup(html, "lxml", parse_only=SoupStrainer("article"))
    for tag in soup.find_all(["nav", "footer", "aside"]):
        tag.decompose()
    joined = "".join(_get_text(soup))
    return re.sub(r"\n\n+", "\n\n", joined)

def _simple_extractor(html):
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text)

#  Teste de ingestão de documentos usando o Azure Search

simple_urls = ["https://api.python.langchain.com/en/latest/"]
doc_urls = [
        "https://python.langchain.com/docs/get_started/introduction",
        "https://python.langchain.com/docs/use_cases",
        "https://python.langchain.com/docs/integrations",
        "https://python.langchain.com/docs/modules",
        "https://python.langchain.com/docs/guides",
        "https://python.langchain.com/docs/ecosystem",
        "https://python.langchain.com/docs/additional_resources",
        "https://python.langchain.com/docs/community",
        "https://python.langchain.com/docs/expression_language",
    ]

urls = [(url, _simple_extractor) for url in simple_urls] + [
        (url, _doc_extractor) for url in doc_urls
    ]

documents = []

for url, extractor in urls:
        loader = RecursiveUrlLoader(
            url=url,
            max_depth=8,
            extractor=extractor,
            prevent_outside=True,
            use_async=True
        )
        temp_docs = loader.load()
        documents += temp_docs
        


text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
docs_transformed = text_splitter.split_documents(documents)

for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""

# embedding = OpenAIEmbeddings(chunk_size=200)  # rate limit
embeddings: OpenAIEmbeddings = OpenAIEmbeddings(openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),deployment="embedding", chunk_size=200, openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"), openai_api_type="azure" )


index_name = "langchain-index-helper-final"
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query, 
)
# quais opções além dessas de embed query eu tenho que eu posso usar e qual a diferença entre elas?
vector_store.add_documents(documents=docs_transformed)

# docs = vector_store.similarity_search(
#     query="What is an OpenAI Adapter",
#     k=3,
#     search_type="similarity",
# )
# print(docs[0].page_content)

# # Perform a hybrid search
# docs = vector_store.similarity_search(
#     query="how to Debugging an LLM",
#     k=3, 
#     search_type="hybrid"
# )
# print(docs[0].page_content)

# docs = vector_store.similarity_search(
#     query="langchain.agents:",
#     k=3, 
#     search_type="hybrid"
# )
# print(docs[0].page_content)

# # Perform a hybrid search
docs = vector_store.hybrid_search(
    query="What is module in Langchain?", 
    k=3
)
print(docs[0].page_content)


if __main__ == "__name__":
    _get_text()

