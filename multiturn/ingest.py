# Import necessary libraries and modules
from langchain_community.document_loaders import (
    WebBaseLoader,
)  # To load documents from the web
from langchain_community.vectorstores import Chroma  # For managing vector databases
from langchain_core.runnables import (  # For defining and chaining tasks
    RunnableAssign,
    RunnablePassthrough,
)
from langchain_nvidia_ai_endpoints import (  # AI models and embeddings
    ChatNVIDIA,
    NVIDIAEmbeddings,
    NVIDIARerank,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)  # For splitting text into chunks

# Initialize a global variable to store the vector database
vectorstore = None


def get_vectorstore():
    """
    Retrieve the current vectorstore as a retriever.

    Returns:
        A retriever instance for the current vectorstore.
    """
    global vectorstore
    return vectorstore.as_retriever()


def ingest_document(url: str):
    """
    Ingest a document from a given URL and store its contents in a vectorstore.

    Args:
        url (str): The URL of the document to be ingested.
    """
    global vectorstore
    urls = [url]  # List containing the single URL

    print("URL: ", url)

    # Load documents from the provided URL
    docs = [WebBaseLoader(url).load() for url in urls]
    # Flatten the list of lists into a single list
    docs_list = [item for sublist in docs for item in sublist]

    # Initialize a text splitter to split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, chunk_overlap=20
    )

    # Split the documents into chunks
    doc_splits = text_splitter.split_documents(docs_list)

    # If a vectorstore already exists, add the new documents to it
    if vectorstore:
        vectorstore.add_documents(doc_splits)
    else:
        # Create a new vectorstore and add documents to it
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",  # Name of the collection in the vectorstore
            embedding=NVIDIAEmbeddings(
                truncate="END"
            ),  # Use NVIDIA embeddings for document representation
        )


def olympics_data_retriever(query: str) -> str:
    """
    Retrieve data related to the 2024 Paris Olympics medal information based on the query.

    Args:
        query (str): The user's query for which data is to be retrieved.

    Returns:
        str: A string containing the relevant documents' content.
    """
    global vectorstore
    docs = []

    # Initialize a ranker to sort and select relevant documents
    ranker = NVIDIARerank(top_n=7, truncate="END")

    # Create a retriever for searching the vectorstore
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 20}  # Number of documents to retrieve
    )

    # Define a task to assign context based on the retriever's output
    context_reranker = RunnableAssign(
        {
            "context": lambda input: ranker.compress_documents(
                query=input["question"], documents=input["context"]
            )
        }
    )

    # Combine the retriever and context ranker tasks
    retriever = {
        "context": retriever,
        "question": RunnablePassthrough(),
    } | context_reranker

    # Invoke the retriever to get relevant documents
    docs = retriever.invoke(query)
    resp = []
    # Extract the content from the retrieved documents
    for doc in docs.get("context"):
        resp.append(doc.page_content)

    print("Relevant Document: ", resp)
    return "\n".join(resp)


# Sample URLs for testing the ingestion functionality
urls = [
    "https://en.wikipedia.org/wiki/2024_Summer_Olympics_medal_table",
    "https://en.wikipedia.org/wiki/India_at_the_2024_Summer_Olympics",
    "https://en.wikipedia.org/wiki/United_States_at_the_2024_Summer_Olympics",
]
