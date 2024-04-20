import argparse
import pickle
from typing import Union

from langchain import hub
from langchain.document_loaders.web_base import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

vectorstore_path = "vectorstore.pkl"


def ingest_document(url: Union[str, None]) -> None:
    """
    Ingests a document from a given URL and stores it in a vector store.

    Args:
        url (Union[str, None]): The URL of the document to ingest.

    Returns:
        None
    """

    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=(url,),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

    vectorstore = FAISS.from_documents(
        documents=splits, embedding=HuggingFaceEmbeddings()
    )

    # Save the vectorstore to a file for qna
    with open(vectorstore_path, "wb") as handle:
        pickle.dump(vectorstore, handle, protocol=pickle.HIGHEST_PROTOCOL)


def qna() -> None:
    """
    Loads the pre-processed document, performs question-answering loop
    on ingested document.
    """
    vectorstore = None
    try:
        print("Loading documents...")
        with open(vectorstore_path, "rb") as handle:
            vectorstore = pickle.load(handle)
    except FileNotFoundError:
        print(
            "No document ingested, use python3 qna.py --ingest <url> to ingest document"
        )
        exit(0)

    if vectorstore is None:
        print("No document ingested, use python3 qna.py --ingest <url>")
        exit(0)

    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = ChatNVIDIA(model="ai-llama3-70b", max_tokens=128)

    # Create a retriever chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Loop to take user query and answer from RAG chain
    while True:
        query = input("\nquery> ")
        resp = rag_chain.stream(query)

        print("resp>> ", end="")
        for r in resp:
            print(r, end="")

        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="qna.py",
        description="Talk to your document",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ingest", metavar="URL", help="Ingest a URL")
    group.add_argument("--qna", action="store_true", help="Interact with documents")

    args = parser.parse_args()

    if args.ingest:
        print(f"Ingesting URL: {args.ingest}")
        ingest_document(url=args.ingest)
    elif args.qna:
        print("Interacting with documents")
        qna()
