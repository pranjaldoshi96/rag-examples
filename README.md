# RAG Examples

This Python repository demonstrates a Retriever-Augmented Generation (RAG) pipeline for document-based question answering

## MultiTurn RAG
A simple RAG chain which allows you to ask followup questions based on document/url you have ingested.

### Setup the pipeline
- Get the `NVIDIA_API_KEY` from [NVIDIA API CATALOG](https://build.nvidia.com/)
- Export the environment variable
  ```
  export NVIDIA_API_KEY="nvapi-*"
  ```
- Move to directory
  ```
  cd multiturn/
  ```
- Install the requirements
  ```
  pip3 install -r requirements.txt
  ```
- Run the application
  ```
  streamlit run frontend.py
  ```
- Open the URL in browser, ingest the URL/document of your choice and start chatting with your data


https://github.com/user-attachments/assets/61535442-9e35-4489-b4f7-67f2879e261c



## qna-rag

A RAG  pipeline that ingest the document from a provided URL and answer question related to document using FAISS as document vectorstore and llama3 70b as llm model.

You can refer to [qna_rag](./qna_rag/README.md) to ingest the URL and interact with it.


https://github.com/pranjaldoshi96/rag-examples/assets/25930426/60016498-3715-4d94-844b-99711384ad47

