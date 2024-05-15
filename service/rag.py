from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from .cohereEmbedding import CohereEmbeddings

def get_rag_chain(llm):
    """Prepare a RAG question answering chain.

      Note: Must use the same embedding model used for creating the semantic search index
      to be used for real-time semantic search.
    """
    vector_store = Chroma(
        collection_name="rag-cohere",
        persist_directory="./chroma_db/cohere/sentencefull",
        embedding_function=CohereEmbeddings()
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(k=5, fetch_k=50),
        return_source_documents=False,
        input_key="question",
    )