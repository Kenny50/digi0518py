from langchain_community.vectorstores import Chroma
from .cohereEmbedding import CohereEmbeddings

useDb = Chroma(
    collection_name="rag-cohere",
    persist_directory="./chroma_db/cohere/sentencefull",
    embedding_function=CohereEmbeddings()
)


def useChromaDb(query, k=2):
    result = useDb.similarity_search_with_score(query=query, k=k)
    return result

# useDb.as_retriever()

# useChromaDb()