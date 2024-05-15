from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from cohereEmbedding import CohereEmbeddings

def metadata_func(record:dict, metadata: dict) -> dict:
    metadata['id'] = record.get('id')
    metadata['url'] = record.get('url')
    metadata['name'] = record.get('name')
    return metadata

loader = JSONLoader(
    file_path='/Users/chang/startup/ai/competition/digitime0518/data/khhattractions.json',
    jq_schema='.[]',
    content_key='.description',
    is_content_key_jq_parsable=True,
    # text_content=False,
    metadata_func=metadata_func)
data = loader.load()
print(data.__len__())

textSplitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    # chunk_overlap = 20,
    length_function = len,
    is_separator_regex=False,
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200B",
        "\uff0c",
        "\u3001",
        "\uff0e",
        "\u3002",
        ""
    ]
)

def loaderSplitter():
    docs = textSplitter.split_documents(data)
    print("loader splitter")
    print(docs.__len__())
    return docs
docs = loaderSplitter()

def saveToChromaDb():
    print("save to db")
    coheredb = Chroma.from_documents(
        collection_name="rag-cohere",
        persist_directory="./chroma_db/cohere/sentencefull",
        documents=docs,
        embedding=CohereEmbeddings()
    )

saveToChromaDb()