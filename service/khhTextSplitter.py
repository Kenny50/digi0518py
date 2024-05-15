import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import cohere
import os
from dotenv import load_dotenv

load_dotenv()
co = cohere.Client(os.environ["COHERE_KEY"])
from typing import List
from langchain_core.embeddings import Embeddings


class CohereEmbeddings():
    cohere = cohere.Client(os.environ["COHERE_KEY"])
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        print("calling embeddings")
        # print(texts)
        embeddings = self.cohere.embed(texts=texts, input_type="search_document", model="embed-multilingual-v3.0").embeddings
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:

        return self.cohere.embed(texts=[text], input_type="search_query", model="embed-multilingual-v3.0").embeddings[0]



# Load data from the JSON file
with open('/Users/chang/startup/ai/competition/digitime0518/data/khhattractions.json') as f:
    khh = json.load(f)

def metadata_func(record:dict, metadata: dict) -> dict:
    metadata['id'] = record.get('id')
    metadata['url'] = record.get('url')
    metadata['name'] = record.get('name')
    return metadata
# Take a subset for demonstration
# shortKhh = khh[:2]

loader = JSONLoader(
    file_path='/Users/chang/startup/ai/competition/digitime0518/data/khhtwo.json',
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
    print(docs[0])
    print(docs.__len__())
    return docs
docs = loaderSplitter()


# all_emb = co.embed(texts=docs, input_type="search_document", model="embed-multilingual-v3.0")

def saveToChromaDb():
    print("save to db")
    coheredb = Chroma.from_documents(
        collection_name="rag-cohere",
        persist_directory="./chroma_db/cohere/sentence",
        documents=docs,
        embedding=CohereEmbeddings()
    )

    # db = Chroma.from_documents(
    #     documents=docs,
    #     collection_name="rag-chroma",
    #     embedding=GPT4AllEmbeddings(),
    #     persist_directory="./chroma_db/json-sample/tnn/en"
    # )

# saveToChromaDb()

def useChromaDb():
    print("using db")
    useDb = Chroma(
        collection_name="rag-cohere",
        persist_directory="./chroma_db/cohere/sentence",
        embedding_function=CohereEmbeddings()
    )

    result = useDb.similarity_search_with_score("I want to see sun rise", k=2)
    print(result)

useChromaDb()
# List to store processed data
processed_data = []

def recurCharSplitter():
    # define a function that returns the cube of `num`
    def getDesc(obj):
        return obj['description']
    descs= list(map(getDesc, khh[:2]))
    # print(descs)
    docs = textSplitter.create_documents(descs)
    for d in docs:
        print(d)
        print("\n")
    print(docs.__len__())

def jeiba():
    # Iterate through each object
    for obj in khh:
        # Split the text using jieba.cut
        seg_list = list(jieba.cut(obj['description'], use_paddle=False))
        
        # Store original data along with split text
        obj['segmented_text'] = seg_list
        processed_data.append(obj)

# print("Length of khh:", len(khh))
# print("Length of processed_data:", len(processed_data))

# # Save processed data to a new file
# with open('data/splitter/khh.json', 'w') as f:
#     json.dump(processed_data, f, ensure_ascii=False, indent=4)
# recurCharSplitter()