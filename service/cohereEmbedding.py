import cohere
import os
from dotenv import load_dotenv
from typing import List

load_dotenv()
co = cohere.Client(os.environ["COHERE_KEY"])

class CohereEmbeddings():
    cohere = cohere.Client(os.environ["COHERE_KEY"])
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        print("calling embeddings")
        # print(texts)
        embeddings = self.cohere.embed(texts=texts, input_type="search_document", model="embed-multilingual-v3.0").embeddings
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:

        return self.cohere.embed(texts=[text], input_type="search_query", model="embed-multilingual-v3.0").embeddings[0]
