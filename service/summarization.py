from langchain_community.document_loaders import JSONLoader
import tiktoken
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
import numpy as np
from sklearn.cluster import KMeans
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from llm import claude_llm

loader = JSONLoader(
    file_path='/Users/chang/startup/ai/competition/digitime0518/data/sumSample.json',
    jq_schema='.[]',
    content_key='.description',
    is_content_key_jq_parsable=True)

# list<document>
data = loader.load()
text = ""

for page in data:
    text += page.page_content
    
text = text.replace('\t', ' ')

# openai embedding encoding model name
encoding = tiktoken.get_encoding('cl100k_base')
num_tokens = len(encoding.encode(text))

print(num_tokens)


def summaryByClustering():
    embeddings = GPT4AllEmbeddings()
    vectors = embeddings.embed_documents([x.page_content for x in data])

    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(vectors)
    closest_indices_list = []

    for i in range(n_clusters):
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        closest_index = np.argmin(distances)
        closest_indices_list.append(closest_index)

    closest_indices_list.sort()

    map_prompt = """
    Provide a short summary of the following in a single paragraph: 
    {text}
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])


    map_chain = load_summarize_chain(
        llm=claude_llm,
        chain_type="stuff",
        prompt=map_prompt_template
    )
    selected_docs = [data[doc] for doc in closest_indices_list]


    new_line = "\n"

    summary_list = []
    for i, doc in enumerate(selected_docs):
        chunk_summary = map_chain.run([doc])
        summary_list.append(chunk_summary)
        print(f"[{i+1}] Cluster Centroid Summary: {chunk_summary}")
        print()

# summaryByClustering()