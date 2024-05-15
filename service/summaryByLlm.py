from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from summarizer import Summarizer
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from llm import claude_llm
import json

path = '/Users/chang/startup/ai/competition/digitime0518/data/review.json'
reviews = json.load(open(path))

def mapToString(obj):
    return obj['text']
rs = list(map(mapToString,reviews))

def prepareForBert():

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=1_000, chunk_overlap=300)
    docs = text_splitter.create_documents(rs)

    print (f"Data is split into {len(docs)} documents")

    model = Summarizer()

    def get_summary(text):
        return model(text, num_sentences=1)

    # 這邊會很久
    bert_summary = []
    for doc in docs:
        bert_summary += [get_summary(doc.page_content)]
    
    return bert_summary
bs = prepareForBert()

bert_summary_res = "\n".join(bs)

def getSummary(textContent):
    summaries = Document(page_content=textContent)
    combine_prompt = """
    Provide a detailed summary of the following:
    {text}
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    reduce_chain = load_summarize_chain(
        llm=claude_llm,
        chain_type="stuff",
        prompt=combine_prompt_template,
    )
    output = reduce_chain.run([summaries])
    return output

print(getSummary(bert_summary_res))
