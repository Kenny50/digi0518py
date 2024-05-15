summarization_description="Prompt template for generating summary of overall chat",
summarization_prompt = """
<br><br>Human: Answer the questions below, defined in <question></question> based on the 
transcript defined in <transcript></transcript>. If you cannot answer the question, 
reply with 'n/a'. Use gender neutral pronouns. When you reply, only respond with the 
answer. Do not use XML tags in the answer.<br><br>
<question>What is a summary of the transcript?</question><br><br><transcript><br>{
transcript}<br></transcript><br><br>Assistant:
"""

topic_description="Prompt template for finding primary topic"
topic_prompt="""
<br><br>Human: Answer the questions below, defined in <question></question> based on the 
transcript defined in <transcript></transcript>. If you cannot answer the question, 
reply with 'n/a'. Use gender neutral pronouns. When you reply, only respond with the 
answer. Do not use XML tags in the answer.<br><br>
<question>What is the topic of the call? For example, iphone issue, 
billing issue, cancellation. Only reply with the topic, 
nothing more.</question><br><br><transcript><br>{transcript}<br></transcript><br><br>Assistant:"""