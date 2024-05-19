# This module will be edited in Lab 03 to add the agent tools.
import boto3
from langchain.agents import Tool
from langchain_community.utilities import SearchApiAPIWrapper
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from .rag import get_rag_chain
from langchain.llms.bedrock import Bedrock
from prompt.prompts import CLAUDE_AGENT_PROMPT
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
import requests
import os
from dotenv import load_dotenv
load_dotenv()

bedrock_runtime = boto3.client(
    "bedrock-runtime", 
    region_name='us-west-2',
    aws_access_key_id=os.environ["AccessKey"],
    aws_secret_access_key=os.environ["SecretAccessKey"],
)

boto_session = boto3.Session(
    region_name='us-west-2',
    aws_access_key_id=os.environ["AccessKey"],
    aws_secret_access_key=os.environ["SecretAccessKey"],
)

claude_llm = Bedrock(
    model_id='anthropic.claude-v2:1',
    client=bedrock_runtime,
    model_kwargs={"max_tokens_to_sample": 500, "temperature": 0.0},
)

search = SearchApiAPIWrapper(searchapi_api_key=os.environ["Search_Api_Key"], engine= "google")

rag_qa_chain = get_rag_chain(claude_llm)

def jsSearch(query):
    url = os.environ["JS_URL"]+"/crawler"
    params = {
        "query": query  # You may want to decode the query if it's URL-encoded
    }
    response = requests.get(url, params=params)
    return response.json()


LLM_AGENT_TOOLS = [
    Tool(
        name="SemanticSearch",
        func=lambda query: rag_qa_chain({"question": query}),
        description=(
            "當你被問到關於高雄的景點推薦時，請使用這項工具"
            "Use when you are asked questions about attraction recommendations in Taiwan kaohsiung"
            " You should ask targeted questions."
        ),
    ),
    Tool(
        name="Search",
        # func=search.run,
        func=jsSearch,
        description=(
            "當你被問到高雄的旅遊活動或是近期活動時，請使用這項工具"
            "Use Only when you need to answer questions about recent events in Kaohsiung or attraction recommendations in Kaohsiung"
            " You should ask targeted questions."
        ),
    ),
]

def get_agentic_chatbot_conversation_chain(
    user_input, session_id, clean_history, verbose=True
):
    message_history = DynamoDBChatMessageHistory(
        table_name="chat_message", session_id=session_id, boto3_session=boto_session
    )
    if clean_history:
        message_history.clear()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=message_history,
        ai_prefix="AI",
        # change the human_prefix from Human to something else
        # to not conflict with Human keyword in Anthropic Claude model.
        human_prefix="Hu",
        return_messages=False,
    )

    agent = create_react_agent(
        llm=claude_llm,
        tools=LLM_AGENT_TOOLS,
        prompt=CLAUDE_AGENT_PROMPT,
    )

    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=LLM_AGENT_TOOLS,
        verbose=verbose,
        memory=memory,
        handle_parsing_errors="Check your output and make sure it conforms!",
    )
    conversation_chain = agent_chain.invoke
    try:
        response = conversation_chain({"input": user_input})
        print(response)
        response = response["output"]

    except Exception as e:
        print(repr(e))
        response = (
            "您的問題好像不夠明確呢，可以再跟我講講嗎？"
        )

    return {"statusCode": 200, "response": response}