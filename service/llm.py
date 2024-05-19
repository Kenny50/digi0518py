# This module will be edited in Lab 03 to add the agent tools.
import boto3
from langchain.llms.bedrock import Bedrock
import os
from dotenv import load_dotenv
load_dotenv()

bedrock_runtime = boto3.client(
    "bedrock-runtime", 
    region_name='us-west-2',
    aws_access_key_id=os.environ["AccessKey"],
    aws_secret_access_key=os.environ["SecretAccessKey"],
)

claude_llm = Bedrock(
    model_id='anthropic.claude-v2:1',
    client=bedrock_runtime,
    model_kwargs={"max_tokens_to_sample": 500, "temperature": 0.0},
)