from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from service.khhCohereUse import useChromaDb
from service.agentic import get_agentic_chatbot_conversation_chain
# import uvicorn

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/vector")
async def vector_query(query, k=50):
    if query:
        result = useChromaDb(query=query, k=k)
        return {"result": result}
    return {"query": None}

@app.get("/rag")
async def rag_query(query, session_id, clean_history: bool):
    print(query)
    print(session_id)
    print(clean_history)
    if query:
        result = get_agentic_chatbot_conversation_chain(
                user_input=query,
                session_id=session_id,
                clean_history= clean_history
            )
        return result
    return {"query": None}

@app.get("/summary")
async def summary():

    return {"query": "None"}

PORT: int  = 8000

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=PORT)  