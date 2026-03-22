from fastapi import FastAPI

from app.api.endpoints import auth, rag, agent, docs

app = FastAPI()

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(rag.router, prefix="/api/rag", tags=["rag"])
app.include_router(agent.router, prefix="/api/agent", tags=["agent"])
app.include_router(docs.router, prefix="/api/docs", tags=["docs"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Enterprise Knowledge Base Agent!"}

