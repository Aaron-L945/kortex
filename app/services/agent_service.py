import os
from typing import List, Optional

from llama_index.core.agent import ReActAgent
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings

from app.services.rag_service import RAGService
from app.services.agent_tools import AgentTools
from models.schemas import UserInfo

class AgentService:
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        # Assuming ANTHROPIC_API_KEY is set in environment variables or .env file
        # You might need to configure other LLM parameters here
        # For example, model name, temperature, etc.
        return Anthropic(model="claude-3-opus-20240229", temperature=0.1)

    def chat(self, query: str, user_info: UserInfo) -> str:
        # Get tools for the agent, binding user_info to the knowledge base query tool
        agent_tools = AgentTools(self.rag_service)
        tools = agent_tools.get_tools(user_info)

        # Initialize the LlamaIndex Agent
        # We use ReActAgent for its planning and tool-use capabilities
        agent = ReActAgent.from_tools(
            tools, 
            llm=self.llm, 
            verbose=True,
            # Add a system prompt to guide the agent's behavior
            system_prompt=(
                "You are an enterprise knowledge base agent. "
                "Your primary goal is to answer user questions by utilizing the provided tools, "
                "especially the 'knowledge_base_query' tool to retrieve information from the knowledge base. "
                "Always consider the user's permissions when retrieving information. "
                "If the user asks a question that requires knowledge base lookup, use the 'knowledge_base_query' tool. "
                "If you cannot find relevant information, state that you don't have enough information. "
                "Be concise and helpful."
            )
        )
        
        response = agent.chat(query)
        return str(response)
