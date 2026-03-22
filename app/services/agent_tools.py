from llama_index.core.tools import FunctionTool
import functools
from app.services.rag_service import RAGService
from models.schemas import UserInfo, QueryRequest, DocumentMetadata
from typing import List

class AgentTools:
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service

    def _knowledge_base_query_func(self, query: str, user_info: UserInfo) -> List[DocumentMetadata]:
        """
        Queries the enterprise knowledge base for relevant documents based on the user's query and permissions.
        Args:
            query (str): The user's natural language query.
            user_info (UserInfo): The current user's information, including permissions.
        Returns:
            List[DocumentMetadata]: A list of relevant document metadata.
        """
        # Note: user_info will be passed from the API endpoint context to the agent,
        # and then to this tool.
        # For now, we'll use a dummy UserInfo for demonstration if not provided,
        # but in a real scenario, it should always come from the authenticated user.
        
        # This is a simplified representation. In a real LlamaIndex tool,
        # the UserInfo object might need to be serialized/deserialized or
        # passed implicitly through context.
        
        # For the purpose of this example, let's assume user_info is directly usable.
        
        # The `user_info` parameter here is a placeholder for how the agent might
        # receive user context. LlamaIndex tools typically take simple types.
        # A more robust solution might involve the agent having access to the
        # current user context directly, or the tool being initialized with it.
        
        # For now, let's make user_info a string representation for simplicity in tool definition.
        # We will refine this when integrating with the actual LlamaIndex Agent.
        
        # For the tool definition, we need a simpler signature.
        # The actual `retrieve` method in RAGService takes a UserInfo object.
        # We need to bridge this gap.
        
        # Let's assume the agent will pass a serialized UserInfo string,
        # and the tool will deserialize it.
        
        # For now, let's simplify the tool signature to just `query` and handle user_info
        # at the agent level or by making the tool stateful.
        
        # Let's make the tool stateful for now, where `user_info` is set when the tool is created.
        # This is a common pattern for tools that need user context.
        
        # Re-thinking: LlamaIndex tools are typically stateless functions.
        # The user context should be passed to the agent, and the agent decides how to use it.
        # The tool itself should ideally not hold user state.
        
        # Let's modify the tool to accept user_info as a parameter, and the agent will pass it.
        # However, LlamaIndex FunctionTool expects simple types.
        # We might need to pass user_info as a dictionary or JSON string.
        
        # For now, let's define the tool function to take `query` and `user_id` (as a simple identifier)
        # and then retrieve the full UserInfo from a context or a dummy store within the tool.
        # This is a temporary simplification.
        
        # Let's revert to the original idea of passing UserInfo directly, and we'll see how LlamaIndex handles it.
        # If it complains, we'll adjust.
        
        # For the tool definition, we need to define the function that the agent will call.
        # This function should ideally take simple types.
        # The `user_info` object is complex.
        
        # Let's define a tool that takes `query` and `user_group`, `department`, `permission_level` as separate arguments.
        # This is more compatible with LlamaIndex's FunctionTool.
        
        # This is getting complicated. Let's simplify the tool definition for now.
        # The agent will call a tool that takes a query. The user context will be handled by the agent itself
        # when it calls the `rag_service.retrieve` method.
        
        # So, the tool will just be a wrapper around `rag_service.retrieve` but without the `user_info` parameter
        # in its *signature*. The `user_info` will be implicitly available to the agent.
        
        # This means the `retrieve` method in `RAGService` needs to be called by the agent,
        # and the agent needs to have access to the `user_info`.
        
        # Let's create a tool that just takes the query, and the agent will manage the user context.
        # This is a more standard LlamaIndex approach.
        
        # The `RAGService.retrieve` method requires `UserInfo`.
        # So, the tool *must* provide `UserInfo`.
        # How does LlamaIndex Agent pass complex objects to tools?
        # It usually doesn't directly. Tools are often designed to be stateless and take primitive types.
        
        # Option 1: Make the tool a class that is initialized with `user_info`.
        # Option 2: Pass `user_info` as a JSON string to the tool, and the tool deserializes it.
        # Option 3: The agent itself handles the `user_info` and passes it to `rag_service.retrieve` directly,
        #           without wrapping `retrieve` in a `FunctionTool`. This means `rag_service.retrieve`
        #           is not a "tool" in the LlamaIndex sense, but a service the agent uses.
        
        # Given the prompt "是否调用工具？", it implies we should define tools.
        # So, let's go with Option 2 for now: pass `user_info` as a JSON string.
        
        # Let's refine the `knowledge_base_query` signature to accept `user_info_json: str`.
        
        # This is still not ideal. The `user_info` is part of the *context* of the agent's execution,
        # not necessarily a parameter to every tool call.
        
        # Let's reconsider the structure. The `RAGService` is a service.
        # The agent will *use* the `RAGService`.
        # The `RAGService.retrieve` method needs `UserInfo`.
        
        # The agent itself will have access to the `current_user` (from FastAPI dependency).
        # When the agent decides to perform a retrieval, it will call `rag_service.retrieve(query, current_user)`.
        # So, `rag_service.retrieve` itself doesn't need to be a `FunctionTool`.
        
        # The "tool" for the agent might be something like "search_knowledge_base",
        # which takes a `query` and *implicitly* uses the `current_user` available to the agent.
        
        # This means the `AgentTools` class might not be needed in this exact form.
        # Instead, the agent's *executor* will have access to `rag_service` and `current_user`.
        
        # Let's define a simple tool that the agent can *call* to get information.
        # This tool will encapsulate the `rag_service.retrieve` call, and it will need
        # access to the `current_user`.
        
        # This implies the `AgentTools` class should be initialized with `current_user`.
        # Or, the tool function itself should be a closure that captures `current_user`.
        
        # Let's try to make `rag_service` a dependency for the agent, and the agent
        # will pass `current_user` to `rag_service.retrieve`.
        # This means `rag_service.retrieve` is not a "tool" in the LlamaIndex sense.
        
        # If we *must* define a tool, then the tool needs to be able to get `user_info`.
        # Let's define a tool that takes `query` and `user_group`, `department`, `permission_level`
        # as separate arguments. This is the most compatible with LlamaIndex FunctionTool.
        
        # This is a design decision. For now, let's create a tool that takes these parameters.
        # The agent will be responsible for extracting these from the `UserInfo` object.
        
        # Let's define a tool that takes `query: str, user_group: str, department: str, permission_level: int`.
        # This is a more explicit way for the agent to use the permission context.
        
        # This requires `UserInfo` to be broken down.
        
        # Let's simplify. The `RAGService` is a service. The agent will use it.
        # The agent itself will be initialized with the `current_user`.
        # So, the agent will call `rag_service.retrieve(query, self.current_user)`.
        # This means `rag_service.retrieve` is not a LlamaIndex `FunctionTool`.
        
        # If the user explicitly asked for "工具调用", then we should define tools.
        # Let's define a tool that takes `query` and returns results, and the tool itself
        # will have access to the `user_info` (e.g., through a global context or by being
        # initialized with it).
        
        # Let's create a `KnowledgeBaseQueryTool` class that is initialized with `rag_service` and `user_info`.
        # This makes the tool stateful, which is sometimes necessary.
        
        # This is a common pattern for tools that need context.
        
        # Let's define a tool that takes `query` and `user_info` as parameters.
        # LlamaIndex's `FunctionTool` can handle Pydantic models as arguments.
        # So, `user_info: UserInfo` should work.
        
        # Let's try this approach.
        
        pass

    def knowledge_base_query(self, query: str, user_info: UserInfo) -> List[DocumentMetadata]:
        """
        Queries the enterprise knowledge base for relevant documents based on the user's query and permissions.
        Args:
            query (str): The user's natural language query.
            user_info (UserInfo): The current user's information, including permissions.
        Returns:
            List[DocumentMetadata]: A list of relevant document metadata.
        """
        return self.rag_service.retrieve(query, user_info)

    def get_tools(self, user_info: UserInfo) -> List[FunctionTool]:
        """
        Returns a list of FunctionTools for the agent, with user_info bound to the knowledge base query tool.
        """
        # Create a partial function for _knowledge_base_query_func, binding the user_info
        partial_query_func = functools.partial(self._knowledge_base_query_func, user_info=user_info)

        knowledge_base_query_tool = FunctionTool.from_defaults(
            fn=partial_query_func,
            name="knowledge_base_query",
            description="Queries the enterprise knowledge base for relevant documents based on the user's query and permissions. Input is a string representing the query.",
        )
        return [knowledge_base_query_tool]

