from google import genai
from google.genai import types
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Google's Gemini API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _build_gemini_tools(self, tools: List[Dict]) -> List[types.Tool]:
        """Convert tool definition dicts to Gemini Tool objects."""
        declarations = []
        for tool_def in tools:
            declarations.append(types.FunctionDeclaration(
                name=tool_def["name"],
                description=tool_def["description"],
                parameters=tool_def["parameters"],
            ))
        return [types.Tool(function_declarations=declarations)]

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system instruction with conversation history
        system_instruction = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Build config
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0,
            max_output_tokens=800,
        )

        # Add tools if available
        if tools:
            config.tools = self._build_gemini_tools(tools)

        # Build user message
        contents = [types.Content(role="user", parts=[types.Part(text=query)])]

        # Get response from Gemini
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        # Handle tool execution if needed
        if response.function_calls and tool_manager:
            return self._handle_tool_execution(response, contents, config, tool_manager)

        # Return direct response
        return response.text

    def _handle_tool_execution(self, initial_response, contents: List, config, tool_manager):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool use requests
            contents: Message history so far
            config: GenerateContentConfig
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Add model's response (with function calls) to history
        contents.append(initial_response.candidates[0].content)

        # Execute all tool calls and build function response parts
        tool_response_parts = []
        for fn_call in initial_response.function_calls:
            tool_result = tool_manager.execute_tool(
                fn_call.name,
                **fn_call.args
            )
            tool_response_parts.append(
                types.Part.from_function_response(
                    name=fn_call.name,
                    response={"result": tool_result}
                )
            )

        # Add tool results as user message
        contents.append(types.Content(role="user", parts=tool_response_parts))

        # Remove tools from config so model generates a text response
        config.tools = None

        # Get final response
        final_response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return final_response.text
