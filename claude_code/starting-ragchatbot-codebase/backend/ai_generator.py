import json
from groq import Groq
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Groq API for generating responses"""

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
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """Generate AI response with optional tool usage and conversation context."""
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]

        api_params = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 800,
        }

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**api_params)
        message = response.choices[0].message

        # Handle tool execution if needed
        if message.tool_calls and tool_manager:
            return self._handle_tool_execution(message, messages, tool_manager)

        return message.content

    def _handle_tool_execution(self, assistant_message, messages: List[Dict], tool_manager):
        """Handle execution of tool calls and get follow-up response."""
        messages = messages.copy()
        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            # Strip null values to avoid schema validation issues
            args = {k: v for k, v in args.items() if v is not None}
            tool_result = tool_manager.execute_tool(
                tool_call.function.name,
                **args
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

        final_response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=800,
        )

        return final_response.choices[0].message.content
