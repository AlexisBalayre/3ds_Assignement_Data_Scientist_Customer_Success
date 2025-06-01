import json
import logging
from enum import Enum
from typing import Any, List, Optional, Dict
from datetime import date, datetime

import pandas as pd
import requests
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI

from config import Config
from knowledge_graph_retriever import KnowledgeGraphRetriever
from compute_embedding import compute_embedding

logger = logging.getLogger(__name__)

config = Config()


class AuraCategoryEnum(str, Enum):
    """Enumeration for Aura support categories."""

    ACCESS = "cat_001"
    RESPONSES = "cat_002"
    FEATURES = "cat_003"
    INTEGRATION = "cat_004"
    TRAINING = "cat_005"
    PERFORMANCE = "cat_006"
    BUG_REPORT = "cat_007"
    ACCOUNT_MANAGEMENT = "cat_008"
    DOCUMENTATION = "cat_009"


class UserQuestionAnalysis(BaseModel):
    """Pydantic model for user question analysis."""

    is_related_to_aura: bool = Field(
        default=True, description="Indicates if the question is related to Aura chatbot"
    )
    category: Optional[AuraCategoryEnum] = Field(
        None, description="Category of the question based on the content"
    )
    title: str = Field(
        ..., description="Concise title for the question (max 10 words)", max_length=100
    )
    description: str = Field(
        ..., description="Detailed description capturing the key issue or request"
    )

    @field_validator("title")
    def validate_title_length(cls, v):
        """Ensure title is concise."""
        word_count = len(v.split())
        if word_count > 10:
            raise ValueError("Title should not exceed 10 words")
        return v.strip()

    @field_validator("description")
    def validate_description(cls, v):
        """Ensure description is not empty."""
        if not v or not v.strip():
            raise ValueError("Description cannot be empty")
        return v.strip()


class LLMHandler:
    def __init__(self, st):
        self.config = Config()
        self.st = st

        self.kg_retriever = KnowledgeGraphRetriever()
        self.tools = self._setup_tools()

        self.top_k = 5  # Number of similar tickets to retrieve
        self.context_comments = 3  # Number of context comments to include
        self.min_similarity_score = 0.85  # Minimum similarity score for results

        self.model_name = self.config.ollama_model  # Default model name
        self.temperature = (
            self.config.ollama_temperature
        )  # Temperature for LLM responses
        self.max_tokens = self.config.ollama_max_tokens  # Max tokens for LLM responses

        self.llm_client = OpenAI(
            base_url=self.config.ollama_base_url,  # Ollama API base URL
            api_key="ollama",
        )

    def _analyze_question(self, question: str) -> UserQuestionAnalysis:
        """
        Analyze a user question about the Aura chatbot and categorize it.

        Args:
            question (str): The user question to analyze.

        Returns:
            UserQuestionAnalysis: Structured analysis including whether it's related
                                  to Aura, its category, a short title, and a detailed description.
        """

        prompt_system = """# Identity
            You are an assistant that analyzes user questions about the Aura chatbot and maps them into a UserQuestionAnalysis Pydantic model with fields for relevance, category, title, and description.

            # Instructions
            Analyze the user's question and determine:

            1. **is_related_to_aura** (bool): true if question relates to Aura chatbot, false otherwise

            2. **category** (AuraCategoryEnum): Choose the most appropriate:
            - **ACCESS**: login, permissions, account access issues
            - **RESPONSES**: response quality, accuracy, relevance problems
            - **FEATURES**: functionality requests or feature availability questions
            - **INTEGRATION**: connecting with external systems, APIs, tools
            - **TRAINING**: usage guidance, best practices, learning resources
            - **PERFORMANCE**: speed, reliability, system performance issues
            - **BUG_REPORT**: technical bugs, errors, unexpected behavior
            - **ACCOUNT_MANAGEMENT**: user profiles, settings, preferences, billing
            - **DOCUMENTATION**: missing, unclear, or inadequate documentation

            3. **title** (str): Concise summary, maximum 10 words, clear and actionable

            4. **description** (str): Detailed explanation of the issue or request, 2-3 sentences providing context and specifics

            Return a valid UserQuestionAnalysis model. For ambiguous cases, choose the primary concern and note secondary issues in the description."""

        try:
            analysis_result = self.llm_client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": prompt_system,
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this question: {question}",
                    },
                ],
                response_format=UserQuestionAnalysis,
                temperature=self.temperature,
            )

            logger.info(
                f"Question analysis result: {json.dumps(analysis_result.choices[0].message.parsed.model_dump(), indent=2)}"
            )

            return analysis_result.choices[0].message.parsed

            return analysis_result
        except Exception as e:
            # Log the error in the provided Streamlit-like logger
            self.st.error(f"Error analyzing question: {str(e)}")
            return UserQuestionAnalysis(
                is_related_to_aura=False,
                category=None,
                title="Error",
                description=str(e),
            )

    def _find_similar_tickets(
        self,
        question: str,
    ) -> List[Dict]:
        """
        Find similar tickets based on the raw user question.

        Use this tool when the user has a specific technical issue, problem, or question
        that might have been solved before in past support tickets. This is especially
        useful for:
        - Technical problems or errors
        - How-to questions about Aura features
        - Access issues
        - Bug reports
        - Performance problems

        Do NOT use this tool for:
        - Simple greetings
        - General information requests
        - Questions that can be answered with general knowledge

        Args:
            question (str): The exact user question as asked to the LLM.

        Returns:
            List[Dict]: List of similar tickets with their details and solutions.
        """
        try:
            logger.info(f"Analyzing question to verify relevance: {question}")
            question_analysis = self._analyze_question(question)

            if not question_analysis.is_related_to_aura:
                logger.warning(
                    "Question is not related to Aura chatbot. No similar tickets found."
                )
                return []

            logger.info(
                f"Finding similar tickets for title: {question_analysis.title}\ndescription: {question_analysis.description}"
            )

            # Combine the title and description into a single text for embedding
            combined_text = (
                question_analysis.title + " " + question_analysis.description
            )

            # Compute an embedding for the combined text
            query_vector = compute_embedding(combined_text)

            # Retrieve similar tickets from the knowledge graph by vector similarity
            results = self.kg_retriever.retrieve_solution_comments_by_vector(
                query_vector,
                min_similarity_score=self.min_similarity_score,
                top_k=self.top_k,
                context_comments=self.context_comments,
            )

            self.results = results

            if not results:
                logger.warning(
                    "No similar tickets found for the provided question analysis."
                )
            return results
        except Exception as e:
            # In case of any error, return a descriptive message
            return [{"error": f"Error finding similar content: {str(e)}"}]

    def _setup_tools(self):
        """
        Setup available tools for the LLM to use.

        Returns:
            list: List of tools available for the LLM to use, including their names and descriptions.
        """
        tools = []
        tools.append(
            {
                "type": "function",
                "name": "find_similar_tickets",
                "description": "Search for similar support tickets in the knowledge base when users have technical issues, problems, or specific questions about Aura that might have been solved before. Use this to find relevant past solutions and help resolve user issues more effectively.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The exact user question as asked to the LLM.",
                        },
                    },
                    "required": ["question"],
                },
            }
        )
        return tools

    def _get_system_prompt_with_tool_guidance(self, base_system_prompt: str) -> str:
        """
        Enhance the system prompt with guidance on when to use tools.

        Args:
            base_system_prompt (str): The base system prompt

        Returns:
            str: Enhanced system prompt with tool usage guidance
        """
        tool_guidance = """

        TOOL USAGE GUIDELINES:
        You have access to a tool called 'find_similar_tickets' that searches the knowledge base for similar past support tickets.

        Use the find_similar_tickets tool when:
        - User reports a specific technical issue, error, or problem
        - User asks how to do something specific in Aura
        - User has access/login problems
        - User reports bugs or performance issues
        - User asks about specific features or functionality
        - The question seems like it might have been asked and solved before

        Do NOT use the tool when:
        - User is just greeting you ("hello", "hi", "how are you")
        - User asks general questions about what Aura is
        - User makes simple requests that don't require past ticket knowledge
        - User is having a casual conversation
        - The question is clearly answerable with general knowledge

        When you do use the tool, incorporate the findings naturally into your response and provide actionable solutions based on the similar tickets found.
        """

        return base_system_prompt + tool_guidance

    def update_settings(
        self,
        model_name: str = None,
        temperature: float = None,
        max_tokens: int = None,
        top_k: int = None,
        context_comments: int = None,
        min_similarity_score: float = None,
    ):
        """
        Update the settings for the LLM handler.

        Args:
            model_name (str): The name of the model to use.
            temperature (float): The temperature for response generation.
            max_tokens (int): The maximum number of tokens for the response.
            top_k (int): Number of similar tickets to retrieve.
            context_comments (int): Number of context comments to include.
            min_similarity_score (float): Minimum similarity score for results.
        """
        if model_name:
            self.model_name = model_name
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if top_k is not None:
            self.top_k = top_k
        if context_comments is not None:
            self.context_comments = context_comments
        if min_similarity_score is not None:
            self.min_similarity_score = min_similarity_score

    def llm_stream(
        self,
        model_name: str,
        messages_history: List[Dict],
        user_prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        history_length: int = 10,
        max_tokens: int = 1024,
        use_tools: bool = True,
    ):
        """
        Synchronous streaming LLM response.

        Args:
            model_name (str): The name or identifier of the model to use.
            messages_history (list): A list of message dicts representing the conversation history.
            user_prompt (str): The user's input prompt.
            system_prompt (str): The system's input prompt.
            temperature (float): Controls randomness (default: 0.0).
            history_length (int): Number of messages to keep in history (default: 10).
            max_tokens (int): Maximum tokens for the response (default: 1024).
            use_tools (bool): Whether to enable tool usage (default: True).

        Yields:
            str: A response token generated by the language model.
        """
        try:
            # Build chat history from recent messages
            chat_history: List[Dict[str, str]] = []
            for message in messages_history[-history_length:]:
                if message.get("role") in ["user", "assistant", "system"]:
                    chat_history.append(
                        {
                            "role": message.get("role"),
                            "content": message.get("content", ""),
                        }
                    )

            # Build final messages list
            messages: List[Dict[str, str]] = []

            # Add system prompt if provided
            if system_prompt:
                messages.append(
                    {
                        "role": "system",
                        "content": self._get_system_prompt_with_tool_guidance(
                            system_prompt
                        ),
                    }
                )

            # Add chat history
            messages.extend(chat_history)

            # Add current user message
            messages.append({"role": "user", "content": user_prompt})

            self.results = []  # Reset results for each new query

            # Use agent with tools if enabled
            if use_tools and self.tools:
                try:
                    # Ask the model to decide if it needs to use tools
                    tool_assessment_response = self.llm_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        tools=self.tools,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    response_message = tool_assessment_response.choices[0].message

                    # Check if the model made a tool call
                    if response_message.tool_calls:
                        # Append the assistant's message with tool calls
                        messages.append(
                            {
                                "role": "assistant",
                                "content": response_message.content,
                                "tool_calls": response_message.tool_calls,
                            }
                        )

                        # Process each tool call
                        for tool_call in response_message.tool_calls:
                            if tool_call.function.name == "find_similar_tickets":

                                logger.info(
                                    f"Tool call detected: {tool_call.function.name}"
                                )
                                tool_response = self._find_similar_tickets(
                                    question=user_prompt
                                )

                                # Append tool response with correct format
                                tool_message = {
                                    "role": "function_call_output",
                                    "tool_call_id": tool_call.id,
                                    "content": str(tool_response),
                                }
                                messages.append(tool_message)

                        # Continue the conversation with the tool response
                        stream = self.llm_client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stream=True,
                        )
                        for chunk in stream:
                            content = chunk.choices[0].delta.content
                            if content:
                                yield content

                        if self.results:
                            self.st.markdown("---")
                            self.st.markdown("### 📚 Sources")

                            for idx, result in enumerate(self.results, 1):
                                title = result.get("title", "No title")
                                ticket_id = result.get("ticketId", "Unknown")
                                similarity_score = result.get("similarityScore", 0)
                                description = result.get("description", "")
                                resolution_summary = result.get("resolutionSummary", "")

                                # Top-level fields
                                self.st.markdown(f"**[{idx}] {title}**  ")
                                self.st.markdown(f"- **Ticket ID:** `{ticket_id}`  ")
                                self.st.markdown(
                                    f"- **Similarity Score:** {similarity_score:.1%}  "
                                )
                                self.st.markdown(f"- **Description:** {description}  ")
                                self.st.markdown(
                                    f"- **Resolution Summary:** {resolution_summary}  "
                                )

                                # Solution Comment
                                sol_comment = result.get("solutionComment", {})
                                if sol_comment:
                                    sol_content = sol_comment.get("content", "")
                                    sol_author_role = sol_comment.get("author", {}).get(
                                        "role", ""
                                    )
                                    sol_creation = sol_comment.get("creationDate", "")
                                    sol_id = sol_comment.get("commentId", "")
                                    self.st.markdown(f"- **Solution Comment:**  ")
                                    self.st.markdown(f"    - Content: {sol_content}  ")
                                    self.st.markdown(
                                        f"    - Author Role: {sol_author_role}  "
                                    )
                                    self.st.markdown(
                                        f"    - Creation Date: {sol_creation}  "
                                    )
                                    self.st.markdown(f"    - Comment ID: {sol_id}  ")

                                # Context Comments
                                context_comments = result.get("contextComments", [])
                                if context_comments:
                                    self.st.markdown(f"- **Context Comments:**  ")
                                    for cc_idx, cc in enumerate(context_comments, 1):
                                        cc_content = cc.get("content", "")
                                        cc_author_role = cc.get("author", {}).get(
                                            "role", ""
                                        )
                                        cc_creation = cc.get("creationDate", "")
                                        cc_id = cc.get("commentId", "")
                                        self.st.markdown(
                                            f"    {cc_idx}. Content: {cc_content}  "
                                        )
                                        self.st.markdown(
                                            f"       - Author Role: {cc_author_role}  "
                                        )
                                        self.st.markdown(
                                            f"       - Creation Date: {cc_creation}  "
                                        )
                                        self.st.markdown(
                                            f"       - Comment ID: {cc_id}  "
                                        )

                                self.st.markdown("")  # Extra line break for spacing

                    else:
                        # No tool calls were made, just return the response
                        if response_message.content:
                            yield response_message.content

                except Exception as e:
                    self.st.warning(
                        f"Agent with tools failed, falling back to direct LLM: {str(e)}"
                    )
                    logger.error(f"Agent error: {str(e)}")

            # Fallback: direct LLM without tools
            stream = self.llm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        except Exception as e:
            error_msg = f"Error in LLM streaming: {str(e)}"
            logger.error(error_msg)
            yield error_msg

    def get_available_models(self) -> List[dict]:
        """
        Get list of available Ollama models by calling the local Ollama API.

        Returns:
            list: List of available models, each represented as a dictionary.
        """
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                return models_data.get("models", [])
            else:
                self.st.warning(
                    f"Ollama API returned status code {response.status_code}"
                )
                return []
        except Exception as e:
            self.st.warning(f"Could not fetch available models: {str(e)}")
            return []

    def get_available_tools(self) -> Dict[str, str]:
        """
        Get list of available tools and their descriptions.

        Returns:
            dict: Dictionary of tool names and descriptions.
        """
        tool_descriptions: Dict[str, str] = {}
        for tool in self.tools:
            tool_name = tool.get("name")
            tool_description = tool.get("description", "")
            if tool_name:
                tool_descriptions[tool_name] = tool_description
            else:
                self.st.warning("Tool name is missing in the configuration.")
        return tool_descriptions


# ---------------------------------------------
# UNIT TESTING (DEMONSTRATION PURPOSES ONLY)
# ---------------------------------------------

# class DummyST:
#     """
#     Dummy Streamlit-like logger to pass into LLMHandler for demonstration purposes.
#     """

#     def error(self, msg: str):
#         print(f"[ERROR] {msg}")

#     def warning(self, msg: str):
#         print(f"[WARNING] {msg}")

# if __name__ == "__main__":
#     # Instantiate the dummy logger
#     st = DummyST()

#     # Create the handler
#     handler = LLMHandler(st)

#     # 1. List available Ollama models
#     try:
#         available_models = handler.get_available_models()
#         print("Available Ollama models:", available_models)
#     except Exception as e:
#         print(f"Could not retrieve models: {e}")

#     # 2. List the tools
#     tools = handler.get_available_tools()
#     print("Available tools and descriptions:")
#     for name, desc in tools.items():
#         print(f"  - {name}: {desc}")

#     # 3. Test different types of questions
#     test_questions = [
#         "Hello, how are you today?",  # Should NOT use tool
#         "I can't access Aura from my account, it says 'access denied'. What should I do?",  # Should use tool
#         "What is Aura?",  # Should NOT use tool
#         "Aura is giving me error code 500 when I try to upload a file",  # Should use tool
#         "How do I reset my password in Aura?",  # Should use tool
#         "Thank you for your help!",  # Should NOT use tool
#     ]

#     system_prompt = """You are an Aura Support Agent helping Dassault Systèmes employees with their Aura chatbot issues.

# **Your mission:** Provide technical support, troubleshoot problems, and guide users through solutions.

# **Key rules:**
# - Be helpful and professional
# - Give complete, step-by-step solutions users can follow themselves
# - Never reference internal tickets or systems users can't access
# - Search knowledge base for technical issues, bugs, access problems, and feature questions
# - Don't search for greetings or general conversations

# **Approach:** Listen to the problem → Search for similar solutions if technical → Provide clear instructions → Offer follow-up help."""

#     for i, question in enumerate(test_questions, 1):
#         print(f"\n{'='*50}")
#         print(f"Test {i}: {question}")
#         print(f"{'='*50}")

#         try:
#             response_text = ""
#             for token in handler.llm_stream(
#                 model_name=handler.model_name,
#                 messages_history=[],
#                 user_prompt=question,
#                 system_prompt=system_prompt,
#                 temperature=0.0,
#                 use_tools=True,
#             ):
#                 print(token, end="", flush=True)
#                 response_text += token
#             print("\n")

#         except Exception as e:
#             print(f"Error with question {i}: {e}")
