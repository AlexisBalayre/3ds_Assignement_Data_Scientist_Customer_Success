import json
import logging
from enum import Enum
from typing import List, Optional, Dict
import time

import requests
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI

from config import Config
from knowledge_graph_retriever import KnowledgeGraphRetriever
from compute_embedding import compute_embedding

logger = logging.getLogger(__name__)

config = Config()


class AuraCategoryEnum(str, Enum):
    """Enumeration for Aura support ticket categories.

    This enum defines the standardized categories used to classify
    user questions and support tickets in the Aura chatbot system.
    Each category has a specific code for database storage and tracking.

    Attributes:
        ACCESS: Login, permissions, and account access issues (cat_001)
        RESPONSES: Response quality, accuracy, and relevance problems (cat_002)
        FEATURES: Functionality requests or feature availability questions (cat_003)
        INTEGRATION: External system, API, and tool connections (cat_004)
        TRAINING: Usage guidance, best practices, and learning resources (cat_005)
        PERFORMANCE: Speed, reliability, and system performance issues (cat_006)
        BUG_REPORT: Technical bugs, errors, and unexpected behavior (cat_007)
        ACCOUNT_MANAGEMENT: User profiles, settings, preferences, billing (cat_008)
        DOCUMENTATION: Missing, unclear, or inadequate documentation (cat_009)
    """

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
    """Pydantic model for structured analysis of user questions about Aura chatbot.

    This model provides a standardized way to analyze and categorize user questions,
    enabling better routing, response generation, and ticket management.

    Attributes:
        is_related_to_aura: Whether the question pertains to Aura chatbot functionality
        category: The most appropriate support category for the question
        title: A concise summary of the question (max 10 words)
        description: Detailed explanation capturing the key issue or request
    """

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
        """Validate that title is concise and within word limit.

        Args:
            v: The title string to validate

        Returns:
            str: The validated and stripped title
        """
        word_count = len(v.split())
        if word_count > 10:
            raise ValueError("Title should not exceed 10 words")
        return v.strip()

    @field_validator("description")
    def validate_description(cls, v):
        """Validate that description is not empty.

        Args:
            v: The description string to validate

        Returns:
            str: The validated and stripped description
        """
        if not v or not v.strip():
            raise ValueError("Description cannot be empty")
        return v.strip()


class LLMHandler:
    """Handler for Large Language Model operations with tool integration for Aura chatbot support.

    This class manages LLM interactions, question analysis, knowledge base retrieval,
    and streaming responses for the Aura chatbot support system. It integrates with
    Ollama for local LLM inference and provides tools for finding similar support tickets.

    Attributes:
        config: Configuration instance with API settings
        st: Streamlit-like interface for UI interactions
        kg_retriever: Knowledge graph retriever for ticket search
        tools: List of available tools for LLM function calling
        top_k: Number of similar tickets to retrieve (default: 5)
        context_comments: Number of context comments to include (default: 3)
        min_similarity_score: Minimum similarity threshold (default: 0.85)
        model_name: Name of the LLM model to use
        temperature: Temperature for response generation
        max_tokens: Maximum tokens for LLM responses
        llm_client: OpenAI client configured for Ollama
        results: Last retrieved similar tickets
    """

    def __init__(self, st):
        """Initialize the LLM Handler with configuration and dependencies.

        Args:
            st: Streamlit-like interface object for UI interactions and logging.
                Must have methods: error(), warning(), markdown(), empty()

        Raises:
            Exception: If configuration loading or client initialization fails
        """
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
        """Analyze and categorize a user question about the Aura chatbot.

        Uses an LLM to perform structured analysis of user questions, determining
        relevance to Aura, appropriate category, and generating a concise title
        and detailed description for better support routing.

        Args:
            question: The raw user question to analyze

        Returns:
            UserQuestionAnalysis: Structured analysis containing:
                - is_related_to_aura: Boolean relevance flag
                - category: Support category enum value
                - title: Concise question summary (≤10 words)
                - description: Detailed issue explanation
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

    def _find_similar_tickets(self, question: str) -> List[Dict]:
        """Find similar support tickets using vector similarity search.

        Searches the knowledge base for tickets similar to the user's question
        by analyzing the question, computing embeddings, and retrieving matching
        tickets with their solution comments and context.

        Args:
            question: The exact user question as asked to the LLM

        Returns:
            List[Dict]: List of similar tickets, each containing:
                - title: Ticket title
                - ticketId: Unique ticket identifier
                - similarityScore: Similarity score (0.0-1.0)
                - description: Ticket description
                - resolutionSummary: Summary of resolution
                - solutionComment: Primary solution comment with metadata
                - contextComments: List of related comments for context
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
        """Setup and configure available tools for LLM function calling.

        Initializes the function calling tools that the LLM can use to enhance
        its responses. Currently includes the ticket similarity search tool.

        Returns:
            List[Dict]: List of tool definitions with OpenAI function calling format.
                Each tool dict contains:
                - type: "function"
                - name: Tool function name
                - description: When and how to use the tool
        """
        tools = []
        tools.append(
            {
                "type": "function",
                "name": "find_similar_tickets",
                "description": "Search for similar support tickets in the knowledge base when users have technical issues, problems, or specific questions about Aura that might have been solved before. Use this to find relevant past solutions and help resolve user issues more effectively.",
            }
        )
        return tools

    def _get_system_prompt_with_tool_guidance(self, base_system_prompt: str) -> str:
        """Enhance system prompt with comprehensive tool usage guidelines.

        Adds detailed instructions to the system prompt about when and how to use
        available tools, helping the LLM make better decisions about tool usage.

        Args:
            base_system_prompt: The original system prompt text

        Returns:
            str: Enhanced prompt with tool usage guidelines appended
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
        """Update configuration settings for the LLM handler.

        Allows runtime modification of various parameters that control
        LLM behavior and knowledge retrieval without reinitializing the handler.

        Args:
            model_name: Name of the LLM model to use (e.g., "llama2", "mistral")
            temperature: Controls randomness in responses (0.0-2.0)
                - 0.0: Deterministic, focused responses
                - 1.0: Balanced creativity and consistency
                - 2.0: Very creative, potentially inconsistent
            max_tokens: Maximum tokens in LLM response (positive integer)
            top_k: Number of similar tickets to retrieve (1-20 recommended)
            context_comments: Number of context comments per ticket (1-10 recommended)
            min_similarity_score: Minimum similarity threshold (0.0-1.0)
                - Higher values: More precise but fewer results
                - Lower values: More results but potentially less relevant
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
        """Generate streaming LLM responses with optional tool usage.

        Processes user input through an LLM with support for function calling,
        conversation history, and streaming output. Integrates tool usage
        for enhanced responses when appropriate.

        Args:
            model_name: Name/identifier of the LLM model to use
            messages_history: List of previous conversation messages
                Each message should have 'role' and 'content' keys
            user_prompt: Current user input to process
            system_prompt: System instructions for the LLM
            temperature: Response randomness control (0.0-2.0)
            history_length: Number of previous messages to include (1-50)
            max_tokens: Maximum tokens in response (1-4096)
            use_tools: Whether to enable function calling tools

        Yields:
            str: Individual response tokens as they are generated
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

                    # 1) Create a placeholder
                    msg_placeholder = self.st.empty()
                    msg_placeholder.markdown(
                        "### 🤖 Agent with Tools\n"
                        "The agent will analyze your question and decide if it needs to use tools to find relevant past tickets."
                    )
                    time.sleep(1)  # Give time for the placeholder to render

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
                        messages.append(response_message)

                        for tool_call in response_message.tool_calls:
                            if tool_call.function.name == "find_similar_tickets":
                                msg_placeholder.markdown(
                                    "### 🔍 Finding Similar Tickets\n"
                                    "The agent has decided to use the tool to find similar past tickets."
                                )

                                logger.info(
                                    f"Tool call detected: {tool_call.function.name}"
                                )
                                tool_response = self._find_similar_tickets(
                                    question=user_prompt
                                )

                                if isinstance(tool_response, list) and len(tool_response) == 0:
                                    r = [
                                        {
                                            "error": "No similar tickets found for the provided question."
                                        }
                                    ]

                                if tool_response:
                                    msg_placeholder.markdown(
                                        "### 📄 Similar Tickets Found\n"
                                        f"The agent found {len(tool_response)} similar tickets."
                                    )
                                    time.sleep(
                                        2
                                    )  # Give time for the placeholder to render

                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": json.dumps(tool_response, indent=2),
                                    }
                                )

                        msg_placeholder.empty()  # Clear the placeholder after tool call

                        # Continue the conversation with the tool response
                        stream = self.llm_client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stream=True,
                            tools=self.tools,
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
                        messages.append(
                            {
                                "role": "assistant",
                                "content": response_message.content,
                            }
                        )
                        # No tool calls were made, just return the response
                        if response_message.content:
                            yield response_message.content

                except Exception as e:
                    self.st.warning(
                        f"Agent with tools failed, falling back to direct LLM: {str(e)}"
                    )
                    logger.error(f"Agent error: {str(e)}")

        except Exception as e:
            error_msg = f"Error in LLM streaming: {str(e)}"
            logger.error(error_msg)
            yield error_msg

    def get_available_models(self) -> List[dict]:
        """Retrieve list of available Ollama models from local API.

        Queries the local Ollama API to get currently installed and available
        models for use with the LLM handler. Useful for dynamic model selection
        and validation.

        Returns:
            List[Dict]: List of available models with metadata. Each model dict contains:
                - name: Model name/identifier
                - size: Model file size
                - modified_at: Last modification timestamp
                - digest: Model hash/digest
                Empty list if API call fails or no models available
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
        """Get mapping of available tools and their descriptions.

        Provides a convenient way to inspect what tools are configured
        for the LLM handler, useful for debugging and UI display.

        Returns:
            Dict[str, str]: Mapping of tool names to their descriptions
                - Keys: Tool function names (e.g., "find_similar_tickets")
                - Values: Human-readable descriptions of tool functionality
                Empty dict if no tools configured or names missing
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
