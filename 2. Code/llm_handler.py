from enum import Enum
import logging
from typing import List, Optional, Dict, Annotated
from pydantic import BaseModel, Field, field_validator

from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.program import LLMTextCompletionProgram
import json
import requests

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

    def _analyze_question(self, question: str) -> UserQuestionAnalysis:
        """
        Analyze a user question about the Aura chatbot and categorize it.

        Args:
            question (str): The user question to analyze.

        Returns:
            UserQuestionAnalysis: Structured analysis including whether it's related
                                  to Aura, its category, a short title, and a detailed description.
        """
        llm_client = Ollama(
            model=self.model_name,
            request_timeout=30.0,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        prompt_template = """\
            Analyze the following user question about Aura (Dassault Systèmes' internal chatbot).

            User Question: "{question}"

            Determine:
            1. Is this question related to Aura chatbot? (true/false)
            2. What category best fits this question from the available categories:
            - ACCESS (cat_001): Login, permissions, account access issues
            - RESPONSES (cat_002): Quality, accuracy, or relevance of Aura's responses
            - FEATURES (cat_003): Functionality, capabilities, or feature requests
            - INTEGRATION (cat_004): Integration with other systems or tools
            - TRAINING (cat_005): How to use Aura, training materials, best practices
            - PERFORMANCE (cat_006): Speed, reliability, or performance issues
            - BUG_REPORT (cat_007): Technical bugs or errors
            - ACCOUNT_MANAGEMENT (cat_008): User profile, settings, preferences
            - DOCUMENTATION (cat_009): Missing or unclear documentation
            3. Create a concise title (max 10 words)
            4. Write a detailed description that captures the key issue or request

            Provide a structured analysis.
        """

        analysis_program = LLMTextCompletionProgram.from_defaults(
            output_cls=UserQuestionAnalysis,
            llm=llm_client,
            prompt_template_str=prompt_template,
            verbose=True,
        )

        try:
            analysis_result = analysis_program(
                question=question,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            logger.info(
                f"Question analysis result: {json.dumps(analysis_result.dict(), indent=2)}"
            )

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

    def _setup_tools(self):
        """
        Setup available tools for the LLM to use.

        Returns:
            list: List of FunctionTool objects.
        """
        tools = []

        # Content similarity tool
        def find_similar_tickets(
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
                )

                if not results:
                    logger.warning(
                        "No similar tickets found for the provided question analysis."
                    )
                return results
            except Exception as e:
                # In case of any error, return a descriptive message
                return [{"error": f"Error finding similar content: {str(e)}"}]

        tools.append(
            FunctionTool.from_defaults(
                fn=find_similar_tickets,
                name="find_similar_tickets",
                description="Search for similar support tickets in the knowledge base when users have technical issues, problems, or specific questions about Aura that might have been solved before. Use this to find relevant past solutions and help resolve user issues more effectively.",
            )
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
        **kwargs,
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
            # Initialize the Ollama LLM client
            llm = Ollama(
                model=model_name,
                request_timeout=120.0,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Set global LLM settings
            Settings.llm = llm

            # Convert the last N messages from plain dicts to ChatMessage objects
            chat_history: List[ChatMessage] = []
            for message in messages_history[-history_length:]:
                if message.get("role") in ["user", "assistant", "system"]:
                    chat_history.append(
                        ChatMessage(
                            role=message.get("role"), content=message.get("content", "")
                        )
                    )

            # Use agent with tools if enabled
            if use_tools and self.tools:
                try:
                    # Enhance system prompt with tool usage guidance
                    enhanced_system_prompt = self._get_system_prompt_with_tool_guidance(
                        system_prompt
                    )

                    # Use ReActAgent
                    agent = ReActAgent.from_tools(
                        tools=self.tools,
                        llm=llm,
                        system_prompt=enhanced_system_prompt,
                        max_iterations=10,
                        verbose=False,
                    )

                    logger.info(f"Agent initialized with {len(self.tools)} tools")

                    # Run the agent
                    response = agent.chat(user_prompt, chat_history=chat_history)

                    # Stream the response character by character
                    response_text = str(response)
                    for char in response_text:
                        yield char

                    return

                except Exception as e:
                    self.st.warning(
                        f"Agent with tools failed, falling back to direct LLM: {str(e)}"
                    )
                    logger.error(f"Agent error: {str(e)}")

            # Fallback: direct LLM without tools
            messages: List[ChatMessage] = []
            if system_prompt:
                messages.append(ChatMessage(role="system", content=system_prompt))
            messages.extend(chat_history)
            messages.append(ChatMessage(role="user", content=user_prompt))

            response = llm.chat(messages)
            response_text = (
                response.message.content
                if hasattr(response, "message")
                else str(response)
            )

            # Stream character by character
            for char in response_text:
                yield char

        except Exception as e:
            error_msg = f"Error in LLM streaming: {str(e)}"
            logger.error(error_msg)
            yield error_msg

    def get_available_models(self) -> List[str]:
        """
        Get list of available Ollama models by calling the local Ollama API.

        Returns:
            list: List of available model names, or an empty list if the request fails.
        """
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                return [model["name"] for model in models_data.get("models", [])]
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
            # Each FunctionTool has metadata with name and description
            tool_descriptions[tool.metadata.name] = tool.metadata.description
        return tool_descriptions


# ---------------------------------------------
# Example usage at the end of the module:
# ---------------------------------------------


class DummyST:
    """
    Dummy Streamlit-like logger to pass into LLMHandler for demonstration purposes.
    """

    def error(self, msg: str):
        print(f"[ERROR] {msg}")

    def warning(self, msg: str):
        print(f"[WARNING] {msg}")


if __name__ == "__main__":
    # Instantiate the dummy logger
    st = DummyST()

    # Create the handler
    handler = LLMHandler(st)

    # 1. List available Ollama models
    try:
        available_models = handler.get_available_models()
        print("Available Ollama models:", available_models)
    except Exception as e:
        print(f"Could not retrieve models: {e}")

    # 2. List the tools
    tools = handler.get_available_tools()
    print("Available tools and descriptions:")
    for name, desc in tools.items():
        print(f"  - {name}: {desc}")

    # 3. Test different types of questions
    test_questions = [
        "Hello, how are you today?",  # Should NOT use tool
        "I can't access Aura from my account, it says 'access denied'. What should I do?",  # Should use tool
        "What is Aura?",  # Should NOT use tool
        "Aura is giving me error code 500 when I try to upload a file",  # Should use tool
        "How do I reset my password in Aura?",  # Should use tool
        "Thank you for your help!",  # Should NOT use tool
    ]

    system_prompt = """You are an Aura Support Agent helping Dassault Systèmes employees with their Aura chatbot issues.

**Your mission:** Provide technical support, troubleshoot problems, and guide users through solutions.

**Key rules:**
- Be helpful and professional
- Give complete, step-by-step solutions users can follow themselves
- Never reference internal tickets or systems users can't access
- Search knowledge base for technical issues, bugs, access problems, and feature questions
- Don't search for greetings or general conversations

**Approach:** Listen to the problem → Search for similar solutions if technical → Provide clear instructions → Offer follow-up help."""

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*50}")
        print(f"Test {i}: {question}")
        print(f"{'='*50}")

        try:
            response_text = ""
            for token in handler.llm_stream(
                model_name=handler.model_name,
                messages_history=[],
                user_prompt=question,
                system_prompt=system_prompt,
                temperature=0.0,
                use_tools=True,
            ):
                print(token, end="", flush=True)
                response_text += token
            print("\n")

        except Exception as e:
            print(f"Error with question {i}: {e}")
