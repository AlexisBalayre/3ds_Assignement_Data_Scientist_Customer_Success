import os
import json
import datetime

import streamlit as st

from config import Config
from llm_handler import LLMHandler


config = Config()


class Chat:
    """
    A Chat class to handle interactions with various language models via a Streamlit interface.
    This class facilitates selecting models, initiating conversations, displaying conversation history,
    and managing input/output of chat messages.
    """

    def __init__(self, output_dir="llm_history", health_check_enabled=False):
        """
        Initializes the Chat class, setting up model connections and configuring the chat interface.

        Args:
            output_dir (str): The directory to save conversation history files.
        """
        self.st = st  # Streamlit instance for UI rendering
        self.llmHandler = LLMHandler(
            st=self.st
        )  # LLM handler for managing language models
        self.LOCAL_MODELS = self.llmHandler.get_available_models()  # Load local models
        self.TOOLS = self.llmHandler.get_available_tools()  # Load available tools
        self.OUTPUT_DIR = output_dir  # Directory for saving conversations
        self.__configure_chat()  # Configure the chat UI

    def __configure_chat(self):
        """
        Configures the Streamlit interface for the chat application, including page layout and model selection.
        """
        # Set Streamlit page config
        self.st.set_page_config(
            layout="wide", page_title="AuraHelpeskGraph", page_icon="🚀"
        )
        # Add sidebar title
        self.st.sidebar.title("AuraHelpeskGraph 🤖")
        # Select the model for conversation
        self.selected_model = self.select_model()
        # Display previous conversations
        self.display_conversation_history()
        # Provide option for new conversation
        self.new_conversation()
        # Set chat parameters
        self.params = self.chat_params()  # Chat parameters

    def run(self):
        """
        Runs the chat interface allowing for user input and displays responses from the selected model.
        """
        # Input box for user's questions
        prompt = self.st.chat_input(f"Ask {self.selected_model} a question ...")
        # Process and display the conversation
        self.chat(prompt)

    def new_conversation(self):
        """
        Initiates a new conversation, generating a unique identifier and resetting chat history.
        """
        # Button to start a new conversation
        new_conversation = self.st.sidebar.button(
            "New conversation", key="new_conversation"
        )
        if new_conversation:
            # Generate unique conversation ID
            self.st.session_state["conversation_id"] = str(datetime.datetime.now())
            # Reset chat history for the new conversation
            self.st.session_state[
                "chat_history_" + self.st.session_state["conversation_id"]
            ] = []
            # Prepare file name for saving the conversation
            file_name = f"{self.st.session_state['conversation_id']}.json"
            # Initialize the conversation file with empty content
            json.dump([], open(os.path.join(self.OUTPUT_DIR, file_name), "w"))
            # Rerun the app to reflect changes
            self.st.rerun()

    def select_model(self):
        """
        Allows the user to select a language model for the conversation, providing details for both local and online models.

        Returns:
            A list containing the selected model's name.
        """
        # List of local models
        model_names = [
            model["name"] for model in self.LOCAL_MODELS if self.LOCAL_MODELS != []
        ]

        # Sidebar selection for model
        self.st.sidebar.subheader("Models")
        llm_name = self.st.sidebar.selectbox(
            f"Select a model ({len(model_names)} available)", model_names
        )

        # Check if the selected model is local or online and extract its details accordingly
        if llm_name:
            llm_details = [
                model for model in self.LOCAL_MODELS if model["name"] == llm_name
            ][0]
            if type(llm_details["size"]) != str:
                llm_details["size"] = f"{round(llm_details['size'] / 1e9, 2)} GB"

            # Display model details for the user's reference
            with self.st.expander("Model Details"):
                self.st.write(llm_details)

            return llm_name

        # Return the default model if no model is selected
        return self.LOCAL_MODELS[0]["name"]

    def display_conversation_history(self):
        """
        Displays the conversation history for the selected model, allowing users to review past interactions.
        """
        # Define the directory where conversation history files are stored
        OUTPUT_DIR = os.path.join(os.getcwd(), self.OUTPUT_DIR)

        # Ensure the output directory exists, create it if not
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, exist_ok=True)

        # List all JSON files in the output directory which are considered as conversation history files
        conversation_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".json")]

        # Sort the conversation files by modification time in descending order
        conversation_files = sorted(
            conversation_files,
            key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)),
            reverse=True,
        )

        # Insert an option at the start of the list for UI purposes, possibly to serve as a 'select' prompt
        conversation_files.insert(0, "")

        def format_id(id):
            date = id.split(".")[0]
            return f"{date}"

        # Add a section in the sidebar for displaying conversation history
        self.st.sidebar.subheader("Conversation History")
        # Allow the user to select a conversation history file from a dropdown list in the sidebar
        selected_conversation = self.st.sidebar.selectbox(
            "Select a conversation", conversation_files, index=0, format_func=format_id
        )

        # Check if a conversation file was selected (not the blank option inserted earlier)
        if selected_conversation:
            # Construct the full path to the selected conversation file
            conversation_file = os.path.join(OUTPUT_DIR, selected_conversation)

            # Display the last modified time of the selected conversation file
            last_modified = datetime.datetime.fromtimestamp(
                os.path.getmtime(conversation_file)
            ).strftime("%Y-%m-%d %H:%M:%S")
            self.st.sidebar.write(f"Last update: {last_modified}")

            # Open and load the conversation JSON data
            with open(conversation_file, "r") as f:
                conversation_data = json.load(f)

            # Extract the conversation ID from the selected filename for state tracking
            self.st.session_state["conversation_id"] = selected_conversation.split(".")[
                0
            ]

            # Load the conversation data into the session state for display
            self.st.session_state[
                "chat_history_" + self.st.session_state["conversation_id"]
            ] = conversation_data

            self.st.session_state["chat_params"] = {
                "system_prompt": "You are a support agent for Dassault Systèmes employees. Use internal ticket insights (never quote them) to provide clear, concise, friendly guidance. If a request is unclear, ask a brief clarifying question; restate the issue before offering help. Give step-by-step solutions in simple language (use numbered/bulleted lists) and highlight menu paths or commands in quotes. If you can't solve it, direct users to escalate (file a ticket or contact the right team). Never ask for passwords or sensitive data. Always answer in English.",
                "temperature": 0.0,
                "top_k": 2,
                "context_comments": 1,
                "history_length": 2,
                "similarity_threshold": 0.80,
                "max_tokens": 8000,
            }

            # Load the system prompt from the conversation history
            for message in conversation_data:
                if message["role"] == "system":
                    system_prompt = message["content"]
                    self.st.session_state["chat_params"] = {
                        "system_prompt": system_prompt,
                        "temperature": 0.0,
                        "top_k": 2,
                        "context_comments": 1,
                        "history_length": 2,
                        "similarity_threshold": 0.80,
                        "max_tokens": 8000,
                    }

    def chat_params(self):
        """
        Displays chat parameters in the sidebar for the user to adjust the language model's behavior.

        Returns:
            A dictionary containing the chat parameters set by the user.
        """
        self.st.sidebar.subheader("Chat Parameters")

        # Load the chat parameters from the session state if available
        if "chat_params" in self.st.session_state:
            chat_params = self.st.session_state["chat_params"]
        else:
            # Set default chat parameters
            chat_params = {
                "system_prompt": "You are a support agent for Dassault Systèmes employees. Use internal ticket insights (never quote them) to provide clear, concise, friendly guidance. If a request is unclear, ask a brief clarifying question; restate the issue before offering help. Give step-by-step solutions in simple language (use numbered/bulleted lists) and highlight menu paths or commands in quotes. If you can't solve it, direct users to escalate (file a ticket or contact the right team). Never ask for passwords or sensitive data. Always answer in English.",
                "temperature": 0.0,
                "top_k": 2,
                "context_comments": 1,
                "history_length": 2,
                "similarity_threshold": 0.80,
                "max_tokens": 8000,
            }

        # System prompt
        system_prompt = self.st.sidebar.text_area(
            key="system_prompt",
            label="System Prompt",
            value=chat_params["system_prompt"],
        )

        # Temperature
        temperature = self.st.sidebar.number_input(
            key="temperature",
            label="Temperature",
            min_value=0.0,
            max_value=1.0,
            value=chat_params["temperature"],
        )

        # Number of Similar Tickets to Retrieve
        top_k = self.st.sidebar.number_input(
            key="top_k",
            label="Number of Similar Tickets to Retrieve",
            min_value=1,
            value=chat_params["top_k"],
        )

        # Number of context comments to include.
        context_comments = self.st.sidebar.number_input(
            key="context_comments",
            label="Number of Context Comments to Include",
            min_value=1,
            value=chat_params["context_comments"],
        )

        # Tickets Similarity Threshold
        similarity_threshold = self.st.sidebar.number_input(
            key="similarity_threshold",
            label="Tickets Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=chat_params["similarity_threshold"],
        )

        # Update LLM Handler with the new parameters
        self.llmHandler.update_settings(
            temperature=temperature,
            max_tokens=chat_params["max_tokens"],
            top_k=top_k,
            context_comments=context_comments,
            model_name=self.selected_model,
            min_similarity_score=similarity_threshold,
        )

        # return chat parameters
        return {
            "system_prompt": system_prompt,
            "temperature": temperature,
            "top_k": top_k,
            "context_comments": context_comments,
            "history_length": chat_params["history_length"],
            "similarity_threshold": similarity_threshold,
            "max_tokens": chat_params["max_tokens"],
            "conversation_language": chat_params.get("conversation_language", "None"),
        }

    def chat(self, prompt):
        """
        Handles sending a prompt to the selected language model and displaying the response in the chat interface.

        Args:
            prompt (str): The user's question or prompt for the language model.

        Returns:
            The response from the language model to the provided prompt.
        """
        # Check if there's an ongoing conversation, otherwise initialize
        if "conversation_id" in self.st.session_state:
            # Use the existing conversation ID to track chat history
            chat_history_key = (
                f"chat_history_{self.st.session_state['conversation_id']}"
            )
        else:
            # If no conversation is active, generate a new ID and initialize chat history
            self.st.session_state["conversation_id"] = str(datetime.datetime.now())
            chat_history_key = (
                f"chat_history_{self.st.session_state['conversation_id']}"
            )
            self.st.session_state[chat_history_key] = []

        # Iterate through the stored chat history and display it
        for message in self.st.session_state[chat_history_key]:
            role = message["role"]
            if role == "user":
                with self.st.chat_message("user"):
                    self.st.markdown(f"{message["content"]}", unsafe_allow_html=True)

            elif role == "assistant":
                with self.st.chat_message("assistant"):
                    self.st.markdown(message["content"], unsafe_allow_html=True)

            elif role == "function_call":
                # Display function call information (optional)
                with self.st.chat_message("assistant"):
                    if message.get("content"):
                        self.st.markdown(message["content"], unsafe_allow_html=True)
                    # Optionally show tool usage indicator
                    if message.get("tool_calls"):
                        self.st.caption("🔧 Using tools...")

        # Check if there is a new prompt from the user
        if prompt:
            with self.st.chat_message("user"):
                self.st.markdown(f"{prompt}", unsafe_allow_html=True)

            messages = []
            for message in self.st.session_state[chat_history_key]:
                msg_dict = {
                    "content": message.get("content", ""),
                    "role": message["role"],
                }

                # Preserve tool-related fields for function calling
                if "tool_calls" in message:
                    msg_dict["tool_calls"] = message["tool_calls"]
                if "tool_call_id" in message:
                    msg_dict["tool_call_id"] = message["tool_call_id"]

                messages.append(msg_dict)

            # Fetch the response from the language model using the connector
            with self.st.chat_message("assistant"):
                chat_box = self.st.empty()  # Placeholder for the model's response
                params = self.params

                # Use the LLM connector to stream the model's response based on the chat history``
                response_message = chat_box.write_stream(
                    self.llmHandler.llm_stream(
                        model_name=self.selected_model,
                        messages_history=messages,
                        user_prompt=prompt,
                        system_prompt=params["system_prompt"],
                        temperature=params["temperature"],
                        history_length=params["history_length"],
                        max_tokens=params["max_tokens"],
                        use_tools=True,
                    )
                )

            # Modify or add a system prompt to chat history
            if self.params["system_prompt"]:
                # Check if a system prompt is already present in the chat history
                system_prompt_exists = any(
                    message["role"] == "system" for message in messages
                )
                # If a system prompt is not present, add it to the chat history
                if not system_prompt_exists:
                    self.st.session_state[chat_history_key].append(
                        {"content": params["system_prompt"], "role": "system"}
                    )
                # If a system prompt is present, update it in the chat history
                else:
                    for message in messages:
                        if message["role"] == "system":
                            message["content"] = params["system_prompt"]

            # Add the new prompt to the chat history
            self.st.session_state[chat_history_key].append(
                {"content": prompt, "role": "user"}
            )
            # Append the model's response to the chat history
            self.st.session_state[chat_history_key].append(
                {"content": f"{response_message}", "role": "assistant"}
            )
            # Save the conversation to a JSON file
            self.save_conversation()
            # Return the response message to be displayed in the chat UI
            return response_message

    def save_conversation(self):
        """
        Saves the current conversation to a JSON file, allowing for persistence of chat history.
        """
        conversation_id = self.st.session_state["conversation_id"]
        conversation_key = f"chat_history_{conversation_id}"
        conversation_chat = self.st.session_state[conversation_key]
        filename = f"{conversation_id}.json"

        # Check if there's any conversation to save
        if conversation_chat:
            # Prepare the file path for saving the conversation
            conversation_file = os.path.join(self.OUTPUT_DIR, filename)

            # Save the updated conversation back to the file
            with open(conversation_file, "w") as f:
                json.dump(conversation_chat, f, indent=4)
                self.st.success(f"Conversation saved to {conversation_file}")
