# 3DS Assignement - Data Scientist - Customer Success: AuraHelpeskGraph 🤖

A support chatbot that leverages local LLMs, vector similarity search, and knowledge graphs to provide contextual assistance by finding and presenting solutions from historical support tickets.

## 🌟 Features

- **Local LLM Integration**: Uses Ollama for privacy-preserving AI inference
- **Vector Similarity Search**: Finds semantically similar past tickets using embeddings
- **Knowledge Graph**: Neo4j-powered ticket relationship management
- **Interactive Chat Interface**: Streamlit-based web application
- **Tool Integration**: LLM can intelligently decide when to search for similar tickets

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   LLM Handler    │    │   Ollama API    │
│   Chat UI       │◄──►│   (OpenAI        │◄──►│   (Local LLM)   │
│                 │    │   Compatible)    │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │
         │                       ▼
         │              ┌──────────────────┐    ┌─────────────────┐
         │              │  Knowledge Graph │◄──►│     Neo4j       │
         │              │    Retriever     │    │   Database      │
         │              └──────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌──────────────────┐
│  Conversation   │    │   Embedding      │
│   History       │    │   Computation    │
│   (JSON Files)  │    │                  │
└─────────────────┘    └──────────────────┘
```

## 📋 Prerequisites

### System Requirements

- Python 3.13.2
- Neo4j Database with APOC plugin
- [Ollama](https://ollama.ai/)

### Required Services

1. **Ollama Server**

   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh

   # Pull required models
   ollama pull llama3.2  # or your preferred chat model
   ollama pull nomic-embed-text  # for embeddings
   ```

2. **Neo4j Database**

   Install Neo4j: https://neo4j.com/deployment-center/

   **Important**: Make sure to install the APOC plugin for Neo4j as it's required for embedding operations.

## 🚀 Installation

### Setup

1. **Clone the repository**

   ```bash
   git clone "https://github.com/AlexisBalayre/3ds_Assignement_Data_Scientist_Customer_Success"
   cd 3ds_Assignement_Data_Scientist_Customer_Success
   ```

2. **Install Python dependencies using Poetry**

   ```bash
   # Install Poetry if you haven't already
   curl -sSL https://install.python-poetry.org | python3 -

   # Install project dependencies
   poetry install

   # Activate the virtual environment
   poetry shell
   ```

3. **Create an environment file**

   ```bash
   cp exemple.env .env
   ```

4. **Set up Neo4j schema**

   ```cypher
   // Create vector index for ticket similarity search
   CREATE VECTOR INDEX ticketsTitleDescription FOR (t:Ticket) ON (t.embedding)
   OPTIONS {indexConfig: {
       `vector.dimensions`: 384,
       `vector.similarity_function`: 'cosine'
   }}

   // Create constraints
   CREATE CONSTRAINT ticket_id IF NOT EXISTS FOR (t:Ticket) REQUIRE t.ticketId IS UNIQUE;
   CREATE CONSTRAINT comment_id IF NOT EXISTS FOR (c:Comment) REQUIRE c.commentId IS UNIQUE;
   CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.userId IS UNIQUE;
   ```

### Configuration

```python
# config.py
class Config:
    # Neo4j Configuration
    neo4j_uri = "bolt://localhost:7687"
    neo4j_username = "neo4j"
    neo4j_password = "password"
    neo4j_database = "neo4j"

    # Ollama Configuration
    ollama_base_url = "http://localhost:11434/v1"
    ollama_model = "llama3.2"  # Your chat model
    ollama_embed_model = "nomic-embed-text"  # Your embedding model
    ollama_temperature = 0.0 # Lower for factual responses
    ollama_max_tokens = 1024
```

## 💾 Data Setup

### Neo4j Database Schema Setup

The project includes Cypher scripts to set up the complete database schema and load sample data:

1. **Database Structure Setup** (`datasets/setup_database.cypher`)

   - Creates unique constraints for all entities
   - Loads sample data from CSV files (users, tickets, comments, etc.)
   - Establishes relationships between entities
   - Cleans up temporary foreign key properties

2. **Embeddings Setup** (`datasets/upload_embeddings.cypher`)
   - Loads pre-computed embeddings for tickets
   - Creates vector index for similarity search
   - Configures cosine similarity with proper dimensions

### Setting Up Your Database

1. **Prepare your CSV files** in the Neo4j import directory:

   ```
   neo4j/import/
   ├── users_sample.csv
   ├── status_sample.csv
   ├── priority_sample.csv
   ├── category_sample.csv
   ├── tickets_sample.csv
   ├── comments_sample.csv
   ├── documentation_sample.csv
   └── tickets_sample_with_embeddings.csv
   ```

   > **Note**: The Cypher scripts use `http://localhost:11001/project-<id>/` URLs for CSV import.
   > Update these URLs in the scripts to match your Neo4j import directory or file server setup.

2. **Run the database setup script**:

   ```cypher
   // In Neo4j Browser or cypher-shell
   // Copy and paste the contents of datasets/setup_database.cypher
   ```

3. **Generate embeddings for your tickets**:

   ```bash
   # Process your tickets CSV to add embeddings
   poetry run python compute_embedding.py
   ```

4. **Upload embeddings to Neo4j**:
   ```cypher
   // In Neo4j Browser
   // Copy and paste the contents of datasets/upload_embeddings.cypher
   ```

### Expected Data Schema

The database follows this entity relationship structure:

```
User ──SUBMITS──► Ticket ──HAS_STATUS──► Status
 │                   │
 └──POSTS──► Comment │
             │       ├──HAS_PRIORITY──► Priority
             │       │
         CONTAINS◄── │ ──HAS_CATEGORY──► Category
                     │                      │
                     └──CONTAINS──► Comment │
                                    │       │
            DocumentationArticle ◄──┴───────┘
                     │
                REFERENCED_BY◄──Comment
```

A more detailed architecture diagram can be found in the [docs/1. Analysis/knowledge_graph_architecture_diagram.png](docs/1.%20Analysis/knowledge_graph_architecture_diagram.png) file.

## 🎯 Usage

### Running the Application

```bash
# Using Poetry (recommended)
poetry run streamlit run app.py

# Or activate the virtual environment first
poetry shell
streamlit run app.py

# Or run directly
poetry run python app.py
```

### Using the Chat Interface

1. **Access the application** at [http://localhost:8501](http://localhost:8501)
2. **Select a model** from the sidebar dropdown
3. **Configure parameters** (optional):
   - System prompt
   - Temperature (creativity level)
   - Number of similar tickets to retrieve
   - Similarity threshold
   - Context comments to include
4. **Start chatting** with questions about technical issues
5. **View sources** when the AI finds similar tickets

### Example Interactions

**User**: "I get a 500 error when I try to open Aura. What should I do?"

**AI**: _[Searches for similar tickets and finds relevant solutions]_
"Based on similar tickets, this is typically caused by authentication service issues. Here's how to resolve it:

1. Check if the authentication service is running
2. Restart the auth service if needed
3. Clear browser cache and cookies
4. Try logging in again"

_[Shows source tickets with similarity scores]_

## 📁 Project Structure

```
3ds_Assignement_Data_Scientist_Customer_Success/
├── main.py                     # Application entry point
├── chatbot.py                  # Streamlit chat interface
├── llm_handler.py             # LLM operations and tool integration
├── knowledge_graph_retriever.py # Neo4j similarity search
├── compute_embedding.py       # Text embedding generation
├── config.py                  # Configuration settings
├── pyproject.toml             # Poetry dependencies and project config
├── poetry.lock               # Poetry lock file for reproducible builds
├── datasets/
│   ├── data/
│   │   ├── tickets_sample.csv
│   │   ├── tickets_sample_with_embeddings.csv
│   │   ├── users_sample.csv
│   │   ├── comments_sample.csv
│   │   ├── status_sample.csv
│   │   ├── priority_sample.csv
│   │   ├── category_sample.csv
│   │   └── documentation_sample.csv
│   ├── setup_database.cypher   # Neo4j database schema and data setup
│   └── upload_embeddings.cypher # Vector embeddings and index setup
├── llm_history/              # Conversation storage
│   └── *.json               # Individual conversation files
└── README.md                 # This file
```

## ⚙️ Configuration

### Model Configuration

- **Chat Models**: Any Ollama-compatible model (llama2, mistral, codellama, etc.)
- **Embedding Models**: Models that produce 768-dimensional vectors (e.g., sentence-transformers)
- **Vector Dimensions**: Must match your embedding model (project configured for 768 dimensions)
- **Neo4j Requirements**: APOC procedures must be installed for embedding handling

### Performance Tuning

- **top_k**: Number of similar tickets to retrieve (1-20)
- **context_comments**: Comments to include for context (0-10)
- **min_similarity_score**: Similarity threshold (0.7-0.95)
- **temperature**: Response creativity (0.0-1.0)

## 🔧 Advanced Features

### Tool Usage

The system intelligently decides when to search for similar tickets:

- **Searches when**: Technical issues, error reports, how-to questions
- **Doesn't search for**: Greetings, general questions, casual conversation

## 📚 Resources & References

This project was built using knowledge and inspiration from:

- **[Knowledge Graphs + LLMs: Multi-Hop Question Answering](https://neo4j.com/blog/developer/knowledge-graphs-llms-multi-hop-question-answering/)** - Neo4j blog on integrating knowledge graphs with LLMs
- **[LLM Fundamentals: Vectors & Semantic Search](https://graphacademy.neo4j.com/courses/llm-fundamentals/2-vectors-semantic-search/2-vector-index/)** - GraphAcademy course on vector indexes and semantic search with neo4j
- **[Structured Outputs With LLM](https://platform.openai.com/docs/guides/structured-outputs?example=structured-data)** - OpenAI guide on structured outputs for LLMs
- **[Function Calling](https://platform.openai.com/docs/guides/function-calling?api-mode=responses)** - OpenAI documentation on function calling with LLMs

### Additional Learning Resources

- [Neo4j Vector Index Documentation](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/)
- [Ollama Documentation](https://ollama.ai/docs)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Streamlit Documentation](https://docs.streamlit.io/)
