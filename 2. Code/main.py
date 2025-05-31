#!/usr/bin/env python3
"""
Enhanced LlamaIndex Features for Knowledge Graph Query System
Advanced capabilities including RAG, query planning, and evaluation
"""

import logging

from llama_index.core.schema import QueryBundle

from config import Config
from query_engine import TicketSupportQueryEngine


logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating enhanced features"""
    config = Config()

    # Initialize enhanced system
    query_engine = TicketSupportQueryEngine(config)

    # Example queries
    sample_questions = [
        "Show me high priority tickets with their assignees",
        "What knowledge articles are most relevant to authentication issues?",
        "Which customers have the most open tickets?",
        "Analyze the resolution patterns for critical tickets",
    ]

    print("🚀 Enhanced LlamaIndex Knowledge Graph System")
    print("=" * 50)
    print(f"Features enabled:")
    print(f"  - RAG: {config.enable_rag}")
    print(f"  - Query Planning: {config.enable_query_planning}")
    print(f"  - Evaluation: {config.enable_evaluation}")
    print()

    for question in sample_questions:
        print(f"❓ Question: {question}")

        try:
            query_bundle = QueryBundle(query_str=question)
            response = query_engine.query(query_bundle)
            print(f"💡 Answer: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")

        print("-" * 50)

    query_engine.close()


if __name__ == "__main__":
    main()
