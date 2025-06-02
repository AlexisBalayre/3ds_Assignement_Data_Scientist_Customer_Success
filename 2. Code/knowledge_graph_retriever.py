import datetime
import logging
from typing import List, Dict, Any, Optional

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from neo4j import GraphDatabase
import neo4j

from config import Config
from compute_embedding import compute_embedding

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

config = Config()


class KnowledgeGraphRetriever(BaseRetriever):
    """
    Retriever that finds solution comments from closed tickets based on
    vector similarity search of ticket title + description.
    """

    def __init__(self, top_k: int = 5, context_comments: int = 3):
        """
        Initialize the KnowledgeGraphRetriever.

        Args:
            top_k (int): Number of similar tickets to retrieve (default: 5).
            context_comments (int): Number of previous comments to include in results (default: 3).
        """
        super().__init__()

        self.neo4j_driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_username, config.neo4j_password),
            database=config.neo4j_database,
        )
        self.top_k = top_k
        self.context_comments = context_comments

    def _retrieve(self, query_bundle, **kwargs) -> List[NodeWithScore]:
        """
        Main retrieve method required by BaseRetriever.
        Converts query to embedding and calls _retrieve_by_vector.
        """
        # Extract query text and convert to embedding
        query_text = query_bundle.query_str
        query_vector = compute_embedding(query_text)

        # Get raw results
        raw_results = self.retrieve_solution_comments_by_vector(
            query_vector=query_vector, **kwargs
        )

        # Convert to NodeWithScore format
        nodes_with_scores = []
        for result in raw_results:
            text_content = self._format_result_as_text(result)
            text_node = TextNode(
                text=text_content,
                metadata={
                    "ticket_id": result["ticketId"],
                    "similarity_score": result["similarityScore"],
                    "solution_comment_id": result["solutionComment"]["commentId"],
                    "context_comment_count": len(result["contextComments"]),
                },
            )
            node_with_score = NodeWithScore(
                node=text_node, score=result["similarityScore"]
            )
            nodes_with_scores.append(node_with_score)

        return nodes_with_scores

    def retrieve_solution_comments_by_vector(
        self,
        query_vector: List[float],
        top_k: Optional[int] = None,
        context_comments: Optional[int] = None,
        min_similarity_score: float = 0.9,
    ) -> List[Dict[str, Any]]:
        """
        Find solution comments from closed tickets based on vector similarity.

        Parameters:
        - query_vector: The embedding vector for similarity search
        - top_k: Number of similar tickets to search (default: instance default)
        - context_comments: Number of previous comments to include (default: instance default)
        - min_similarity_score: Minimum similarity threshold

        Returns:
        - List of dictionaries containing ticket info, solution comment, and context
        """
        top_k = top_k or self.top_k
        context_comments = context_comments or self.context_comments

        cypher_query = """
        // Step 1: Find top_k closed tickets with highest similarity to query vector
        CALL db.index.vector.queryNodes(
            'ticketsTitleDescription',
            $top_k,
            $query_vector
        ) YIELD node AS ticket, score

        // Step 2: Keep only closed tickets with similarity above threshold
        MATCH (ticket:Ticket)-[:HAS_STATUS]->(status:Status)
        WHERE status.isClosed = true
        AND score >= $min_similarity_score

        // Step 2: Find the solution comment
        MATCH (ticket)-[:CONTAINS]->(solutionComment:Comment)
        WHERE solutionComment.isSolution = true

        // Step 3: Find solution author
        MATCH (solutionAuthor:User)-[:POSTS]->(solutionComment)

        // Step 4: Find and order context comments before solution
        OPTIONAL MATCH (ticket)-[:CONTAINS]->(contextComment:Comment)
        WHERE contextComment.creationDate < solutionComment.creationDate
        WITH ticket, score, solutionComment, solutionAuthor, contextComment
        ORDER BY contextComment.creationDate DESC

        // Step 5: Collect ordered comments and limit to N most recent
        WITH ticket, score, solutionComment, solutionAuthor,
            collect(contextComment)[0..$context_comments-1] AS recentContextComments
        UNWIND CASE 
            WHEN size(recentContextComments) > 0 
            THEN recentContextComments 
            ELSE [null] 
        END AS contextComment

        // Step 6: Get authors for context comments
        OPTIONAL MATCH (contextAuthor:User)-[:POSTS]->(contextComment)
        WHERE contextComment IS NOT NULL
        WITH ticket, score, solutionComment, solutionAuthor,
            collect(CASE 
                WHEN contextComment IS NOT NULL 
                THEN {
                    comment: contextComment,
                    author: contextAuthor
                } 
                ELSE null 
            END) AS contextData

        // Return complete ticket and solution data
        RETURN {
            ticketId: ticket.ticketId,
            title: ticket.title,
            description: ticket.description,
            resolutionSummary: ticket.resolutionSummary,
            solutionComment: {
                commentId: solutionComment.commentId,
                content: solutionComment.content,
                creationDate: solutionComment.creationDate,
                author: {
                    role: solutionAuthor.role
                }
            },
            contextComments: [
                item IN contextData
                WHERE item IS NOT NULL AND item.comment IS NOT NULL |
                {
                    commentId: item.comment.commentId,
                    content: item.comment.content,
                    creationDate: item.comment.creationDate,
                    author: CASE
                        WHEN item.author IS NOT NULL
                        THEN {
                            role: item.author.role
                        }
                        ELSE null
                    END
                }
            ],
            similarityScore: score
        } AS result
        ORDER BY result.similarityScore DESC;
        """

        def _convert_neo4j_datetimes(obj: Any) -> Any:
            """
            Recursively walk through obj (which may be a dict, list, or scalar).
            Whenever we see a neo4j.time.DateTime, convert it to an ISO string.
            """
            # 1) If it’s exactly a neo4j.time.DateTime, convert via to_native().isoformat()
            if isinstance(obj, neo4j.time.DateTime):
                # to_native() returns a built-in datetime.datetime with tzinfo
                return obj.to_native().isoformat()

            # 2) If it’s a dict, recurse on all values
            if isinstance(obj, dict):
                return {key: _convert_neo4j_datetimes(val) for key, val in obj.items()}

            # 3) If it’s a list (or tuple), recurse on each element
            if isinstance(obj, list):
                return [_convert_neo4j_datetimes(elem) for elem in obj]

            # 4) Otherwise, leave as-is (e.g. float, str, int)
            return obj


        with self.neo4j_driver.session() as session:
            try:
                records = session.run(
                    cypher_query,
                    query_vector=query_vector,
                    top_k=top_k,
                    context_comments=context_comments,
                    min_similarity_score=min_similarity_score,
                )
                raw_results = [record["result"] for record in records]

                # Post‐process each result to turn datetime objects into strings:
                cleaned_results = [_convert_neo4j_datetimes(r) for r in raw_results]
                return cleaned_results

            except Exception as e:
                logger.error(f"Error executing Cypher query: {e}")
                return []

    def _format_result_as_text(self, result: Dict[str, Any]) -> str:
        """
        Helper to format the structured result as text for Document creation.

        Args:
            result (Dict[str, Any]): The structured result from the Neo4j query.

        Returns:
            str: Formatted text representation of the result.
        """
        text_parts = []

        # Ticket information
        text_parts.append(f"TICKET: {result['title'] or 'No title'}")
        text_parts.append(f"Ticket ID: {result['ticketId']}")
        text_parts.append(f"Description: {result['description'] or 'No description'}")
        if result.get("resolutionSummary"):
            text_parts.append(f"Resolution: {result['resolutionSummary']}")

        # Context comments
        if result["contextComments"]:
            text_parts.append("\nCONTEXT COMMENTS:")
            for comment in result["contextComments"]:
                author = comment.get("author")
                author_name = author["role"] if author else "Unknown"
                text_parts.append(f"- {author_name}: {comment['content']}")

        # Solution comment
        solution = result["solutionComment"]
        solution_author = solution["author"]
        solution_author_name = solution_author["role"] if solution_author else "Unknown"
        text_parts.append(f"\nSOLUTION by {solution_author_name}:")
        text_parts.append(solution["content"])

        text_parts.append(f"\nSimilarity Score: {result['similarityScore']:.3f}")

        return "\n".join(text_parts)


# ---------------------------------------------
# UNIT TESTING (DEMONSTRATION PURPOSES ONLY)
# ---------------------------------------------

# if __name__ == "__main__":

#     retriever = KnowledgeGraphRetriever(top_k=5, context_comments=3)

#     # Example query vector (this should be generated using an embedding model)
#     title = "Cannot login to Aura"
#     description = "I'm getting a 404 error message when trying to log into Aura with my 3DS credentials"
#     query_vector = compute_embedding(title + " " + description)
#     results = retriever.retrieve_solution_comments_by_vector(query_vector)

#     # Process results
#     for result in results:
#         print(f"Ticket: {result['title']}")
#         print(f"Similarity: {result['similarityScore']:.3f}")
#         print(f"Solution: {result['solutionComment']['content'][:100]}...")
#         print(f"Context comments: {len(result['contextComments'])}")
#         for comment in result['contextComments']:
#             print(f"- {comment['content'][:50]}... ({comment['author']['role']})")
#         print("---")

#     # Close the driver when done
#     driver.close()
