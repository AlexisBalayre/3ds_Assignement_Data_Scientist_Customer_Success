// ===================================================================
// Neo4j Aura Support Database - Upload Embeddings for Tickets
// ===================================================================

// Load ticket embeddings from CSV and set the titleDescriptionEmbedding property
LOAD CSV WITH HEADERS
FROM 'http://localhost:11001/project-e3be308c-3ce6-43b8-93c8-52f1f8e3707f/tickets_sample_with_embeddings.csv'
AS row
MATCH (t:Ticket {ticketId: row.ticketId})
CALL db.create.setNodeVectorProperty(t, 'titleDescriptionEmbedding', apoc.convert.fromJsonList(row.embedding))
RETURN count(*);


// Create a vector index for the titleDescriptionEmbedding property
CREATE VECTOR INDEX ticketsTitleDescription IF NOT EXISTS
FOR (t:Ticket)
ON t.titleDescriptionEmbedding
OPTIONS {indexConfig: {
 `vector.dimensions`: 768,
 `vector.similarity_function`: 'cosine'
}};