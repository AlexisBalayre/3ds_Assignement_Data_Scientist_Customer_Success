// ===================================================================
// Neo4j Aura Support Database Setup Script 
// ===================================================================

// ===================================================================
// 0. RESET DATABASE - Delete all existing data
// =================================================================== 
MATCH (n) DETACH DELETE n;

// ===================================================================
// 1. Create Unique Constraints 
// =================================================================== 
CREATE CONSTRAINT user_id_unique FOR (u:User) REQUIRE u.userId IS UNIQUE;
CREATE CONSTRAINT status_id_unique FOR (s:Status) REQUIRE s.statusId IS UNIQUE;
CREATE CONSTRAINT priority_id_unique FOR (p:Priority) REQUIRE p.priorityId IS UNIQUE;
CREATE CONSTRAINT category_id_unique FOR (c:Category) REQUIRE c.categoryId IS UNIQUE;
CREATE CONSTRAINT ticket_id_unique FOR (t:Ticket) REQUIRE t.ticketId IS UNIQUE;
CREATE CONSTRAINT comment_id_unique FOR (c:Comment) REQUIRE c.commentId IS UNIQUE;
CREATE CONSTRAINT article_id_unique FOR (a:DocumentationArticle) REQUIRE a.articleId IS UNIQUE;

// ===================================================================
// 2. Load Nodes 
// =================================================================== 

// Users
LOAD CSV WITH HEADERS FROM 'http://localhost:11001/project-e3be308c-3ce6-43b8-93c8-52f1f8e3707f/users_sample.csv' AS row
WITH row WHERE row.userId IS NOT NULL
CREATE (:User {
  userId: row.userId,
  username: row.username,
  firstName: row.firstName,
  lastName: row.lastName,
  email: row.email,
  role: row.role,
  isActive: CASE WHEN toLower(row.isActive) IN ['true', '1', 'yes'] THEN true ELSE false END
});

// Statuses
LOAD CSV WITH HEADERS FROM 'http://localhost:11001/project-e3be308c-3ce6-43b8-93c8-52f1f8e3707f/status_sample.csv' AS row
WITH row WHERE row.statusId IS NOT NULL
CREATE (:Status {
  statusId: row.statusId,
  name: row.name,
  isClosed: CASE WHEN toLower(row.isClosed) IN ['true', '1', 'yes'] THEN true ELSE false END
});

// Priorities
LOAD CSV WITH HEADERS FROM 'http://localhost:11001/project-e3be308c-3ce6-43b8-93c8-52f1f8e3707f/priority_sample.csv' AS row
WITH row WHERE row.priorityId IS NOT NULL
CREATE (:Priority {
  priorityId: row.priorityId,
  name: row.name,
  level: CASE WHEN row.level IS NOT NULL AND row.level <> '' THEN toInteger(row.level) ELSE NULL END
});

// Categories
LOAD CSV WITH HEADERS FROM 'http://localhost:11001/project-e3be308c-3ce6-43b8-93c8-52f1f8e3707f/category_sample.csv' AS row
WITH row WHERE row.categoryId IS NOT NULL
CREATE (:Category {
  categoryId: row.categoryId,
  name: row.name,
  description: row.description
});

// Tickets
LOAD CSV WITH HEADERS FROM 'http://localhost:11001/project-e3be308c-3ce6-43b8-93c8-52f1f8e3707f/tickets_sample.csv' AS row
WITH row WHERE row.ticketId IS NOT NULL
CREATE (:Ticket {
  ticketId: row.ticketId,
  title: row.title,
  description: row.description,
  creationDate: CASE 
    WHEN row.creationDate IS NOT NULL AND row.creationDate <> '' 
    THEN datetime(replace(row.creationDate, ' ', 'T')) 
    ELSE NULL 
  END,
  resolvedDate: CASE 
    WHEN row.resolvedDate IS NOT NULL AND row.resolvedDate <> '' 
    THEN datetime(replace(row.resolvedDate, ' ', 'T')) 
    ELSE NULL 
  END,
  resolutionSummary: row.resolutionSummary,
  createdByUserId: row.createdByUserId,
  assignedToUserId: row.assignedToUserId,
  statusId: row.statusId,
  priorityId: row.priorityId,
  categoryId: row.categoryId
});

// Comments
LOAD CSV WITH HEADERS FROM 'http://localhost:11001/project-e3be308c-3ce6-43b8-93c8-52f1f8e3707f/comments_sample.csv' AS row
WITH row WHERE row.commentId IS NOT NULL
CREATE (:Comment {
  commentId: row.commentId,
  content: row.content,
  creationDate: CASE 
    WHEN row.creationDate IS NOT NULL AND row.creationDate <> '' 
    THEN datetime(replace(row.creationDate, ' ', 'T')) 
    ELSE NULL 
  END,
  isSolution: CASE WHEN toLower(row.isSolution) IN ['true', '1', 'yes'] THEN true ELSE false END,
  ticketId: row.ticketId,
  authorUserId: row.authorUserId,
  articleId: row.articleId
});

// Documentation Articles
LOAD CSV WITH HEADERS FROM 'http://localhost:11001/project-e3be308c-3ce6-43b8-93c8-52f1f8e3707f/documentation_sample.csv' AS row
WITH row WHERE row.articleId IS NOT NULL
CREATE (:DocumentationArticle {
  articleId: row.articleId,
  title: row.title,
  summary: row.summary,
  referenceUrl: row.referenceUrl,
  categoryId: row.categoryId
});

// ===================================================================
// 3. Create Relationships 
// =================================================================== 

// Tickets created by Users
MATCH (t:Ticket), (u:User)
WHERE t.createdByUserId = u.userId
CREATE (u)-[:SUBMITS]->(t);

// Tickets assigned to Users
MATCH (t:Ticket), (u:User)
WHERE t.assignedToUserId = u.userId AND t.assignedToUserId IS NOT NULL
CREATE (u)-[:ASSIGNED_TO]->(t);

// Tickets have Status
MATCH (t:Ticket), (s:Status)
WHERE t.statusId = s.statusId
CREATE (t)-[:HAS_STATUS]->(s);

// Tickets have Priority
MATCH (t:Ticket), (p:Priority)
WHERE t.priorityId = p.priorityId
CREATE (t)-[:HAS_PRIORITY]->(p);

// Tickets belong to Category
MATCH (t:Ticket), (c:Category)
WHERE t.categoryId = c.categoryId
CREATE (t)-[:HAS_CATEGORY]->(c);

// Comments belong to Tickets
MATCH (c:Comment), (t:Ticket)
WHERE c.ticketId = t.ticketId
CREATE (t)-[:CONTAINS]->(c);

// Comments authored by Users
MATCH (c:Comment), (u:User)
WHERE c.authorUserId = u.userId
CREATE (u)-[:POSTS]->(c);

// Documentation Articles belong to Category
MATCH (a:DocumentationArticle), (c:Category)
WHERE a.categoryId = c.categoryId
CREATE (c)-[:HAS_CATEGORY]->(a);

// Comments reference Documentation Articles
MATCH (c:Comment), (a:DocumentationArticle)
WHERE c.articleId = a.articleId AND c.articleId IS NOT NULL
CREATE (a)-[:REFERENCED_BY]->(c);

// ===================================================================
// 5. Remove temporary foreign key properties  
// ================================================================== 
MATCH (t:Ticket) REMOVE t.createdByUserId, t.assignedToUserId, t.statusId, t.priorityId, t.categoryId;
MATCH (c:Comment) REMOVE c.ticketId, c.authorUserId, c.articleId;
MATCH (a:DocumentationArticle) REMOVE a.categoryId;
