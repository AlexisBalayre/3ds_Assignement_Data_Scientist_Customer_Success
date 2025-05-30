// ===================================================================
// Neo4j Helpdesk Database Setup Script 
// ===================================================================

// 1. Create Unique Constraints 
CREATE CONSTRAINT org_id_unique FOR (o:Organization) REQUIRE o.orgId IS UNIQUE;
CREATE CONSTRAINT user_id_unique FOR (u:User) REQUIRE u.userId IS UNIQUE;
CREATE CONSTRAINT status_id_unique FOR (s:Status) REQUIRE s.statusId IS UNIQUE;
CREATE CONSTRAINT priority_id_unique FOR (p:Priority) REQUIRE p.priorityId IS UNIQUE;
CREATE CONSTRAINT category_id_unique FOR (c:Category) REQUIRE c.categoryId IS UNIQUE;
CREATE CONSTRAINT ticket_id_unique FOR (t:Ticket) REQUIRE t.ticketId IS UNIQUE;
CREATE CONSTRAINT comment_id_unique FOR (c:Comment) REQUIRE c.commentId IS UNIQUE;
CREATE CONSTRAINT article_id_unique FOR (a:KnowledgeArticle) REQUIRE a.articleId IS UNIQUE;
CREATE CONSTRAINT attachment_id_unique FOR (f:Attachment) REQUIRE f.attachmentId IS UNIQUE;

// 2. Load Nodes 

// Organizations
LOAD CSV WITH HEADERS FROM 'http://localhost:11001/project-e3be308c-3ce6-43b8-93c8-52f1f8e3707f/organizations.csv' AS row
WITH row WHERE row.orgId IS NOT NULL
CREATE (:Organization {
  orgId: row.orgId,
  name: row.name,
  creationDate: CASE 
    WHEN row.creationDate IS NOT NULL AND row.creationDate <> '' 
    THEN datetime(row.creationDate) 
    ELSE NULL 
  END
});

// Users
LOAD CSV WITH HEADERS FROM 'http://localhost:11001/project-e3be308c-3ce6-43b8-93c8-52f1f8e3707f/users.csv' AS row
WITH row WHERE row.userId IS NOT NULL
CREATE (:User {
  userId: row.userId,
  username: row.username,
  firstName: row.firstName,
  lastName: row.lastName,
  email: row.email,
  telephone: row.telephone,
  role: row.role,
  company: row.company,
  accountCreationDate: CASE 
    WHEN row.accountCreationDate IS NOT NULL AND row.accountCreationDate <> '' 
    THEN datetime(row.accountCreationDate) 
    ELSE NULL 
  END,
  lastLoginDate: CASE 
    WHEN row.lastLoginDate IS NOT NULL AND row.lastLoginDate <> '' 
    THEN datetime(row.lastLoginDate) 
    ELSE NULL 
  END,
  isActive: CASE WHEN toLower(row.isActive) IN ['true', '1', 'yes'] THEN true ELSE false END,
  timezone: row.timezone,
  orgId: row.orgId
});

// Statuses
LOAD CSV WITH HEADERS FROM 'http://localhost:11001/project-e3be308c-3ce6-43b8-93c8-52f1f8e3707f/statuses.csv' AS row
WITH row WHERE row.statusId IS NOT NULL
CREATE (:Status {
  statusId: row.statusId,
  name: row.name,
  description: row.description,
  isActive: CASE WHEN toLower(row.isActive) IN ['true', '1', 'yes'] THEN true ELSE false END
});

// Priorities
LOAD CSV WITH HEADERS FROM 'http://localhost:11001/project-e3be308c-3ce6-43b8-93c8-52f1f8e3707f/priorities.csv' AS row
WITH row WHERE row.priorityId IS NOT NULL
CREATE (:Priority {
  priorityId: row.priorityId,
  name: row.name,
  level: CASE WHEN row.level IS NOT NULL AND row.level <> '' THEN toInteger(row.level) ELSE NULL END,
  slaHours: CASE WHEN row.slaHours IS NOT NULL AND row.slaHours <> '' THEN toInteger(row.slaHours) ELSE NULL END
});

// Categories
LOAD CSV WITH HEADERS FROM 'http://localhost:11001/project-e3be308c-3ce6-43b8-93c8-52f1f8e3707f/categories.csv' AS row
WITH row WHERE row.categoryId IS NOT NULL
CREATE (:Category {
  categoryId: row.categoryId,
  name: row.name,
  description: row.description,
  autoAssignmentRules: row.autoAssignmentRules
});

// Tickets
LOAD CSV WITH HEADERS FROM 'http://localhost:11001/project-e3be308c-3ce6-43b8-93c8-52f1f8e3707f/tickets.csv' AS row
WITH row WHERE row.ticketId IS NOT NULL
CREATE (:Ticket {
  ticketId: row.ticketId,
  title: row.title,
  description: row.description,
  creationDate: CASE 
    WHEN row.creationDate IS NOT NULL AND row.creationDate <> '' 
    THEN datetime(row.creationDate) 
    ELSE NULL 
  END,
  lastUpdateDate: CASE 
    WHEN row.lastUpdateDate IS NOT NULL AND row.lastUpdateDate <> '' 
    THEN datetime(row.lastUpdateDate) 
    ELSE NULL 
  END,
  dueDate: CASE 
    WHEN row.dueDate IS NOT NULL AND row.dueDate <> '' 
    THEN datetime(row.dueDate) 
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
LOAD CSV WITH HEADERS FROM 'http://localhost:11001/project-e3be308c-3ce6-43b8-93c8-52f1f8e3707f/comments.csv' AS row
WITH row WHERE row.commentId IS NOT NULL
CREATE (:Comment {
  commentId: row.commentId,
  content: row.content,
  creationDate: CASE 
    WHEN row.creationDate IS NOT NULL AND row.creationDate <> '' 
    THEN datetime(row.creationDate) 
    ELSE NULL 
  END,
  isInternal: CASE WHEN toLower(row.isInternal) IN ['true', '1', 'yes'] THEN true ELSE false END,
  isSolution: CASE WHEN toLower(row.isSolution) IN ['true', '1', 'yes'] THEN true ELSE false END,
  ticketId: row.ticketId,
  authorUserId: row.authorUserId
});

// Knowledge Articles
LOAD CSV WITH HEADERS FROM 'http://localhost:11001/project-e3be308c-3ce6-43b8-93c8-52f1f8e3707f/knowledge_articles.csv' AS row
WITH row WHERE row.articleId IS NOT NULL
CREATE (:KnowledgeArticle {
  articleId: row.articleId,
  title: row.title,
  content: row.content,
  createdDate: CASE 
    WHEN row.createdDate IS NOT NULL AND row.createdDate <> '' 
    THEN datetime(row.createdDate) 
    ELSE NULL 
  END,
  updatedDate: CASE 
    WHEN row.updatedDate IS NOT NULL AND row.updatedDate <> '' 
    THEN datetime(row.updatedDate) 
    ELSE NULL 
  END,
  viewCount: CASE WHEN row.viewCount IS NOT NULL AND row.viewCount <> '' THEN toInteger(row.viewCount) ELSE 0 END,
  isPublished: CASE WHEN toLower(row.isPublished) IN ['true', '1', 'yes'] THEN true ELSE false END,
  authorUserId: row.authorUserId,
  categoryId: row.categoryId
});

// Attachments
LOAD CSV WITH HEADERS FROM 'http://localhost:11001/project-e3be308c-3ce6-43b8-93c8-52f1f8e3707f/attachments.csv' AS row
WITH row WHERE row.attachmentId IS NOT NULL
CREATE (:Attachment {
  attachmentId: row.attachmentId,
  filename: row.filename,
  fileSize: CASE WHEN row.fileSize IS NOT NULL AND row.fileSize <> '' THEN toInteger(row.fileSize) ELSE NULL END,
  fileType: row.fileType,
  uploadDate: CASE 
    WHEN row.uploadDate IS NOT NULL AND row.uploadDate <> '' 
    THEN datetime(row.uploadDate) 
    ELSE NULL 
  END,
  storagePath: row.storagePath,
  ticketId: row.ticketId
});

// 3. Create Relationships 

// Users belong to Organizations
MATCH (u:User), (o:Organization)
WHERE u.orgId = o.orgId
CREATE (u)-[:BELONGS_TO]->(o);

// Tickets created by Users
MATCH (t:Ticket), (u:User)
WHERE t.createdByUserId = u.userId
CREATE (u)-[:CREATED]->(t);

// Tickets assigned to Users
MATCH (t:Ticket), (u:User)
WHERE t.assignedToUserId = u.userId AND t.assignedToUserId IS NOT NULL
CREATE (t)-[:ASSIGNED_TO]->(u);

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
CREATE (t)-[:BELONGS_TO_CATEGORY]->(c);

// Comments belong to Tickets
MATCH (c:Comment), (t:Ticket)
WHERE c.ticketId = t.ticketId
CREATE (c)-[:BELONGS_TO]->(t);

// Comments authored by Users
MATCH (c:Comment), (u:User)
WHERE c.authorUserId = u.userId
CREATE (u)-[:AUTHORED]->(c);

// Knowledge Articles authored by Users
MATCH (a:KnowledgeArticle), (u:User)
WHERE a.authorUserId = u.userId
CREATE (u)-[:AUTHORED]->(a);

// Knowledge Articles belong to Category
MATCH (a:KnowledgeArticle), (c:Category)
WHERE a.categoryId = c.categoryId
CREATE (a)-[:BELONGS_TO_CATEGORY]->(c);

// Attachments attached to Tickets
MATCH (f:Attachment), (t:Ticket)
WHERE f.ticketId = t.ticketId
CREATE (f)-[:ATTACHED_TO]->(t);

// 4. Create Indexes for Performance
CREATE INDEX user_email_index FOR (u:User) ON (u.email);
CREATE INDEX ticket_creation_date_index FOR (t:Ticket) ON (t.creationDate);
CREATE INDEX comment_creation_date_index FOR (c:Comment) ON (c.creationDate);
CREATE INDEX knowledge_article_published_index FOR (a:KnowledgeArticle) ON (a.isPublished);

// 5. Remove temporary foreign key properties 
MATCH (u:User) REMOVE u.orgId;
MATCH (t:Ticket) REMOVE t.createdByUserId, t.assignedToUserId, t.statusId, t.priorityId, t.categoryId;
MATCH (c:Comment) REMOVE c.ticketId, c.authorUserId;
MATCH (a:KnowledgeArticle) REMOVE a.authorUserId, a.categoryId;
MATCH (f:Attachment) REMOVE f.ticketId;