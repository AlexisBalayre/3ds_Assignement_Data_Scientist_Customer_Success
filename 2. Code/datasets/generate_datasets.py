import os
import pandas as pd
import random
import uuid
from datetime import datetime, timedelta


# Helper functions
def random_datetime(start, end):
    return start + (end - start) * random.random()


# Root directory for data files
data_dir = "./datasets/data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
# Ensure the directory exists

# 1. Organizations
org_names = ["Acme Corp", "Globex", "Initech"]
orgs = []
base_date = datetime(2020, 1, 1)
for name in org_names:
    orgs.append(
        {
            "orgId": str(uuid.uuid4())[:8],
            "name": name,
            "creationDate": random_datetime(base_date, datetime.now()).isoformat(),
        }
    )
df_org = pd.DataFrame(orgs)
df_org.to_csv(f"{data_dir}/organizations.csv", index=False)

# 2. Users
first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones"]
roles = ["agent", "admin", "user"]
timezones = ["UTC", "Europe/Paris", "America/New_York"]
users = []
for i in range(10):
    fn = random.choice(first_names)
    ln = random.choice(last_names)
    org = random.choice(orgs)
    users.append(
        {
            "userId": str(uuid.uuid4())[:8],
            "username": f"{fn.lower()}.{ln.lower()}{i}",
            "firstName": fn,
            "lastName": ln,
            "email": f"{fn.lower()}.{ln.lower()}{i}@example.com",
            "telephone": f"+1-555-{random.randint(1000,9999)}",
            "role": random.choice(roles),
            "company": org["name"],
            "accountCreationDate": random_datetime(
                base_date, datetime.now()
            ).isoformat(),
            "lastLoginDate": random_datetime(base_date, datetime.now()).isoformat(),
            "isActive": random.choice([True, False]),
            "timezone": random.choice(timezones),
            "orgId": org["orgId"],
        }
    )
df_user = pd.DataFrame(users)
df_user.to_csv(f"{data_dir}/users.csv", index=False)

# 3. Status
status_values = ["Open", "In Progress", "Closed"]
statuses = []
for name in status_values:
    statuses.append(
        {
            "statusId": str(uuid.uuid4())[:8],
            "name": name,
            "description": f"Ticket is {name.lower()}",
            "isActive": name != "Closed",
        }
    )
df_status = pd.DataFrame(statuses)
df_status.to_csv(f"{data_dir}/statuses.csv", index=False)

# 4. Priority
priority_levels = [("Low", 1), ("Medium", 2), ("High", 3)]
priorities = []
for name, level in priority_levels:
    priorities.append(
        {
            "priorityId": str(uuid.uuid4())[:8],
            "name": name,
            "level": level,
            "slaHours": level * 24,
        }
    )
df_priority = pd.DataFrame(priorities)
df_priority.to_csv(f"{data_dir}/priorities.csv", index=False)

# 5. Category
cat_names = ["Incident", "Service Request", "Problem"]
categories = []
for name in cat_names:
    categories.append(
        {
            "categoryId": str(uuid.uuid4())[:8],
            "name": name,
            "description": f"{name} tickets",
            "autoAssignmentRules": "none",
        }
    )
df_category = pd.DataFrame(categories)
df_category.to_csv(f"{data_dir}/categories.csv", index=False)

# 6. Tickets
tickets = []
for i in range(20):
    creator = random.choice(users)
    assignee = random.choice(users)
    status = random.choice(statuses)
    priority = random.choice(priorities)
    category = random.choice(categories)
    creation = random_datetime(base_date, datetime.now())
    last_update = creation + timedelta(days=random.randint(0, 30))
    due = creation + timedelta(days=random.randint(1, 60))
    tickets.append(
        {
            "ticketId": str(uuid.uuid4())[:8],
            "title": f"Ticket {i+1}",
            "description": f"Description for ticket {i+1}",
            "creationDate": creation.isoformat(),
            "lastUpdateDate": last_update.isoformat(),
            "dueDate": due.isoformat(),
            "resolutionSummary": "",
            "createdByUserId": creator["userId"],
            "assignedToUserId": assignee["userId"],
            "statusId": status["statusId"],
            "priorityId": priority["priorityId"],
            "categoryId": category["categoryId"],
        }
    )
df_ticket = pd.DataFrame(tickets)
df_ticket.to_csv(f"{data_dir}/tickets.csv", index=False)

# 7. Comments
comments = []
for i in range(30):
    ticket = random.choice(tickets)
    author = random.choice(users)
    comments.append(
        {
            "commentId": str(uuid.uuid4())[:8],
            "content": f"Comment {i+1} on ticket {ticket['ticketId']}",
            "creationDate": random_datetime(base_date, datetime.now()).isoformat(),
            "isInternal": random.choice([True, False]),
            "isSolution": random.choice([True, False]),
            "ticketId": ticket["ticketId"],
            "authorUserId": author["userId"],
        }
    )
df_comment = pd.DataFrame(comments)
df_comment.to_csv(f"{data_dir}/comments.csv", index=False)

# 8. Knowledge Articles
articles = []
for i in range(5):
    author = random.choice(users)
    category = random.choice(categories)
    created = random_datetime(base_date, datetime.now())
    updated = created + timedelta(days=random.randint(0, 100))
    articles.append(
        {
            "articleId": str(uuid.uuid4())[:8],
            "title": f"Article {i+1}",
            "content": f"Knowledge base content {i+1}",
            "createdDate": created.isoformat(),
            "updatedDate": updated.isoformat(),
            "viewCount": random.randint(0, 500),
            "isPublished": random.choice([True, False]),
            "authorUserId": author["userId"],
            "categoryId": category["categoryId"],
        }
    )
df_article = pd.DataFrame(articles)
df_article.to_csv(f"{data_dir}/knowledge_articles.csv", index=False)

# 9. Attachments
attachments = []
file_types = ["pdf", "png", "jpg", "txt"]
for i in range(10):
    ticket = random.choice(tickets)
    ft = random.choice(file_types)
    attachments.append(
        {
            "attachmentId": str(uuid.uuid4())[:8],
            "filename": f"file_{i+1}.{ft}",
            "fileSize": random.randint(1000, 1000000),
            "fileType": ft,
            "uploadDate": random_datetime(base_date, datetime.now()).isoformat(),
            "storagePath": f"/attachments/file_{i+1}.{ft}",
            "ticketId": ticket["ticketId"],
        }
    )
df_attachment = pd.DataFrame(attachments)
df_attachment.to_csv(f"{data_dir}/attachments.csv", index=False)

# 10. Relations
ticket_knowledge = []
for t in tickets[:10]:
    art = random.choice(articles)
    ticket_knowledge.append({"ticketId": t["ticketId"], "articleId": art["articleId"]})
df_ticket_knowledge = pd.DataFrame(ticket_knowledge)
df_ticket_knowledge.to_csv(f"{data_dir}/ticket_knowledge_ref.csv", index=False)

comment_knowledge = []
for c in comments[:10]:
    art = random.choice(articles)
    comment_knowledge.append(
        {"commentId": c["commentId"], "articleId": art["articleId"]}
    )
df_comment_knowledge = pd.DataFrame(comment_knowledge)
df_comment_knowledge.to_csv(f"{data_dir}/comment_knowledge_ref.csv", index=False)

ticket_relations = []
relation_types = ["DUPLICATE", "BLOCKS", "RELATES_TO"]
for tr in tickets[:8]:
    target = random.choice(tickets)
    ticket_relations.append(
        {
            "sourceTicketId": tr["ticketId"],
            "targetTicketId": target["ticketId"],
            "relationType": random.choice(relation_types),
        }
    )
df_ticket_rel = pd.DataFrame(ticket_relations)
df_ticket_rel.to_csv(f"{data_dir}/ticket_relations.csv", index=False)
