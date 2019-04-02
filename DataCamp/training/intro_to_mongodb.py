# Count of prizes versus laureates
# We retrieved two sets of data from the Nobel Prize API. We saved these data to a nobel database as two collections, prizes and laureates. Given a connected client object, client, count the number of documents in the prizes and laureates collections, and pick one of the following statements as true.
#
# Recall that you can access databases by name as attributes of the client (client.<db_name>), collections by name as attributes of databases (<db>.<collection_name>), and that the count_documents method of a collection requires a (perhaps-empty, i.e. {}) filter argument.

# Listing databases and collections
# Our MongoClient object is not actually a dictionary, so we can't call keys() to list the names of accessible databases. The same is true for listing collections of a database. Instead, we can list database names by calling list_database_names() on a client instance, and we can list collection names by calling list_collection_names() on a database instance.

# Save a list, called db_names, of the names of the databases managed by our connected client.
# Similarly, save a list, called nobel_coll_names, of the names of the collections managed by the "nobel" database.

# Save a list of names of the databases managed by client
db_names = client.list_database_names()
print(db_names)

# Save a list of names of the collections managed by the "nobel" database
nobel_coll_names = client.nobel.list_collection_names()
print(nobel_coll_names)

# Excellent! Did you notice any strange database/collection names? Every Mongo host has 'admin' and 'local' databases for internal bookkeeping, and every Mongo database has a 'system.indexes' collection to store indexes that make searches faster.

# List fields and count laureates' prizes
# Use a collection’s find_one method to return a document. This method takes an optional filter argument. Passing an empty filter ({}) is the same as passing no filter. In Python, the returned document takes the form of a dictionary. The keys of the dictionary are the (root-level) "fields" of the document.
#
# Each laureate document has a "prizes" field. This field, an array, stores info about each of the laureate’s (possibly shared) prizes. You may iterate over the collection, collecting from each document. However, a collection is not a list, so we can't write for doc in <collection> to iterate over documents. Instead, call find() to produce an iterable called a cursor, and write for doc in <collection>.find() to iterate over documents.

# Connect to the "nobel" database
db = client.nobel

# Retrieve sample prize and laureate documents
prize = db.prizes.find_one()
laureate = db.laureates.find_one()

# Get lists of the fields present in each type of document
prize_fields = list(prize.keys())
laureate_fields = list(laureate.keys())

# Compute the total number of laureate prizes
count = sum(len(doc["prizes"]) for doc in db.laureates.find())
print(count)

