import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'application.settings.development'
import django
django.setup()
from ai_chatbot import vector_utils

col = vector_utils.get_or_create_collection()
print('Collection count:', col.count())

# Test search using query_texts (same as views.py does)
results = col.query(
    query_texts=['what are the three dimensions of business growth'],
    n_results=3,
    include=['documents', 'metadatas', 'distances']
)
docs = results.get('documents', [[]])[0]
metas = results.get('metadatas', [[]])[0]
dists = results.get('distances', [[]])[0]
print('Results found:', len(docs))
for d, m, dist in zip(docs, metas, dists):
    title = m.get('document_title', '?')
    print(f'  dist={dist:.3f} [{title}]: {str(d)[:120]}')
