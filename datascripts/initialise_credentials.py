import os

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
except ImportError:
    pass

from helper import get_collection

GOOGLE_REFRESH_TOKEN = os.environ.get('GOOGLE_REFRESH_TOKEN', '')
HONEYWELLS_REFRESH_TOKEN = os.environ.get('HONEYWELLS_REFRESH_TOKEN', '')
EMPORIA_ACCESS_TOKEN = os.environ.get('EMPORIA_ACCESS_TOKEN', '')

credentials = get_collection('credentials')
credentials.delete_many({})
credentials.insert_one({'google': GOOGLE_REFRESH_TOKEN,
                        'honeywells': HONEYWELLS_REFRESH_TOKEN,
                        'emporia': EMPORIA_ACCESS_TOKEN})
