import re


# Simple English stopwords (Python 3.13+ compatible, no NLTK)
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that', 'the', 'to', 'was', 'will',
    'with', 'this', 'but', 'they', 'have', 'had', 'what', 'when', 'where', 'who',
    'which', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'some', 'such','nor', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 'can', 'just', 'should', 'now'
}


def clean_text(text: str) -> str:
    """Normalize text: lowercase, remove punctuation, remove stopwords.

    Inputs: raw text (any type) -> returns cleaned string.
    """
    if not isinstance(text, str):
        return " "
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation and extra whitespace
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t and t not in STOP_WORDS]
    
    return ' '.join(tokens)
