import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'section\s*\d+[a-zA-Z()]*', '', text)  # remove 'section xx'
    text = re.sub(r'\d+', '', text)  # remove digits
    text = text.translate(str.maketrans(
        '', '', string.punctuation))  # remove punctuation
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word)
              for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def merge_rare_labels(df, min_samples, label_col):
    counts = df[label_col].value_counts()
    rare = counts[counts < min_samples].index.tolist()
    df = df.copy()
    if rare:
        df[label_col] = df[label_col].apply(
            lambda v: v if v not in rare else "Other")
    return df, rare
