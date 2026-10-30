import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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


def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    weighted_p, weighted_r, weighted_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)
    return [acc, macro_p, macro_r, macro_f, weighted_p, weighted_r, weighted_f]
