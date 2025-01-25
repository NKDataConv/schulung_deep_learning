from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
import requests
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def clean_text(text):
    """
    Bereinigt den Text durch Entfernen von Sonderzeichen und Kleinbuchstaben.
    :param text: Eingabetext
    :return: Bereinigter Text
    """
    # Entfernen von HTML-Tags, falls vorhanden
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n', '', text)
    # Entfernen von nicht-alphanumerischen Zeichen
    text = re.sub(r'[^a-zA-ZöüäÖÜÄ\s]', '', text)
    # Konvertieren in Kleinbuchstaben
    text = text.lower()
    return text

def tokenize_text(text):
    """
    Tokenisiert den Text in einzelne Wörter.
    :param text: Bereinigter Text
    :return: Liste von Tokens
    """
    return word_tokenize(text)

# Schritt 3: Entfernen von Stopwörtern
def remove_stopwords(tokens):
    """
    Entfernt Stopwörter aus der Liste von Tokens.
    :param tokens: Liste von Tokens
    :return: Liste von Tokens ohne Stopwörter
    """
    stop_words_de = set(stopwords.words('german'))  # deutsche Stopwörter
    stop_words_en = set(stopwords.words('english'))  # deutsche Stopwörter
    stop_words = stop_words_de.union(stop_words_en)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

def lemmatize_tokens(tokens):
    """
    Lemmatisiert die Tokens auf ihre Grundform.
    :param tokens: Liste von Tokens
    :return: Liste von lemmatisierten Tokens
    """
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

# data = requests.get("https://de.wikipedia.org/wiki/K%C3%BCnstliches_neuronales_Netz")
data = requests.get("https://www.lippstadt.de")
text = data.text

text = clean_text(text)
print(text)
text = tokenize_text(text)
text = remove_stopwords(text)
text = lemmatize_tokens(text)

text = [t for t in text if len(t) < 20]

text = ", ".join(text)



# Erzeugen einer Wordcloud, ersetzen Sie 'wort1, wort2' durch Ihre eigenen Wörter
wordcloud = WordCloud(background_color = 'white',
                       width = 512,
                       height = 384).generate(text)
# Speichern des Wordcloud-Bildes
wordcloud.to_file('wordcloud.png')
