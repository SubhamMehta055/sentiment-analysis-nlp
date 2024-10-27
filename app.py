
from flask import Flask, render_template, request
import pickle
import re
import sklearn
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Load the sentiment analysis model and TF-IDF vectorizer which we saved in clf.pkl and tdidf.pkl respectively
with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

app = Flask(__name__)


# Define stopwords and a comprehensive emoji pattern
stopwords_set = set(stopwords.words('english'))
emoji_pattern = re.compile(
    '['
    u'\U0001F600-\U0001F64F'  # Emoticons
    u'\U0001F300-\U0001F5FF'  # Symbols & Pictographs
    u'\U0001F680-\U0001F6FF'  # Transport & Map Symbols
    u'\U0001F700-\U0001F77F'  # Alchemical Symbols
    u'\U0001F780-\U0001F7FF'  # Geometric Shapes Extended
    u'\U0001F800-\U0001F8FF'  # Supplemental Arrows-C
    u'\U0001F900-\U0001F9FF'  # Supplemental Symbols and Pictographs
    u'\U0001FA00-\U0001FA6F'  # Chess Symbols
    u'\U0001FA70-\U0001FAFF'  # Symbols and Pictographs Extended-A
    u'\U00002702-\U000027B0'  # Dingbats
    u'\U000024C2-\U0001F251'  # Enclosed characters
    ']+', flags=re.UNICODE
)

def preprocessing(text):
    # Step 1: Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)

    # Step 2: Extract emojis from text
    emojis = emoji_pattern.findall(text)

    # Step 3: Remove special characters, numbers, and extra whitespace, convert text to lowercase
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()

    # Step 4: Initialize the Porter Stemmer
    prter = PorterStemmer()

    # Step 5: Tokenize, remove stopwords, and apply stemming
    words = [prter.stem(word) for word in text.split() if word not in stopwords_set]

    # Step 6: Combine cleaned text and extracted emojis
    processed_text = " ".join(words + emojis)

    return processed_text




@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods =['POST','GET'])
def predict():
    if request.method == 'POST':
        comment = request.form['text']  #whatever user give text in GUI it will be accessible here or from index.html (for front-end)

        # Preprocess the comment by calling function
        preprocessed_comment = preprocessing(comment)

        # Transform the preprocessed comment into a feature vector
        comment_vector = tfidf.transform([preprocessed_comment])

        # Predict the sentiment
        sentiment = clf.predict(comment_vector)[0]

        return render_template('index.html', sentiment=sentiment)



if __name__ == '__main__':
    app.run(debug=True)
