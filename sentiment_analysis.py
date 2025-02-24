import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer as TFIDF
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.model_selection import train_test_split
import warnings
from datasets import load_dataset

warnings.filterwarnings("ignore", category=UserWarning, 
                        message=".*token_pattern.*")
nltk.download('stopwords')
np.set_printoptions(threshold= np.inf)
np.set_printoptions(precision = 2)
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping to a new line
# URL of the CSV file
url = 'https://raw.githubusercontent.com/surge-ai/stock-sentiment/main/sentiment.csv'
url1 = "https://github.com/cblancac/SentimentAnalysisBert/blob/main/data/train_150k.txt"
path = r"c:\Users\Sreek\Downloads\tweets_labelled_09042020_16072020.csv"
#Load the data into a DataFrame
df = pd.read_csv(path, delimiter=';')

quotes = df.iloc[:,1] 
all_quotes = ''.join(quotes)

count = CountVectorizer()
bag = count.fit_transform(quotes)

vocab = count.vocabulary_
count.vocabulary_



tfidf = TFIDF(use_idf=True, norm = "l2", smooth_idf= True  )
transformed =  tfidf.fit_transform(bag)
vocab = Counter()
for twit in df["text"]:
    for word in twit.split(' '):
        vocab[word] += 1

stop = stopwords.words('english')
mispell = ['abov', 'ani', 'becaus', 'befor', 'doe', 'dure', 'ha', 'hi', "it'", 'onc', 'onli', 'ourselv', "she'", "should'v", 'themselv', 'thi', 'veri', 'wa', 'whi', "you'r", "you'v", 'yourselv', "r", "v", "becau"]
for element in mispell:
    stop.append(element)

vocab_reduced = Counter()
for w, c in vocab.items():
    if not w in stop:
        vocab_reduced[w]=c



def preprocessor(text):
    """ Return a cleaned version of text
    """
    # Remove HTML markup
    text = re.sub('<[^>]*>', '', text)
    negations = {"don't": "do not", "can't": "cannot", "won't": "will not"}
    for negation in negations:
        text = re.sub(r'\b' + negation + r'\b', negations[negation], text)
    # Save emoticons for later appending
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # Remove any non-word character and append the emoticons,
    # removing the nose character for standarization. Convert to lower case
    text = (re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-', ''))
    
    return text

porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


def tokenizer(text):
    return text.split()




df = df.dropna(subset=['text', 'sentiment'])  # Replace with actual column names
X = df['text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
print(y_train.value_counts())

#Now let us actually add the lpogistic regression

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None,)

param_grid = [{'vect__ngram_range': [(1, 5)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__preprocessor': [None, preprocessor],
               'clf__penalty': [ 'l2'],
               'clf__C': [1.0, 2.0, 5.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__preprocessor': [None, preprocessor],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': [ 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0,  class_weight='balanced'))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)
gs_lr_tfidf.fit(X_train, y_train)
print('Best parameter set: ' + str(gs_lr_tfidf.best_params_))
print('Best accuracy: %.3f' % gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print('Accuracy in test: %.3f' % clf.score(X_test, y_test))

df2 = pd.read_csv(url)
twits = df2["Tweet Text"]
from sklearn.metrics import accuracy_score, classification_report

preds = clf.predict(twits)
pred_probs = clf.predict_proba(twits)
for i in range(len(twits)):
    print(f'{twits[i]} --> {preds[i]} (Probabilities: {pred_probs[i]})')

df2['Sentiment'] = df2['Sentiment'].str.lower()
df2['Tweet Text'] = df2['Tweet Text'].apply(preprocessor)

# Step 3: Separate features and labels
X_test_url1 = df2['Tweet Text']
y_test_url1 = df2['Sentiment']  # Assuming it matches 0 for negative, 1 for neutral, 2 for positive
# Step 4: Make predictions and calculate accuracy
y_pred_url1 = clf.predict(X_test_url1)
# Step 5: Calculate accuracy or other metrics
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test_url1, y_pred_url1)
print(f'Accuracy on url1 dataset: {accuracy:.3f}')
# Optional: Detailed classification report
print("Classification Report on url1 Dataset:\n", classification_report(y_test_url1, y_pred_url1))
twits = [
    "KO sucks bad. It will go down", "AMD down for a rough time. It will close to a new low and go negative", "INTC headed for the moon up up up", 

    "#PCBL CMP 411 Stock down close to 30 % from the top ⬇️⬇️ PE 31 ROE 16 Fundamentals are average (above) Last 6 years EPS increasing YoY Plan to buy this stock as cheap as possible. Lower price of buying is marked on the chart.", r"Trent has given a negative close of more than 1% on the monthly chart for the first time in the last 20 months.Is the trend going to change ?"
    ]
preds = clf.predict(twits)
for i in range(len(twits)):
    print(f'{twits[i]} --> {preds[i]}')