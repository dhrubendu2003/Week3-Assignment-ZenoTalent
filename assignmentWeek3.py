import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt')
nltk.download('stopwords')

# 1. Load the Dataset
df = pd.read_csv('IMDB Dataset.csv')

REVIEW_COLUMN = 'review'
if REVIEW_COLUMN not in df.columns:
    raise ValueError(f"Column '{REVIEW_COLUMN}' not found in dataset. Available columns: {list(df.columns)}")

# 2. Text Preprocessing
def preprocess_text(text):
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize into words
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens


print("\nPreprocessing text...")
print("\nPlease wait")
df['cleaned_tokens'] = df[REVIEW_COLUMN].apply(preprocess_text)

# Remove rows with empty tokens
df = df[df['cleaned_tokens'].map(len) > 0]
print(f"Remaining rows after cleaning: {len(df)}")

# Create cleaned text column for vectorization
df['cleaned_text'] = df['cleaned_tokens'].apply(lambda tokens: ' '.join(tokens))

# 3. N-gram Generation
def get_top_ngrams(tokens_list, n, top_k=10):
    
    all_tokens = [token for tokens in tokens_list for token in tokens]

    n_grams = list(ngrams(all_tokens, n))
    
    # frequency count
    freq_dist = Counter(n_grams)
    
    return freq_dist.most_common(top_k)

tokens_list = df['cleaned_tokens'].tolist()

# Unigrams
print("\n" + "="*50)
print("TOP 10 UNIGRAMS :")
print("="*50)
unigrams = get_top_ngrams(tokens_list, 1, 10)
for i, (word, count) in enumerate(unigrams, 1):
    print(f"{i:2}. {word}: {count}")

# Bigrams
print("\n" + "="*50)
print("TOP 10 BIGRAMS :")
print("="*50)
bigrams = get_top_ngrams(tokens_list, 2, 10)
for i, (gram, count) in enumerate(bigrams, 1):
    print(f"{i:2}. {' '.join(gram)}: {count}")

# Trigrams
print("\n" + "="*50)
print("TOP 10 TRIGRAMS :")
print("="*50)
trigrams = get_top_ngrams(tokens_list, 3, 10)
for i, (gram, count) in enumerate(trigrams, 1):
    print(f"{i:2}. {' '.join(gram)}: {count}")

# 4. Plot top 20 words
print("\nGenerating word frequency plot...")

# all unigrams
all_unigrams = [word for tokens in tokens_list for word in tokens]
word_freq = Counter(all_unigrams)
top_20_words = word_freq.most_common(20)

# separate words and counts
words, counts = zip(*top_20_words)

# plot
plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")
sns.barplot(x=list(words), y=list(counts), palette='Blues_d')
plt.title('Top 20 Most Frequent Words in Reviews', fontsize=16)
plt.xlabel('Words', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()
plt.show()

# 5. Bag-of-Words using CountVectorizer
print("\n" + "="*50)
print("Bag-of-Words Model :")
print("="*50)

cleaned_texts = df['cleaned_text']

# CountVectorizer & results
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(cleaned_texts)
 
print(f"Shape of Bag-of-Words matrix: {X.shape}")
print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
print("\nSample feature names (first 20 words in vocabulary):")
print(vectorizer.get_feature_names_out()[:20])
