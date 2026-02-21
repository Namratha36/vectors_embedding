
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from wordcloud import WordCloud

# Machine Learning & Metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Embeddings
from sentence_transformers import SentenceTransformer

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')


# We use SentenceTransformer (all-MiniLM-L6-v2) as it is fast and effective
print("\n⏳ Loading Embedding Model (Sentence Transformers)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("⏳ Generating Embeddings (this may take a moment)...")
X_embeddings = embedding_model.encode(df['clean_text'].tolist(), show_progress_bar=True)

# Encode Target Labels
le = LabelEncoder()
y_labels = le.fit_transform(df['sentiment'])
# Keep mapping for later
label_mapping = dict(zip(le.transform(le.classes_), le.classes_))
print(f"Label Mapping: {label_mapping}")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y_labels, test_size=0.2, random_state=42)

# Initialize Classifier (Logistic Regression works very well with high-dim embeddings)
clf = LogisticRegression(max_iter=1000, random_state=42)

print("\n⏳ Training Model...")
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# --- METRICS ---
print("\n" + "="*40)
print("MODEL PERFORMANCE REPORT")
print("="*40)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("\n💡 INTERPRETATION:")
print("- Diagonal elements represent correct predictions.")
print("- Off-diagonal elements represent misclassifications.")
print("- Check if the model confuses 'Neutral' with 'Positive' or 'Negative' often.")

print("\n" + "="*40)
print("CUSTOM PREDICTIONS")
print("="*40)

custom_tweets = [
    "I absolutely love the new features in this update! Fantastic work.",
    "This service is terrible. I've been waiting for hours and no response.",
    "I went to the store today and bought some groceries.",
    "The movie was okay, not great but not bad either.",
    "Can't believe how fast the delivery was! Super happy."
]

# 1. Clean
clean_custom = [clean_text(t) for t in custom_tweets]
# 2. Embed
custom_embeddings = embedding_model.encode(clean_custom)
# 3. Predict
custom_preds = clf.predict(custom_embeddings)
# 4. Decode
custom_labels = le.inverse_transform(custom_preds)

for tweet, label in zip(custom_tweets, custom_labels):
    print(f"Tweet: {tweet}\nPred : {label}\n" + "-"*20)
