# Import required libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold

# Load and shuffle the datasets
train_df = pd.read_csv("train_emails.csv")
test_df = pd.read_csv("test_emails.csv")
train_df = train_df.sample(frac=1, random_state=np.random.randint(1000))

# Combine and preprocess text
train_df['text'] = (train_df['subject'] + " " + train_df['body']).str.lower()
test_df['text'] = (test_df['subject'] + " " + test_df['body']).str.lower()
train_df = train_df.drop(columns=['subject', 'body'])
test_df = test_df.drop(columns=['subject', 'body'])

# Create and apply TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_df['text'])
X_test = vectorizer.transform(test_df['text'])

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_df['label'])
y_test = le.transform(test_df['label'])

# Initialize and train models
random_state = np.random.randint(1000)
print(f"Using random state: {random_state}")

modelNB = MultinomialNB(class_prior=[0.5, 0.5])
modelLR = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=random_state)
modelSVM = SVC(kernel='linear', class_weight='balanced', probability=True, random_state=random_state)

models = {
    "Naive Bayes": modelNB,
    "Logistic Regression": modelLR,
    "Support Vector Machine": modelSVM
}

# Train and evaluate each model
print("\nModel Accuracies:")
print("-----------------")
for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name}: {accuracy * 100:.2f}%")

# Perform cross-validation
print("\nCross-validation Scores:")
print("----------------------")
kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf)
    print(f"{name}: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
