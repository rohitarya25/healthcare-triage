import pandas as pd
import numpy as np
import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# LOAD DATA

data = pd.read_csv(r"C:\Users\Rohit\Desktop\medAI\dataset\Symptom2Disease.csv")


# TEXT NORMALIZATION

def normalize_text(text):
    text = text.lower()
    
    # remove special characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # remove stopwords
    stop_words = ["i", "have", "am", "feel", "since", "for", 
                  "been", "having", "from", "the", "an"]
    
    words = text.split()
    words = [w for w in words if w not in stop_words]
    
    return " ".join(words)


# MERGE PHRASES (FIXED)

def merge_phrases(text):
    phrases = {
        "chest pain": "chest_pain",
        "difficulty breathing": "breathing_difficulty",
        "breathing difficulty": "breathing_difficulty",
        "high fever": "fever",
        "mild fever": "fever",
        "severe headache": "headache"
    }
    
    for k, v in phrases.items():
        text = text.replace(k, v)
    
    return text


# IMPORTANT WORD FILTER

important_words = [
    "pain", "fever", "cough", "headache", "nausea", "vomiting",
    "dizziness", "fatigue", "chills", "sweating", "weakness",
    "chest_pain", "breathing_difficulty", "blood",
    "throat", "abdominal", "stomach", "back", "neck",
    "joints", "muscles", "eyes", "nose", "mouth",
    "rash", "itching", "swollen", "red", "pimples", "spots",
    "urine", "diarrhea", "constipation", "appetite", "indigestion",
    "cold", "sore", "burning", "cramps", "swelling", "phlegm", "numbness"
]

def keep_medical_terms(text):
    return " ".join([w for w in text.split() if w in important_words])


# APPLY PIPELINE (FIXED ORDER)

def preprocess(text):
    text = normalize_text(text)
    text = merge_phrases(text)
    text = keep_medical_terms(text)
    return text

data["newText"] = data["text"].apply(preprocess)


# TF-IDF + MODEL

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['newText'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


# PREDICTION

def predict_disease(text):
    text = preprocess(text)
    vec = vectorizer.transform([text])
    
    probs = model.predict_proba(vec)[0]
    sorted_idx = np.argsort(probs)[::-1]
    
    top1 = sorted_idx[0]
    top2 = sorted_idx[1]
    
    if probs[top1] - probs[top2] > 0.2:
        return [model.classes_[top1]]
    else:
        return [model.classes_[top1], model.classes_[top2]]


# URGENCY

def get_urgency(text):
    if any(x in text for x in ["chest_pain", "breathing_difficulty", "blood"]):
        return "HIGH"
    elif any(x in text for x in ["fever", "vomiting", "headache"]):
        return "MEDIUM"
    else:
        return "LOW"

def get_advice(urgency):
    if urgency == "HIGH":
        return "Go to hospital immediately"
    elif urgency == "MEDIUM":
        return "Consult a doctor soon"
    else:
        return "Rest and monitor symptoms"

# MAIN SYSTEM

def triage_system(user_input):
    clean = preprocess(user_input)
    
    return {
        "symptoms": clean,
        "diseases": predict_disease(user_input),
        "urgency": get_urgency(clean),
        "advice": get_advice(get_urgency(clean))
    }

# TEST

print(triage_system("I have chest pain and difficulty breathing"))
print(triage_system("vomiting blood and weakness"))
print(triage_system("vomiting and nausea for two days"))
print(triage_system("I have mild cough and cold"))