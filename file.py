import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


dataset_path = "dataset"  

texts = []
labels = []

for author_folder in os.listdir(dataset_path):
    author_path = os.path.join(dataset_path, author_folder)
    if os.path.isdir(author_path):
        for file in os.listdir(author_path):
            file_path = os.path.join(author_path, file)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                texts.append(f.read())
                labels.append(author_folder)

df = pd.DataFrame({'text': texts, 'label': labels})
print("Yazar dağılımı:\n", df['label'].value_counts())


label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])


X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label_encoded'], test_size=0.2, random_state=42, stratify=df['label_encoded']
)


def get_tfidf_features(train, test, analyzer, ngram_range):
    tfidf = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, max_features=5000)
    X_train_vec = tfidf.fit_transform(train)
    X_test_vec = tfidf.transform(test)
    return X_train_vec, X_test_vec


def get_bert_embeddings(texts):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    embeddings = []

    for text in tqdm(texts, desc="BERT Embedding"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(cls_embedding)
    return np.array(embeddings)


def train_and_evaluate(X_train_vec, X_test_vec, y_train, y_test, feature_name):
    results = []

    models = {
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        "MLP": MLPClassifier(),
        "DecisionTree": DecisionTreeClassifier()
    }

    if "TFIDF" in feature_name:
        models["NaiveBayes"] = MultinomialNB()

    for name, model in models.items():
        print(f"\nTraining {name} on {feature_name}...")
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        # Doğru accuracy hesabı
        overall_acc = accuracy_score(y_test, y_pred)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        for author_idx in np.unique(y_test):
            label_name = label_encoder.inverse_transform([author_idx])[0]
            metrics = report.get(str(author_idx), {})
            if metrics:
                results.append({
                    "Author": label_name,
                    "Feature": feature_name,
                    "Model": name,
                    "Accuracy": overall_acc,
                    "Precision": metrics.get("precision", 0),
                    "Recall": metrics.get("recall", 0),
                    "F1": metrics.get("f1-score", 0)
                })

    return results


all_results = []

# TF-IDF Word 2-gram
X_train_vec, X_test_vec = get_tfidf_features(X_train, X_test, analyzer='word', ngram_range=(2,2))
all_results += train_and_evaluate(X_train_vec, X_test_vec, y_train, y_test, "TFIDF_Word_2gram")

# TF-IDF Word 3-gram
X_train_vec, X_test_vec = get_tfidf_features(X_train, X_test, analyzer='word', ngram_range=(3,3))
all_results += train_and_evaluate(X_train_vec, X_test_vec, y_train, y_test, "TFIDF_Word_3gram")

# TF-IDF Char 2-gram
X_train_vec, X_test_vec = get_tfidf_features(X_train, X_test, analyzer='char', ngram_range=(2,2))
all_results += train_and_evaluate(X_train_vec, X_test_vec, y_train, y_test, "TFIDF_Char_2gram")

# TF-IDF Char 3-gram
X_train_vec, X_test_vec = get_tfidf_features(X_train, X_test, analyzer='char', ngram_range=(3,3))
all_results += train_and_evaluate(X_train_vec, X_test_vec, y_train, y_test, "TFIDF_Char_3gram")

# BERT Embedding
X_train_vec = get_bert_embeddings(X_train.tolist())
X_test_vec = get_bert_embeddings(X_test.tolist())
all_results += train_and_evaluate(X_train_vec, X_test_vec, y_train, y_test, "BERT")


results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(by=["Author", "Feature", "Model"])
results_df.to_csv("author_classification_results.csv", index=False)
print("\nSonuçlar kaydedildi: author_classification_results.csv")


print("\n==== Sum Results ====\n")
for author in sorted(results_df['Author'].unique()):
    print(f"\nYazar: {author}")
    author_df = results_df[results_df['Author'] == author]
    for feature in author_df['Feature'].unique():
        print(f"  Özellik: {feature}")
        for _, row in author_df[author_df['Feature'] == feature].iterrows():
            print(f"    {row['Model']:15} Acc: {row['Accuracy']:.3f}  Prec: {row['Precision']:.3f}  Rec: {row['Recall']:.3f}  F1: {row['F1']:.3f}")
