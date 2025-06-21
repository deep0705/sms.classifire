
## 🚀 Project Overview

Spam messages can be annoying, risky, and even dangerous. This project demonstrates how machine learning can be used to detect spam SMS messages using natural language processing (NLP) techniques.

- ✅ Cleaned and preprocessed real-world SMS data
- ✅ Extracted features using **TF-IDF Vectorizer**
- ✅ Trained with models like **Naive Bayes**, **Logistic Regression**, and **SVM**
- ✅ Evaluated using **accuracy**, **precision**, **recall**, and **confusion matrix**

---

## 🧠 Technologies & Libraries Used

- **Python 3.8+**
- **Pandas** – for data loading & manipulation
- **NumPy** – for numerical ops
- **Scikit-learn** – ML algorithms and model evaluation
- **Matplotlib / Seaborn** – visualizations
- **NLTK / re** – text preprocessing

## 📊 Dataset

Dataset used: [`spam.csv`](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

- ~5,500 labeled SMS messages
- Labels: `ham` (non-spam) and `spam`
- Format: `label` and `text`

## 🔍 Features

- Text cleaning (lowercase, punctuation removal, etc.)
- Tokenization and stopwords removal
- Vectorization using **TF-IDF**
- Multiple classifiers for comparison
- Confusion matrix & performance metrics

## 🧪 Model Training & Evaluation

Trained models:
- ✅ Multinomial Naive Bayes (best performance for text data)
- ✅ Logistic Regression
- ✅ Support Vector Machine (SVM)

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🖥️ How to Run Locally

### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/spam-sms-classifier.git
cd spam-sms-classifier
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Script
bash
python spam_classifier.py

## 📈 Example Output

Sample Message: "Free entry in 2 a wkly comp to win FA Cup..."
Prediction: Spam ❌

Accuracy: 97.8%
Precision: 0.99
Recall: 0.94

## 📦 Folder Structure

├── data/
│   └── spam.csv
├── spam_classifier.py
├── requirements.txt
├── README.md
└── models/
    └── trained_model.pkl
## 📌 Future Improvements

* Deploy as a web app using Flask or Streamlit
* Add advanced preprocessing (lemmatization, stemming)
* Integrate with SMS APIs to scan live messages
* Use deep learning (e.g., LSTM with Keras)
