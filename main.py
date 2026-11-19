
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =====================================
# FILE PATHS
# =====================================

DATA_PATH = "data/SMSSpamCollection.txt"
CORRECTED_PATH = "data/corrected_messages.csv"


# =====================================
# LOAD DATASET
# =====================================

print("Loading updated dataset...\n")

if os.path.exists(CORRECTED_PATH):
    df = pd.read_csv(CORRECTED_PATH)
else:
    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1"
    )

df = df[["label", "message"]]

df["label"] = df["label"].astype(str).str.lower().str.strip()
df["message"] = df["message"].astype(str).str.strip()

df = df[df["label"].isin(["ham", "spam"])]
df = df[df["message"] != ""]
df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)

print("Dataset loaded successfully!")
print("Total messages:", len(df))

print("\nLabel distribution:")
print(df["label"].value_counts())


# =====================================
# DATA DISTRIBUTION GRAPH
# =====================================

df["label"].value_counts().plot(
    kind="bar",
    title="Ham vs Spam Message Distribution"
)

plt.xlabel("Message Type")
plt.ylabel("Number of Messages")
plt.tight_layout()
plt.savefig("data_distribution.png")
plt.close()


# =====================================
# TRAIN TEST SPLIT
# =====================================

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data size:", len(X_train))
print("Testing data size:", len(X_test))


# =====================================
# TF-IDF VECTORIZATION
# =====================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF completed!")
print("Training data shape:", X_train_tfidf.shape)
print("Testing data shape:", X_test_tfidf.shape)
print("Vocabulary size:", len(vectorizer.vocabulary_))


# =====================================
# MODEL OPTIMIZATION - ALPHA TUNING
# =====================================

alphas = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]

best_alpha = None
best_f1 = -1

print("\n=======================================")
print("             ALPHA TUNING")
print("=======================================")

for alpha in alphas:

    test_model = MultinomialNB(alpha=alpha)

    test_model.fit(X_train_tfidf, y_train)

    test_predictions = test_model.predict(X_test_tfidf)

    report = classification_report(
        y_test,
        test_predictions,
        output_dict=True,
        zero_division=0
    )

    spam_precision = report["spam"]["precision"]
    spam_recall = report["spam"]["recall"]
    spam_f1 = report["spam"]["f1-score"]

    print(f"\nAlpha: {alpha}")
    print(f"Spam Precision: {spam_precision:.2%}")
    print(f"Spam Recall: {spam_recall:.2%}")
    print(f"Spam F1-score: {spam_f1:.2%}")

    if spam_f1 > best_f1:
        best_f1 = spam_f1
        best_alpha = alpha


print("\n=======================================")
print("Best Alpha:", best_alpha)
print("Best Spam F1-score:", f"{best_f1:.2%}")


# =====================================
# FINAL MODEL TRAINING
# =====================================

model = MultinomialNB(alpha=best_alpha)

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("\nModel training completed!")


# =====================================
# MODEL EVALUATION
# =====================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", f"{accuracy * 100:.2f}%")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["ham", "spam"]
)

print("\nConfusion Matrix:\n")
print(cm)


# =====================================
# CONFUSION MATRIX GRAPH
# =====================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

display.plot(cmap="Blues")

plt.title("MailWarden Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()


# =====================================
# LINK SECURITY ANALYSIS
# =====================================

def analyze_links(message):

    urls = re.findall(
        r"https?://\S+|www\.\S+",
        message,
        flags=re.IGNORECASE
    )

    print("\n--- LINK SECURITY ANALYSIS ---")

    if not urls:
        print("No links found.")
        return

    suspicious_domains = [
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "goo.gl",
        "ow.ly",
        "is.gd",
        "cutt.ly",
        "rb.gy"
    ]

    for url in urls:

        print("\nFound URL:", url)

        if any(domain in url.lower() for domain in suspicious_domains):
            print("Warning: URL shortener detected!")

        if not url.lower().startswith("https://"):
            print("Warning: Link is not using HTTPS!")

        if re.search(
            r"https?://\d{1,3}(?:\.\d{1,3}){3}",
            url
        ):
            print("Warning: IP-based URL detected!")

        if any(
            word in url.lower()
            for word in [
                "login",
                "verify",
                "password",
                "free",
                "claim"
            ]
        ):
            print("Warning: Suspicious keyword found in URL!")


# =====================================
# SPAM REASON ANALYSIS
# =====================================

def analyze_spam_reasons(message):

    text = message.lower()

    categories = {

        "Prize / Reward Words": [
            "prize",
            "free",
            "win",
            "reward",
            "gift"
        ],

        "Urgency Words": [
            "urgent",
            "immediately",
            "limited",
            "now",
            "today",
            "expires",
            "hurry"
        ],

        "Money Related Words": [
            "cash",
            "money",
            "claim",
            "credit",
            "loan",
            "bonus",
            "payment"
        ],

        "Suspicious Actions": [
            "click",
            "call",
            "reply",
            "subscribe",
            "contact",
            "download"
        ]
    }

    print("\n--- SPAM REASON ANALYSIS ---")

    found = False

    for category, words in categories.items():

        matched = [
            word for word in words
            if re.search(
                r"\b" + re.escape(word) + r"\b",
                text
            )
        ]

        if matched:
            found = True
            print(category + ":", ", ".join(matched))

    if not found:
        print("No common spam keywords found.")


# =====================================
# SENSITIVE INFORMATION ANALYSIS
# =====================================

def analyze_sensitive_info(message):

    patterns = {

        "OTP / Verification Code": (
            r"\botp\b|verification code|security code|one.time password"
        ),

        "Password / PIN": (
            r"password|passcode|\bpin\b"
        ),

        "Bank / Card Information": (
            r"bank account|account number|credit card|debit card|\bcvv\b"
        ),

        "Personal Information": (
            r"personal information|\bnid\b|national id|date of birth"
        )
    }

    print("\n--- SENSITIVE INFORMATION CHECK ---")

    found = False

    for category, pattern in patterns.items():

        if re.search(
            pattern,
            message,
            flags=re.IGNORECASE
        ):
            found = True
            print("Warning:", category, "mentioned!")

    if not found:
        print("No common sensitive information requests detected.")


# =====================================
# ADULT / SEXUAL CONTENT DETECTION
# =====================================

def analyze_adult_content(message):

    patterns = [
        r"\bsex\b",
        r"\bsexual\b",
        r"\bsexually\b",
        r"\bporn\b",
        r"\bpornography\b",
        r"\bnude\b",
        r"\bnudes\b",
        r"\bnaked\b",
        r"\bxxx\b",
        r"\berotic\b",
        r"\bexplicit content\b",
        r"\bmasturbat\w*\b",
        r"\borgasm\w*\b",
        r"\bblowjob\b"
    ]

    found_words = []

    for pattern in patterns:

        matches = re.findall(
            pattern,
            message,
            flags=re.IGNORECASE
        )

        found_words.extend(matches)

    print("\n--- ADULT CONTENT ANALYSIS ---")

    if found_words:

        print("WARNING: Adult / Sexual Content Detected!")

        print(
            "Detected terms:",
            ", ".join(set(found_words))
        )

        print(
            "Please be careful while opening this message."
        )

    else:

        print("No common adult content keywords detected.")


# =====================================
# COMPLETE SECURITY ANALYSIS
# =====================================

def display_security_analysis(message):

    analyze_links(message)

    analyze_spam_reasons(message)

    analyze_sensitive_info(message)

    analyze_adult_content(message)


# =====================================
# INTERACTIVE MESSAGE DETECTOR
# =====================================

print("\n=======================================")
print("       MAILWARDEN MESSAGE DETECTOR")
print("=======================================")

print("Enter 'exit' to stop the program.")


while True:

    message = input("\nEnter a message to check: ").strip()

    if message.lower() == "exit":

        print("\nThank you for using MailWarden!")

        break

    if not message:

        print("Please enter a message.")

        continue

    message_tfidf = vectorizer.transform([message])

    prediction = model.predict(message_tfidf)[0]

    probabilities = model.predict_proba(message_tfidf)[0]

    classes = list(model.classes_)

    spam_probability = probabilities[classes.index("spam")]

    ham_probability = probabilities[classes.index("ham")]


    # =====================================
    # PREDICTION RESULT
    # =====================================

    print("\n=======================================")
    print("                 RESULT")
    print("=======================================")

    print("Prediction:", prediction.upper())

    print(f"Ham Probability: {ham_probability * 100:.2f}%")

    print(f"Spam Probability: {spam_probability * 100:.2f}%")


    # =====================================
    # SECURITY ANALYSIS
    # =====================================

    display_security_analysis(message)


    # =====================================
    # SPAM WARNING AND MESSAGE CONTROL
    # =====================================

    if prediction == "spam":

        print("\n ⚠️ WARNING! This message might be SPAM!")

        choice = input(
            "\nDo you want to open this message? (yes/no): "
        ).strip().lower()

        if choice == "yes":

            print("\nMessage opened:")

            print(message)

            print("\nStay safe with MailWarden!")

        else:

            print("\nMessage hidden for your safety. ")

            print(
                "\nStay cautious and avoid interacting with suspicious messages."
            )

            print("\nStay safe with MailWarden!")


    else:

        print("\nThis message appears to be HAM.")

        print("\nMessage:")

        print(message)

        print("\nStay safe with MailWarden! 🚨")
