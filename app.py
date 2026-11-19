
import streamlit as st
import pandas as pd
import os
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="MailWarden | Message Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = "data/SMSSpamCollection.txt"
CORRECTED_PATH = "data/corrected_messages.csv"
COMMUNITY_PATH = "data/community_submissions.csv"

os.makedirs("data", exist_ok=True)

# ==========================================
# 2. CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #080F1D;
    color: #E6EDF7;
}

header[data-testid="stHeader"] {
    background: #080F1D;
}

[data-testid="stSidebar"] {
    background: #0D192B;
    border-right: 1px solid #1D3049;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

[data-testid="stSidebar"] * {
    color: #D7E4F5;
}

h1, h2, h3 {
    color: #F1F6FF !important;
}

p, label {
    color: #B6C5D9;
}

.warden-brand {
    font-size: 27px;
    font-weight: 800;
    color: #42D9D0;
    letter-spacing: 1px;
}

.warden-subtitle {
    font-size: 10px;
    color: #8196B2;
    letter-spacing: 2px;
    margin-top: -5px;
}

.section-tag {
    color: #42D9D0;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
}

.main-heading {
    font-size: 30px;
    font-weight: 800;
    color: #F1F6FF;
    margin-bottom: 4px;
}

.muted {
    color: #8196B2;
    font-size: 14px;
}

.metric-card {
    background: linear-gradient(135deg, #14243A, #101D30);
    border: 1px solid #263C58;
    border-radius: 15px;
    padding: 21px;
    min-height: 125px;
}

.metric-label {
    color: #8FA6C2;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
}

.metric-value {
    color: #F1F6FF;
    font-size: 30px;
    font-weight: 800;
    margin-top: 9px;
}

.metric-note {
    color: #42D9D0;
    font-size: 12px;
    margin-top: 3px;
}

.panel {
    background: #101D30;
    border: 1px solid #263C58;
    border-radius: 15px;
    padding: 20px;
}

.status-online {
    color: #42D9D0;
    background: #12333A;
    border: 1px solid #1D6265;
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 11px;
    font-weight: 700;
}

div.stButton > button {
    background: linear-gradient(90deg, #087F91, #10A6AC);
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    min-height: 43px;
    transition: 0.2s;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #10A6AC, #25C6C2);
    color: white;
    border: none;
}

.stTextArea textarea, .stTextInput input {
    background: #0B1728 !important;
    color: #E6EDF7 !important;
    border: 1px solid #304661 !important;
    border-radius: 10px !important;
}

.stSelectbox div[data-baseweb="select"] > div {
    background: #0B1728;
    border-color: #304661;
    color: #E6EDF7;
}

[data-testid="stMetric"] {
    background: #101D30;
    border: 1px solid #263C58;
    padding: 15px;
    border-radius: 12px;
}

[data-testid="stMetricValue"] {
    color: #42D9D0;
}

[data-testid="stDataFrame"] {
    border: 1px solid #263C58;
    border-radius: 10px;
}

hr {
    border-color: #263C58;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

/* Inner sections */

[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stForm"],
[data-testid="stExpander"] {
    background: #101D30 !important;
    border: 1px solid #263C58 !important;
    border-radius: 14px !important;
}

/* Detection alerts: dark text on light background */

div[data-testid="stAlert"] {
    background: #E8EEF6 !important;
    border: 1px solid #B8C8DA !important;
    border-radius: 12px !important;
}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] div,
div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] * {
    color: #10243D !important;
    font-weight: 650 !important;
}

/* Input placeholder */

textarea::placeholder,
input::placeholder {
    color: #8196B2 !important;
    opacity: 1;
}

/* Radio options */

[data-testid="stRadio"] label {
    color: #E6EDF7 !important;
}

/* Dropdown menu */

div[data-baseweb="popover"],
li[role="option"] {
    background: #14243A !important;
    color: #F1F6FF !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. DATASET FUNCTIONS
# ==========================================

def load_dataset():

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

    for column in ["label", "message"]:
        if column not in df.columns:
            df[column] = ""

    df["label"] = (
        df["label"].fillna("")
        .astype(str).str.strip().str.lower()
    )

    df["message"] = (
        df["message"].fillna("")
        .astype(str).str.strip()
    )

    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["message"] != ""]
    df = df.drop_duplicates(subset=["message"])

    return df.reset_index(drop=True)


def load_community():

    columns = [
        "message",
        "submitted_at",
        "status",
        "reviewed_label"
    ]

    if os.path.exists(COMMUNITY_PATH):
        df = pd.read_csv(COMMUNITY_PATH)
    else:
        df = pd.DataFrame(columns=columns)

    for column in columns:
        if column not in df.columns:
            df[column] = ""

    for column in columns:
        df[column] = df[column].fillna("").astype(str)

    return df


def save_dataset(df):

    df.to_csv(CORRECTED_PATH, index=False)
    train_model.clear()


def save_community(df):

    df.to_csv(COMMUNITY_PATH, index=False)


# ==========================================
# 4. MACHINE LEARNING MODEL
# ==========================================

@st.cache_resource
def train_model(data_version):

    df = load_dataset()

    if len(df) < 2 or df["label"].nunique() < 2:
        raise ValueError(
            "Dataset needs both Ham and Spam messages."
        )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    X = vectorizer.fit_transform(df["message"])
    y = df["label"]

    model = MultinomialNB(alpha=0.1)
    model.fit(X, y)

    return model, vectorizer


# ==========================================
# 5. SECURITY ANALYSIS
# ==========================================

def analyze_links(message):

    urls = re.findall(
        r"https?://\S+|www\.\S+",
        message,
        flags=re.IGNORECASE
    )

    results = []

    suspicious_domains = [
        "bit.ly", "tinyurl.com", "t.co",
        "goo.gl", "ow.ly", "is.gd",
        "cutt.ly", "rb.gy"
    ]

    for url in urls:

        warnings = []

        if any(domain in url.lower()
               for domain in suspicious_domains):
            warnings.append("URL shortener detected")

        if not url.lower().startswith("https://"):
            warnings.append("Link is not using HTTPS")

        if re.search(
            r"https?://\d{1,3}(?:\.\d{1,3}){3}",
            url
        ):
            warnings.append("IP-based URL detected")

        if any(word in url.lower() for word in [
            "login", "verify", "password", "free", "claim"
        ]):
            warnings.append("Suspicious keyword in URL")

        results.append({
            "url": url,
            "warnings": warnings
        })

    return results


def analyze_spam_reasons(message):

    text = message.lower()

    categories = {
        "Prize / Reward": [
            "prize", "free", "win", "reward", "gift"
        ],
        "Urgency": [
            "urgent", "immediately", "limited",
            "now", "today", "expires", "hurry"
        ],
        "Money Related": [
            "cash", "money", "claim",
            "credit", "loan", "bonus", "payment"
        ],
        "Suspicious Actions": [
            "click", "call", "reply",
            "subscribe", "contact", "download"
        ]
    }

    found = {}

    for category, words in categories.items():

        matched = [
            word for word in words
            if re.search(
                r"\b" + re.escape(word) + r"\b",
                text
            )
        ]

        if matched:
            found[category] = matched

    return found


def analyze_sensitive_info(message):

    patterns = {
        "OTP / Verification Code":
            r"\botp\b|verification code|security code|one.time password",

        "Password / PIN":
            r"password|passcode|\bpin\b",

        "Bank / Card Information":
            r"bank account|account number|credit card|debit card|\bcvv\b",

        "Personal Information":
            r"personal information|\bnid\b|national id|date of birth"
    }

    found = []

    for category, pattern in patterns.items():

        if re.search(pattern, message, flags=re.IGNORECASE):
            found.append(category)

    return found


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

    return list(set(found_words))


# ==========================================
# 6. COMMON UI FUNCTIONS
# ==========================================

def page_header(tag, title, description):

    st.markdown(
        f'<div class="section-tag">{tag}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="main-heading">{title}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="muted">{description}</div>',
        unsafe_allow_html=True
    )

    st.write("")


def metric_card(label, value, note):

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 7. SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:

    st.markdown("""
    <div class="warden-brand">🛡️ MAILWARDEN</div>
    <div class="warden-subtitle">
        INTELLIGENT MESSAGE SECURITY
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown("---")

    st.markdown(
        '<div class="section-tag">WORKSPACE</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Message Scanner",
            "Community",
            "Review Center",
            "Database"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("""
    <div class="panel">
        <div class="metric-label">SYSTEM STATUS</div>
        <br>
        <span class="status-online">● SYSTEM ONLINE</span>
        <br><br>
        <div class="muted">
            MailWarden Security Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.caption("MailWarden v1.0")


# ==========================================
# 8. LOAD DATA
# ==========================================

data = load_dataset()
community = load_community()

total = len(data)
ham_count = int((data["label"] == "ham").sum())
spam_count = int((data["label"] == "spam").sum())

pending_count = int(
    (community["status"] == "pending_review").sum()
)

approved_count = int(
    community["status"].str.startswith("approved_").sum()
)


# ==========================================
# 9. DASHBOARD
# ==========================================

if page == "Dashboard":

    page_header(
        "OVERVIEW",
        "Security Dashboard",
        "Monitor your message security system and dataset."
    )

    st.markdown("""
    <span class="status-online">● ALL SYSTEMS OPERATIONAL</span>
    """, unsafe_allow_html=True)

    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "TOTAL MESSAGES",
            f"{total:,}",
            "Messages in dataset"
        )

    with c2:
        metric_card(
            "HAM MESSAGES",
            f"{ham_count:,}",
            "Legitimate messages"
        )

    with c3:
        metric_card(
            "SPAM MESSAGES",
            f"{spam_count:,}",
            "Detected spam samples"
        )

    with c4:
        metric_card(
            "PENDING REVIEWS",
            f"{pending_count:,}",
            "Awaiting approval"
        )

    st.write("")
    st.markdown("---")

    left, right = st.columns([1.2, 1])

    with left:

        st.markdown("### Dataset Distribution")
        st.caption("Ham and Spam message distribution")

        chart_data = pd.DataFrame({
            "Category": ["Ham", "Spam"],
            "Messages": [ham_count, spam_count]
        })

        st.bar_chart(
            chart_data.set_index("Category"),
            color="#25C6C2"
        )

    with right:

        st.markdown("### Community Activity")

        metric_card(
            "TOTAL SUBMISSIONS",
            len(community),
            "Community contributions"
        )

        st.write("")

        metric_card(
            "APPROVED MESSAGES",
            approved_count,
            "Added to review history"
        )

    st.write("")
    st.markdown("---")

    st.markdown("### Quick Access")

    q1, q2, q3 = st.columns(3)

    with q1:
        st.markdown(
            '<div class="panel"><h4>🔍 Scanner</h4>'
            '<p class="muted">Analyze suspicious messages.</p></div>',
            unsafe_allow_html=True
        )

    with q2:
        st.markdown(
            '<div class="panel"><h4>📩 Community</h4>'
            '<p class="muted">Submit suspicious messages.</p></div>',
            unsafe_allow_html=True
        )

    with q3:
        st.markdown(
            '<div class="panel"><h4>🗄️ Database</h4>'
            '<p class="muted">Explore message records.</p></div>',
            unsafe_allow_html=True
        )


# ==========================================
# 10. MESSAGE SCANNER
# ==========================================

elif page == "Message Scanner":

    page_header(
        "THREAT DETECTION",
        "Message Scanner",
        "Analyze a message for spam, suspicious links and security risks."
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown("### Enter message to analyze")

    user_message = st.text_area(
        "Message",
        placeholder="Paste your message here...",
        height=170,
        label_visibility="collapsed",
        key="detector_message"
    )

    scan_button = st.button(
        "🔎  SCAN MESSAGE",
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if scan_button:

        if not user_message.strip():

            st.warning("Please enter a message first.")

        else:

            try:

                data_version = (
                    os.path.getmtime(CORRECTED_PATH)
                    if os.path.exists(CORRECTED_PATH)
                    else os.path.getmtime(DATA_PATH)
                )

                model, vectorizer = train_model(data_version)

                message_tfidf = vectorizer.transform(
                    [user_message]
                )

                prediction = model.predict(message_tfidf)[0]

                probabilities = model.predict_proba(
                    message_tfidf
                )[0]

                classes = list(model.classes_)

                spam_probability = probabilities[
                    classes.index("spam")
                ]

                ham_probability = probabilities[
                    classes.index("ham")
                ]

                st.write("")
                st.markdown("---")

                st.markdown("## Detection Result")

                r1, r2, r3 = st.columns(3)

                with r1:
                    metric_card(
                        "PREDICTION",
                        prediction.upper(),
                        "ML classification result"
                    )

                with r2:
                    metric_card(
                        "SPAM PROBABILITY",
                        f"{spam_probability * 100:.1f}%",
                        "Estimated spam probability"
                    )

                with r3:
                    metric_card(
                        "HAM PROBABILITY",
                        f"{ham_probability * 100:.1f}%",
                        "Estimated ham probability"
                    )

                st.write("")

                if prediction == "spam":

                    st.error(
                        "⚠️ WARNING: This message might be SPAM!"
                    )

                    open_message = st.radio(
                        "Do you want to open this message?",
                        [
                            "No, hide message",
                            "Yes, open message"
                        ],
                        horizontal=True
                    )

                    if open_message == "Yes, open message":

                        st.warning(
                            "Be careful while interacting with this message."
                        )

                        st.code(user_message)

                    else:

                        st.info(
                            "Message hidden for your safety."
                        )

                else:

                    st.success(
                        "✓ This message appears to be HAM."
                    )

                    st.write("Message:")
                    st.code(user_message)

                st.markdown("---")

                # LINK ANALYSIS

                st.markdown("### 🔗 Link Security Analysis")

                links = analyze_links(user_message)

                if not links:

                    st.success("No links found.")

                else:

                    for item in links:

                        st.write("Found URL:", item["url"])

                        if item["warnings"]:

                            for warning in item["warnings"]:
                                st.warning(warning)

                        else:

                            st.info(
                                "No common link warning detected. "
                                "This does not guarantee the link is safe."
                            )

                st.markdown("---")

                # SPAM REASONS

                st.markdown("### 🚨 Spam Reason Analysis")

                reasons = analyze_spam_reasons(user_message)

                if reasons:

                    for category, words in reasons.items():

                        st.write(
                            f"**{category}:** {', '.join(words)}"
                        )

                else:

                    st.info(
                        "No common spam keywords detected."
                    )

                st.markdown("---")

                # SENSITIVE INFORMATION

                st.markdown("### 🔐 Sensitive Information Check")

                sensitive = analyze_sensitive_info(user_message)

                if sensitive:

                    for item in sensitive:

                        st.warning(
                            f"Sensitive information mentioned: {item}"
                        )

                else:

                    st.success(
                        "No common sensitive information requests detected."
                    )

                st.markdown("---")

                # ADULT CONTENT

                st.markdown("### 🔞 Adult Content Analysis")

                adult_words = analyze_adult_content(user_message)

                if adult_words:

                    st.error(
                        "Adult / Sexual Content Detected!"
                    )

                    st.write(
                        "Detected terms:",
                        ", ".join(adult_words)
                    )

                    st.warning(
                        "Please be careful while opening this message."
                    )

                else:

                    st.success(
                        "No common adult content keywords detected."
                    )

                st.caption(
                    "Security checks are keyword/rule-based. "
                    "They do not guarantee that a message or link is safe."
                )

            except Exception as e:

                st.error(f"Message analysis failed: {e}")


# ==========================================
# 11. COMMUNITY SUBMISSION
# ==========================================

elif page == "Community":

    page_header(
        "COMMUNITY CONTRIBUTION",
        "Submit a Message",
        "Help improve MailWarden by submitting suspicious messages."
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown("### 📩 New Submission")

    with st.form(
        "community_submission_form",
        clear_on_submit=True
    ):

        new_message = st.text_area(
            "Message",
            placeholder="Paste your suspicious message here...",
            height=160
        )

        submit_button = st.form_submit_button(
            "SUBMIT FOR REVIEW",
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if submit_button:

        if not new_message.strip():

            st.warning("Please enter a message first.")

        else:

            community = load_community()

            existing_messages = (
                community["message"]
                .str.strip()
                .str.casefold()
            )

            if new_message.strip().casefold() in existing_messages.values:

                st.warning(
                    "This message has already been submitted."
                )

            else:

                new_submission = pd.DataFrame([{
                    "message": new_message.strip(),
                    "submitted_at": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "status": "pending_review",
                    "reviewed_label": ""
                }])

                community = pd.concat(
                    [community, new_submission],
                    ignore_index=True
                )

                save_community(community)

                st.success(
                    "Message submitted successfully!"
                )

                st.info(
                    "Your message is now waiting for review."
                )

                st.rerun()

    st.write("")
    st.markdown("---")

    st.markdown("### Your Contribution")

    community = load_community()

    c1, c2 = st.columns(2)

    with c1:
        metric_card(
            "TOTAL SUBMISSIONS",
            len(community),
            "Messages submitted"
        )

    with c2:
        metric_card(
            "PENDING REVIEWS",
            int(
                (community["status"] == "pending_review").sum()
            ),
            "Waiting for review"
        )


# ==========================================
# 12. REVIEW CENTER
# ==========================================

elif page == "Review Center":

    page_header(
        "MODERATION",
        "Review Center",
        "Review, classify, approve or delete community submissions."
    )

    community = load_community()

    pending_indices = community[
        community["status"] == "pending_review"
    ].index.tolist()

    approved = community[
        community["status"].str.startswith("approved_")
    ]

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "TOTAL SUBMISSIONS",
            len(community),
            "All community messages"
        )

    with c2:
        metric_card(
            "PENDING",
            len(pending_indices),
            "Waiting for review"
        )

    with c3:
        metric_card(
            "APPROVED",
            len(approved),
            "Reviewed messages"
        )

    st.write("")
    st.markdown("---")

    st.markdown("## Pending Messages")

    if not pending_indices:

        st.success("No pending messages to review.")

    else:

        for index in pending_indices:

            message = str(
                community.loc[index, "message"]
            )

            with st.container(border=True):

                st.markdown(
                    f"### 📩 Submission #{index + 1}"
                )

                safe_message = (
                    message.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )

                st.markdown(
                    f"""
                    <div style="
                        background:#0B1728;
                        border:1px solid #35516F;
                        border-radius:10px;
                        padding:16px;
                        color:#F1F6FF;
                        margin-bottom:15px;
                    ">
                    {safe_message}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("**Select message category:**")

                selected_label = st.radio(
                    "Choose Spam or Ham",
                    ["spam", "ham"],
                    horizontal=True,
                    key=f"review_label_{index}",
                    label_visibility="collapsed"
                )

                col1, col2 = st.columns(2)

                with col1:

                    if st.button(
                        "✓ APPROVE & ADD",
                        key=f"approve_{index}",
                        use_container_width=True
                    ):

                        try:

                            data = load_dataset()

                            existing = (
                                data["message"]
                                .fillna("")
                                .astype(str)
                                .str.strip()
                                .str.casefold()
                            )

                            is_duplicate = (
                                message.strip().casefold()
                                in existing.values
                            )

                            if not is_duplicate:

                                new_row = pd.DataFrame([{
                                    "label": selected_label,
                                    "message": message
                                }])

                                data = pd.concat(
                                    [data, new_row],
                                    ignore_index=True
                                )

                                save_dataset(data)

                            community = load_community()

                            community.loc[index, "status"] = (
                                f"approved_{selected_label}"
                            )

                            community.loc[index, "reviewed_label"] = (
                                selected_label
                            )

                            save_community(community)

                            if is_duplicate:

                                st.warning(
                                    "Message already exists in dataset. "
                                    "Review status has been saved."
                                )

                            else:

                                st.success(
                                    f"Approved as {selected_label.upper()}!"
                                )

                            st.rerun()

                        except Exception as e:

                            st.error(
                                f"Approval failed: {e}"
                            )

                with col2:

                    if st.button(
                        "🗑 DELETE",
                        key=f"delete_pending_{index}",
                        use_container_width=True
                    ):

                        community = load_community()

                        community = community.drop(
                            index
                        ).reset_index(drop=True)

                        save_community(community)

                        st.success(
                            "Submission deleted successfully."
                        )

                        st.rerun()

    # ======================================
    # REVIEW HISTORY
    # ======================================

    st.markdown("---")

    st.markdown("## Reviewed Messages")

    community = load_community()

    reviewed_indices = community[
        community["status"].str.startswith("approved_")
    ].index.tolist()

    if not reviewed_indices:

        st.info("No reviewed messages yet.")

    else:

        for index in reviewed_indices:

            message = str(
                community.loc[index, "message"]
            )

            label = str(
                community.loc[index, "reviewed_label"]
            ).upper()

            with st.container(border=True):

                col1, col2 = st.columns([4, 1])

                with col1:

                    st.markdown(
                        f"### Submission #{index + 1}"
                    )

                    st.code(message)

                    if label == "SPAM":
                        st.error("Classification: SPAM")
                    else:
                        st.success("Classification: HAM")

                with col2:

                    st.write("")

                    if st.button(
                        "🗑 DELETE",
                        key=f"delete_reviewed_{index}",
                        use_container_width=True
                    ):

                        community = load_community()

                        community = community.drop(
                            index
                        ).reset_index(drop=True)

                        save_community(community)

                        st.success(
                            "Reviewed submission deleted."
                        )

                        st.rerun()


# ==========================================
# 13. DATABASE
# ==========================================

elif page == "Database":

    page_header(
        "DATA MANAGEMENT",
        "Message Database",
        "Explore, search and filter the MailWarden dataset."
    )

    data = load_dataset()

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "TOTAL MESSAGES",
            f"{len(data):,}",
            "All stored messages"
        )

    with c2:
        metric_card(
            "HAM",
            f"{int((data['label'] == 'ham').sum()):,}",
            "Legitimate samples"
        )

    with c3:
        metric_card(
            "SPAM",
            f"{int((data['label'] == 'spam').sum()):,}",
            "Spam samples"
        )

    st.write("")
    st.markdown("---")

    st.markdown("### 🔎 Search & Filter")

    col1, col2 = st.columns([2, 1])

    with col1:

        search_text = st.text_input(
            "Search message",
            placeholder="Type a word or sentence..."
        )

    with col2:

        filter_label = st.selectbox(
            "Category",
            ["All", "Ham", "Spam"]
        )

    filtered_data = data.copy()

    if filter_label != "All":

        filtered_data = filtered_data[
            filtered_data["label"] == filter_label.lower()
        ]

    if search_text:

        filtered_data = filtered_data[
            filtered_data["message"].str.contains(
                search_text,
                case=False,
                na=False,
                regex=False
            )
        ]

    st.write(
        f"Showing **{len(filtered_data)}** messages"
    )

    st.dataframe(
        filtered_data,
        use_container_width=True,
        hide_index=True,
        height=450
    )

    csv_data = filtered_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered Dataset",
        data=csv_data,
        file_name="mailwarden_dataset.csv",
        mime="text/csv"
    )


# ==========================================
# 14. FOOTER
# ==========================================

st.write("")
st.markdown("---")

st.markdown("""
<div style="text-align:center; padding:15px;">
    <div style="color:#42D9D0; font-weight:800; font-size:16px;">
        🛡️ MAILWARDEN
    </div>
    <div style="color:#8196B2; font-size:11px; margin-top:5px;">
        INTELLIGENT MESSAGE SECURITY · STAY SAFE, STAY AWARE
    </div>
</div>
""", unsafe_allow_html=True)
