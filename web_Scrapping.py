# ----------------- Imports -----------------
import asyncio

# Fix for Streamlit + KeyBERT + torch error
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import requests
from bs4 import BeautifulSoup
import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from st_aggrid import AgGrid, GridOptionsBuilder
import streamlit as st
import base64
import fitz  # PyMuPDF
from keybert import KeyBERT


# Only download stopwords if not already available
try:
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# ------------------- Functions ----------------------

def get_book_url(book_name):
    search_url = f"https://www.gutenberg.org/ebooks/search/?query={book_name.replace(' ', '+')}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    first_result = soup.find('li', class_='booklink')
    if first_result:
        book_link = first_result.find('a')['href']
        return f"https://www.gutenberg.org{book_link}.txt.utf-8"
    return None

@st.cache_data(show_spinner=False)
def scrape_novel(url):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the novel: {e}")
        return None

@st.cache_data(show_spinner=False)
def fetch_text_from_url(url):
    try:
        response = requests.get(url)
        if response.ok:
            return response.text
    except:
        return None
    return None

def extract_text_from_pdf(pdf_file):
    text = ""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return words

def get_word_stats(words):
    word_freq = Counter(words)
    total_words = sum(word_freq.values())
    df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency'])
    df['% of Text'] = (df['Frequency'] / total_words * 100).round(2).astype(str) + ' %'
    df = df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Rank'}, inplace=True)
    return df, word_freq

def plot_wordcloud(word_freq):
    wc = WordCloud(width=800, height=400, background_color='white', max_words=200).generate_from_frequencies(word_freq)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def plot_top_words(word_freq, top_n=15):
    common = word_freq.most_common(top_n)
    words, counts = zip(*common)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(words, counts, color='skyblue')
    plt.xticks(rotation=45)
    st.pyplot(fig)

def download_button(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="word_list.csv">📥 Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

# --- Keyword extraction using KeyBERT ---
def extract_top_keywords(text, top_n=5):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text, top_n=top_n, stop_words='english')
    return keywords

def plot_keywords(keywords):
    phrases = [kw for kw, score in keywords]
    scores = [score for kw, score in keywords]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(phrases[::-1], scores[::-1], color='mediumseagreen')
    ax.set_xlabel('Keyword Score')
    ax.set_title('🔑 Top 5 Keywords')
    st.pyplot(fig)

# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="Word Frequency Analyzer", layout="wide")
st.title("📖 Word Frequency Analyzer")

# Initialize session_state for text
if "text" not in st.session_state:
    st.session_state.text = ""

input_method = st.radio(
    "Choose input method:", 
    ["Enter Novel Name", "Upload PDF", "Paste Paragraph", "Provide URL"],
    horizontal=True
)

text = ""  # Initialize empty

if input_method == "Enter Novel Name":
    book_name = st.text_input("Enter the name of the novel:")
    if st.button("Fetch & Analyze"):
        with st.spinner("Fetching and processing book..."):
            book_url = get_book_url(book_name)
            if book_url:
                st.session_state.text = scrape_novel(book_url)
                if st.session_state.text:
                    st.success("Book successfully fetched!")
                else:
                    st.error("❌ Failed to fetch the novel.")
            else:
                st.error("❌ Book not found.")
    text = st.session_state.get("text", "")

elif input_method == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)

elif input_method == "Paste Paragraph":
    text = st.text_area("Paste your paragraph here:")

elif input_method == "Provide URL":
    url = st.text_input("Enter a URL to a plain text file or article:")
    if st.button("Fetch & Analyze"):
        with st.spinner("Fetching text from URL..."):
            fetched = fetch_text_from_url(url)
            if fetched:
                st.session_state.text = fetched
            else:
                st.error("❌ Failed to fetch or unsupported URL.")
    text = st.session_state.get("text", "")

# ---------- Analyze Button (shown only when there's text) ----------

if text.strip():
    if st.button("Analyze Text"):
        with st.spinner("Processing text..."):
            words = clean_text(text)
            if not words:
                st.warning("Text is too short or contains no valid words.")
            else:
                df, word_freq = get_word_stats(words)
                top_keywords = extract_top_keywords(text)

                # --- Top Keywords ---
                keyword_col, _ = st.columns([1, 1])
                with keyword_col:
                    st.subheader("🔑 Top 5 Keywords")
                    plot_keywords(top_keywords)

                # --- Row 1: Top Words and Word Cloud ---
                row1_col1, row1_col2 = st.columns(2)
                with row1_col1:
                    st.subheader("📊 Top 15 Most Frequent Words")
                    plot_top_words(word_freq)
                with row1_col2:
                    st.subheader("☁ Word Cloud")
                    plot_wordcloud(word_freq)

                # --- Row 2: Full Word List ---
                st.subheader("📋 Full Word List (Paginated)")
                gb = GridOptionsBuilder.from_dataframe(df)
                gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
                gb.configure_side_bar()
                gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, editable=False)
                grid_options = gb.build()
                AgGrid(df, gridOptions=grid_options, height=500, theme='material')

                st.markdown("### 💾 Download Options:")
                download_button(df)