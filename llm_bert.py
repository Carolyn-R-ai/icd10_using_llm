import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import re
import streamlit as st
import pandas as pd
import torch
from rapidfuzz import fuzz
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="ICD-10 Matcher",
    layout="centered",
    page_icon="🧬"
)

st.title("ICD-10 Matcher")

GENERIC_WORDS = {
    "cancer", "tumor", "tumour", "malignant",
    "carcinoma", "neoplasm", "lesion", "disease"
}

def normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", "", str(text).lower())
    words = [w for w in text.split() if w not in GENERIC_WORDS]
    return " ".join(words).strip()

@st.cache_data
def load_data():
    df = pd.read_excel("icd10_with_diagnosis.xlsx")
    required = ["ICD10_Code", "Diagnosis_Name", "WHO_Full_Desc", "ICD10_Block"]
    for col in required:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            return pd.DataFrame()
    df = df.dropna(subset=["Diagnosis_Name"])
    df["Diagnosis_Name"] = df["Diagnosis_Name"].astype(str)
    df["diag_clean"] = df["Diagnosis_Name"].apply(normalize)
    return df

@st.cache_resource
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT",
        use_fast=False
    )
    model = AutoModel.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT"
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_bert()

def embed_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  

@st.cache_resource
def build_embeddings(df):
    embeddings = []
    for text in df["diag_clean"]:
        embeddings.append(embed_text(text))
    return embeddings

def bert_top_matches(user_input, df, embeddings, top_k=5):
    user_vec = embed_text(user_input)
    results = []

    input_words = set(user_input.lower().split())

    for i, row in df.iterrows():
        diag_text = row["diag_clean"]
        diag_words = set(diag_text.lower().split())

        if len(input_words & diag_words) == 0:
            continue

        diag_vec = embeddings[i]
        score = cosine_similarity(user_vec, diag_vec)[0][0]
        results.append((row, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k] if results else []

def map_icd10(user_input, df):
    clean_input = normalize(user_input)

    exact = df[df["diag_clean"] == clean_input]
    if not exact.empty:
        return exact.iloc[0], "Exact Match", 100

    best_row = None
    best_score = 0
    for _, row in df.iterrows():
        score = fuzz.token_set_ratio(clean_input, row["diag_clean"])
        if score > best_score:
            best_score = score
            best_row = row
    if best_score >= 85:
        return best_row, "Fuzzy Match", best_score

    return None, "Not Found", 0

df = load_data()
if not df.empty:
    df_embeddings = build_embeddings(df)  

    query = st.text_input(
        "Enter Diagnosis"
    )

    if st.button("Search ICD-10") and query.strip():
        
        result, method, confidence = map_icd10(query, df)

        if result is not None:
            st.success(f"Match Found ({method})")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("ICD-10 Code", result["ICD10_Code"])
            with col2:
                st.metric("Confidence", f"{confidence}%")

            st.write(f"**Diagnosis:** {result['Diagnosis_Name']}")
            st.write(f"**ICD-10 Block:** {result['ICD10_Block']}")
            st.info(result["WHO_Full_Desc"])

            suggestions = bert_top_matches(query, df, df_embeddings, top_k=5)
            if suggestions:
                st.markdown("### AI Suggested Related Diagnoses")
                for i, (row, sim) in enumerate(suggestions, start=1):
                    st.write(
                        f"{i}. **{row['Diagnosis_Name']}** — {row['ICD10_Code']}"
                    )
        else:

            st.error("Not found")
