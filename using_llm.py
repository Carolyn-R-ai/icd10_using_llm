import re
import streamlit as st
import pandas as pd
import google.generativeai as genai

st.set_page_config(
    page_title="🧬 ICD-10 Matcher",
    layout="centered"
)

st.title("🧬 ICD-10 Matcher")

genai.configure(api_key = API_KEY )

STOP_WORDS = {
    "cancer", "tumor", "tumour", "malignant",
    "neoplasm", "disease", "lesion", "of", "the", "and"
}

def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)

@st.cache_data
def load_data():
    df = pd.read_excel("icd10_with_diagnosis.xlsx")

    required = ["ICD10_Code", "Diagnosis_Name", "WHO_Full_Desc", "ICD10_Block"]
    for col in required:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            return pd.DataFrame()

    df["diag_norm"] = df["Diagnosis_Name"].apply(normalize)
    return df

def llm_select_exact(user_input, excel_diagnoses):
    """LLM MUST choose ONE diagnosis from excel_diagnoses OR return NONE."""
    model = genai.GenerativeModel("gemini-2.5-flash")
    options = "\n".join(f"- {d}" for d in excel_diagnoses)

    prompt = f"""
You are NOT allowed to guess.

TASK:
Choose ONE diagnosis from the list that EXACTLY matches
the user's intent.

RULES:
- You MUST return EXACTLY one string from the list
- OR return the word NONE
- Do NOT infer medical subtypes
- Do NOT explain anything

USER INPUT:
"{user_input}"

DIAGNOSIS LIST:
{options}
"""
    try:
        resp = model.generate_content(prompt, generation_config={"temperature": 0})
        answer = resp.text.strip()
        return answer
    except:
        return "NONE"

def map_icd10(user_input, df):
    user_tokens = set(normalize(user_input).split())

    matches = []
    for _, row in df.iterrows():
        row_tokens = set(row["diag_norm"].split())
        if user_tokens.issubset(row_tokens):
            matches.append(row)

    if matches:
        if len(matches) == 1:
            return [matches[0]], "Token Match"
        else:
            return matches, "Multiple Related Matches"

    excel_list = df["Diagnosis_Name"].tolist()
    llm_answer = llm_select_exact(user_input, excel_list)
    if llm_answer != "NONE" and llm_answer in excel_list:
        row = df[df["Diagnosis_Name"] == llm_answer].iloc[0]
        return [row], "LLM Exact Match"

    return None, "Not Found"

df = load_data()

if not df.empty:
    query = st.text_input("Enter Diagnosis")

    if st.button("Search ICD-10") and query.strip():
        results, method = map_icd10(query, df)

        if results is not None:
            st.success(f"### Match Found ({method})")
            for r in results:
                st.markdown(f"**ICD-10 Code:** `{r['ICD10_Code']}`  |  **Block:** {r['ICD10_Block']}")
                st.info(r["WHO_Full_Desc"])
                st.markdown("---")
        else:
            st.error("Not Found")
else:
    st.error("Dataset not loaded")