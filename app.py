import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="CareWise | Healthcare AI", page_icon="🩺", layout="wide")

FEATURES = ["fever", "cough", "fatigue", "difficulty_breathing", "age",
            "gender", "blood_pressure", "cholesterol_level"]
TARGET = "disease"

st.markdown("""
<style>
.main {background-color:#f7f9fc;}
.block-container {padding-top:1.5rem;}
[data-testid="stMetric"] {background:#ffffff;padding:16px;border-radius:12px;border:1px solid #e7edf5;}
div[data-testid="stSidebar"] {background:#eef5fb;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_Dataset.csv")
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[required].copy()
    for c in ["fever", "cough", "fatigue", "difficulty_breathing", "gender"]:
        df[c] = df[c].astype(str).str.strip().str.lower()
    for c in ["age", "blood_pressure", "cholesterol_level"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[TARGET] = df[TARGET].astype(str)
    return df.dropna(subset=[TARGET])

@st.cache_resource
def train_model():
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42,
        stratify=y if y.value_counts().min() >= 2 else None
    )
    cat_cols = ["fever", "cough", "fatigue", "difficulty_breathing", "gender"]
    num_cols = ["age", "blood_pressure", "cholesterol_level"]
    prep = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), num_cols)
    ])
    model = Pipeline([
        ("preprocess", prep),
        ("classifier", RandomForestClassifier(n_estimators=250, random_state=42,
                                               class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, X_test, y_test, pred, accuracy_score(y_test, pred), df

try:
    model, X_test, y_test, y_pred, accuracy, df = train_model()
except Exception as e:
    st.error(f"Could not load/train the model: {e}")
    st.info("Keep Cleaned_Dataset.csv in the same folder as app.py and install requirements.txt.")
    st.stop()

st.sidebar.title("🩺 CareWise AI")
st.sidebar.caption("Personalized Healthcare & Medicine Information")
page = st.sidebar.radio("Navigate", ["Dashboard", "Health Assessment", "Medicine Information", "Dataset Explorer", "Model Evaluation"])
st.sidebar.divider()
st.sidebar.caption("Educational demo only. Not a medical device or substitute for a clinician.")

st.title("🩺 CareWise")
st.write("Personalized Healthcare & Medicine Recommendation System")
st.caption("Machine-learning demonstration using the provided cleaned dataset.")

if page == "Dashboard":
    st.subheader("Overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Dataset records", f"{len(df):,}")
    c2.metric("Disease classes", f"{df[TARGET].nunique()}")
    c3.metric("Model test accuracy", f"{accuracy:.1%}")
    c4.metric("Input features", len(FEATURES))
    st.markdown("### Dataset snapshot")
    left,right = st.columns(2)
    with left:
        counts = df[TARGET].value_counts().reset_index()
        counts.columns = ["Disease label", "Records"]
        st.plotly_chart(px.bar(counts, x="Disease label", y="Records",
                               title="Records by dataset label", color="Records"), use_container_width=True)
    with right:
        symptom_cols = ["fever","cough","fatigue","difficulty_breathing"]
        symptom_rates = pd.DataFrame({"Symptom":symptom_cols,
            "Yes count":[df[c].astype(str).str.lower().eq("yes").sum() for c in symptom_cols]})
        st.plotly_chart(px.bar(symptom_rates, x="Symptom", y="Yes count",
                               title="Reported symptom counts"), use_container_width=True)
    st.info("The model learns patterns from a small supplied dataset. Accuracy is a classroom/demo metric and does not establish clinical reliability.")

elif page == "Health Assessment":
    st.subheader("Symptom & profile assessment")
    st.warning("This tool cannot diagnose illness or safely prescribe medication. Use results only for a learning demonstration and consult a qualified healthcare professional.")
    with st.form("assessment"):
        a,b,c = st.columns(3)
        with a:
            age = st.slider("Age", 1, 100, 28)
            gender = st.selectbox("Gender (dataset category)", ["female","male"])
            fever = st.selectbox("Fever?", ["No","Yes"])
        with b:
            cough = st.selectbox("Cough?", ["No","Yes"])
            fatigue = st.selectbox("Fatigue?", ["No","Yes"])
            breathing = st.selectbox("Difficulty breathing?", ["No","Yes"])
        with c:
            bp = st.selectbox("Blood pressure category (dataset code)", [0,1,2], help="Use the coding defined by your dataset; this is not a clinical measurement.")
            chol = st.selectbox("Cholesterol category (dataset code)", [0,1,2], help="Use the coding defined by your dataset; this is not a lab result.")
        submitted = st.form_submit_button("Run demo assessment", type="primary", use_container_width=True)
    if submitted:
        row = pd.DataFrame([{
            "fever":fever.lower(), "cough":cough.lower(), "fatigue":fatigue.lower(),
            "difficulty_breathing":breathing.lower(), "age":age, "gender":gender,
            "blood_pressure":bp, "cholesterol_level":chol
        }], columns=FEATURES)
        predicted = model.predict(row)[0]
        probs = model.predict_proba(row)[0]
        classes = model.named_steps["classifier"].classes_
        ranked = sorted(zip(classes, probs), key=lambda x:x[1], reverse=True)[:5]
        st.markdown("### Model output")
        st.metric("Top predicted dataset label", str(predicted))
        st.caption("The percentage below is model confidence for this dataset, not the probability that you have a disease.")
        st.progress(float(max(probs)))
        st.write(f"Model confidence: **{max(probs):.1%}**")
        st.markdown("#### Other model outputs")
        st.dataframe(pd.DataFrame(ranked, columns=["Dataset label", "Model score"]).assign(
            **{"Model score": lambda x: x["Model score"].map(lambda v:f"{v:.1%}")}), use_container_width=True, hide_index=True)
        st.markdown("#### Safe next steps")
        st.write("- Record when symptoms began and whether they are worsening.")
        st.write("- Discuss symptoms, current medicines, allergies, pregnancy status (if relevant), and medical history with a clinician or pharmacist.")
        st.write("- Do not start, stop, or change a medicine based on this prediction.")
        if breathing == "Yes":
            st.error("Difficulty breathing can be urgent. Seek prompt medical assessment; if severe or sudden, call your local emergency number.")
        if fever == "Yes" and breathing == "Yes":
            st.error("Fever with difficulty breathing needs prompt medical attention.")

elif page == "Medicine Information":
    st.subheader("Medicine information & safety")
    st.write("This demo intentionally does not generate a personal prescription. The provided dataset contains disease labels and profile fields, but it does not establish safe medication choices for an individual.")
    st.markdown("### Before taking a medicine")
    st.write("- Confirm the medicine and dose with a licensed doctor or pharmacist.")
    st.write("- Share allergies, pregnancy/breastfeeding status, age, existing conditions, and every medicine/supplement you take.")
    st.write("- Check interactions, contraindications, expiry date, and the label.")
    st.write("- Do not use antibiotics, steroids, or someone else's prescription without a clinician's direction.")
    st.markdown("### Personal medicine discussion checklist")
    allergies = st.text_input("Known allergies (optional)", placeholder="e.g., penicillin; or none known")
    current = st.text_area("Current medicines/supplements (optional)")
    concern = st.text_area("What would you like to ask the pharmacist/doctor?")
    if st.button("Prepare discussion summary"):
        st.text_area("Copy this summary to discuss with a clinician",
                     f"Allergies: {allergies or 'Not provided'}\nCurrent medicines: {current or 'Not provided'}\nQuestions/concerns: {concern or 'Not provided'}",
                     height=150)

elif page == "Dataset Explorer":
    st.subheader("Explore the supplied dataset")
    st.write("Only the model input columns and disease target are shown.")
    q = st.text_input("Filter disease label")
    view = df.copy()
    if q:
        view = view[view[TARGET].str.contains(q, case=False, na=False)]
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("Download displayed rows as CSV", view.to_csv(index=False).encode("utf-8"),
                       file_name="healthcare_dataset_view.csv", mime="text/csv")
    st.markdown("### Missing values")
    missing = df.isna().sum().reset_index()
    missing.columns = ["Column", "Missing values"]
    st.dataframe(missing, use_container_width=True, hide_index=True)

elif page == "Model Evaluation":
    st.subheader("Held-out test evaluation")
    st.metric("Accuracy on test split", f"{accuracy:.1%}")
    st.caption("One random train/test split only; small or synthetic datasets can produce misleading results.")
    labels = sorted(df[TARGET].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    st.plotly_chart(px.imshow(cm_df, text_auto=True, aspect="auto",
                              labels=dict(x="Predicted label", y="Actual label", color="Count"),
                              title="Confusion matrix"), use_container_width=True)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(report).T.reset_index().rename(columns={"index":"Class / metric"})
    st.dataframe(rep_df, use_container_width=True)
    try:
        importances = model.named_steps["classifier"].feature_importances_
        names = model.named_steps["preprocess"].get_feature_names_out()
        imp = pd.DataFrame({"Feature":names, "Importance":importances}).sort_values("Importance", ascending=False).head(15)
        st.plotly_chart(px.bar(imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                               title="Top model feature importances"), use_container_width=True)
    except Exception:
        st.caption("Feature importance chart unavailable for this environment.")

st.divider()
st.caption("CareWise AI • Student project • Predictions and scores are not medical advice.")
