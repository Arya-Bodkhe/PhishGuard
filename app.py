from pathlib import Path
import ipaddress, re
from urllib.parse import urlsplit
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
package = joblib.load(ROOT / "phishguard_models.joblib")
models = package["models"]
feature_names = package["feature_names"]
best_model_name = package["best_model_name"]
url_model = package["url_model"]
url_features = package["url_features"]
metrics = pd.read_csv(ROOT / "model_metrics.csv")
eda = __import__("json").loads((ROOT / "eda_summary.json").read_text())

st.set_page_config(page_title="PhishGuard", page_icon="🔎", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root {
  --ink: #edf4ff;
  --muted: #9aabc2;
  --surface: rgba(17, 29, 47, .82);
  --surface-raised: rgba(23, 39, 62, .88);
  --line: rgba(137, 164, 198, .17);
  --accent: #48d7c0;
  --accent-strong: #22b8a5;
}
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse at 12% 0%, rgba(32, 79, 112, .38), transparent 38%),
    radial-gradient(ellipse at 100% 18%, rgba(24, 91, 83, .16), transparent 32%),
    linear-gradient(145deg, #0a1220 0%, #0b1422 52%, #0a111d 100%);
  color:var(--ink);
}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stSidebar"] {
  background:linear-gradient(180deg, #101b2b 0%, #0d1725 100%);
  border-right:1px solid var(--line);
}
[data-testid="stSidebar"] > div {background:transparent;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:var(--muted);}
[data-testid="stSidebar"] [role="radiogroup"] {gap:.35rem;}
[data-testid="stSidebar"] [role="radiogroup"] label {
  border:1px solid transparent;
  border-radius:11px;
  padding:.48rem .62rem;
  transition:background .18s ease, border-color .18s ease, transform .18s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  background:rgba(72, 215, 192, .07);
  border-color:rgba(72, 215, 192, .16);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background:linear-gradient(100deg, rgba(34, 184, 165, .18), rgba(34, 184, 165, .06));
  border-color:rgba(72, 215, 192, .28);
}
.block-container {max-width:1440px;padding:2.1rem 2.8rem 4rem;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif !important;letter-spacing:-.035em;}
h1 {font-size:2.65rem !important;line-height:1.12 !important;margin:0 0 .35rem !important;color:#f5f8ff;}
h2 {font-size:1.65rem !important;margin-top:1.35rem !important;color:#f0f6ff;}
h3 {font-size:1.2rem !important;color:#e9f2ff;}
[data-testid="stAppViewContainer"] .block-container > div:first-child [data-testid="stCaptionContainer"] {
  color:#9aabc2;font-size:.98rem;line-height:1.6;margin-top:.3rem;
}
[data-testid="stMetric"] {
  background:linear-gradient(145deg, rgba(22, 38, 60, .96), rgba(15, 27, 44, .96));
  border:1px solid var(--line);border-radius:16px;padding:1rem 1.1rem;
  box-shadow:0 12px 28px rgba(0,0,0,.14);min-height:104px;
}
[data-testid="stMetricLabel"] {color:#a9bbd1 !important;font-size:.84rem !important;}
[data-testid="stMetricValue"] {color:#f2f7ff !important;font-family:'Space Grotesk',sans-serif;}
[data-testid="stMetricDelta"] {font-size:.82rem;}
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div,
[data-testid="stTextArea"] textarea {
  background:#111e30 !important;color:var(--ink) !important;
  border:1px solid rgba(137,164,198,.24) !important;border-radius:11px !important;
}
[data-testid="stTextInput"] input:focus, [data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
  border-color:rgba(72,215,192,.7) !important;box-shadow:0 0 0 3px rgba(72,215,192,.12) !important;
}
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {color:#d4e0f0 !important;}
[data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button {
  border:1px solid rgba(72,215,192,.36);border-radius:10px;
  background:linear-gradient(135deg, #1eb5a0, #168f86);color:#f5fffd;
  font-weight:700;min-height:2.65rem;transition:all .18s ease;
  box-shadow:0 7px 18px rgba(18,154,140,.16);
}
[data-testid="stButton"] button:hover, [data-testid="stFormSubmitButton"] button:hover,
[data-testid="stDownloadButton"] button:hover {
  border-color:#6be4d1;background:linear-gradient(135deg, #2bcab3, #1b9f94);
  transform:translateY(-1px);box-shadow:0 10px 22px rgba(18,154,140,.23);
}
[data-testid="stFileUploader"] section {
  background:rgba(17,30,48,.7);border:1px dashed rgba(137,164,198,.38);
  border-radius:14px;
}
[data-testid="stAlert"] {border-radius:12px;border-width:1px;}
[data-testid="stExpander"] {
  background:rgba(16,29,46,.58);border:1px solid var(--line);border-radius:12px;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border:1px solid var(--line);border-radius:12px;overflow:hidden;
}
[data-testid="stImage"] img {border-radius:14px;}
[data-testid="stProgress"] > div > div {background:linear-gradient(90deg,#21b9a5,#69e0c8);}
hr {border-color:var(--line);margin:1.4rem 0;}
small, [data-testid="stCaptionContainer"] {color:var(--muted);}
@media (max-width: 900px) {
  .block-container {padding:1.5rem 1.2rem 3rem;}
  h1 {font-size:2.15rem !important;}
}
@media (max-width: 600px) {
  .block-container {padding:1rem .85rem 2rem;}
  h1 {font-size:1.85rem !important;}
}
</style>
""", unsafe_allow_html=True)
st.title("PhishGuard")
st.caption("Phishing website classification demo · five tuned classifiers from the project model list")
page = st.sidebar.radio("Choose a tool", ["URL Scanner", "Threshold Demo", "Feature CSV Scoring", "Model Comparison", "EDA Dashboard"])
st.sidebar.caption(f"Benchmark rows: {eda['rows']:,} · Best CV model: {best_model_name}")

# Same lexical-only implementation used to train URL model; never contacts the input URL.
def get_url_features(url):
    raw = str(url).strip()
    if not raw: raise ValueError("Enter a URL.")
    target = raw if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw) else "//" + raw
    try:
        parsed=urlsplit(target); host=(parsed.hostname or "").lower().strip("."); port=parsed.port is not None
    except ValueError:
        host,port="",False
    try: ipaddress.ip_address(host.strip("[]")); is_ip=1
    except ValueError: is_ip=0
    du=sum(ch.isdigit() for ch in raw); dh=sum(ch.isdigit() for ch in host); low=raw.lower()
    v={"length_url":len(raw),"length_hostname":len(host),"ip":is_ip,"nb_dots":raw.count("."),
    "nb_hyphens":raw.count("-"),"nb_at":raw.count("@"),"nb_qm":raw.count("?"),"nb_and":raw.count("&"),
    "nb_or":raw.count("|"),"nb_eq":raw.count("="),"nb_underscore":raw.count("_"),"nb_tilde":raw.count("~"),
    "nb_percent":raw.count("%"),"nb_slash":raw.count("/"),"nb_star":raw.count("*"),"nb_colon":raw.count(":"),
    "nb_comma":raw.count(","),"nb_semicolumn":raw.count(";"),"nb_dollar":raw.count("$"),"nb_space":raw.count(" "),
    "nb_www":low.count("www"),"nb_com":low.count("com"),"ratio_digits_url":du/max(len(raw),1),
    "ratio_digits_host":dh/max(len(host),1),"punycode":int("xn--" in host),"port":int(port)}
    return pd.DataFrame([[v[c] for c in url_features]],columns=url_features)

def get_hostname(url):
    raw = str(url).strip()
    target = raw if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw) else "//" + raw
    try:
        host = urlsplit(target).hostname or ""
        return host.rstrip(".").encode("idna").decode("ascii").lower()
    except (ValueError, UnicodeError):
        return ""

TRUSTED_HOSTS = {"chatgpt.com", "www.chatgpt.com", "claude.ai", "www.claude.ai"}

if page == "URL Scanner":
    st.subheader("Check a URL")
    st.write("The model automatically predicts a class from URL-text features. The site is not opened.")
    with st.form("url_scan"):
        raw_url=st.text_input("URL",placeholder="https://example.com/login")
        go=st.form_submit_button("Analyze URL",type="primary")
    if go:
        try:
            row=get_url_features(raw_url)
            score=float(url_model.predict_proba(row)[0,1])
            prediction=int(url_model.predict(row)[0])
            hostname=get_hostname(raw_url)
            if hostname in TRUSTED_HOSTS:
                st.info(f"{hostname} exactly matches a manually maintained hostname list. This is not a safety guarantee; check the address carefully.")
            if prediction == 1:
                st.error("Model prediction: PHISHING")
            else:
                st.success("Model prediction: LEGITIMATE")
            st.metric("Estimated phishing score",f"{score:.1%}")
            st.progress(min(max(score,0.0),1.0))
            st.caption("The class is predicted automatically by the trained model. Its score is an estimate, not proof a site is safe or malicious.")
            with st.expander("Computed URL features"):
                st.dataframe(row.T.rename(columns={0:"Value"}),use_container_width=True)
        except Exception as e: st.error(str(e))
elif page == "Threshold Demo":
    st.subheader("Threshold Demo — experiment only")
    st.warning("Changing these cutoffs changes only this demo label. It does not change the URL Scanner's automatic model prediction or retrain the model.")
    demo_url=st.text_input("URL to test",placeholder="https://example.com/login",key="threshold_demo_url")
    low_cutoff=st.slider("Below this score → Legitimate (demo)",0.0,0.90,0.35,0.01,key="demo_low")
    high_min=min(0.99,low_cutoff+0.01)
    high_cutoff=st.slider("At or above this score → Phishing (demo)",high_min,1.0,max(0.65,high_min),0.01,key="demo_high")
    st.caption("Scores between the two cutoffs are labelled Review. These illustrative cutoffs are not validated operating thresholds.")
    if demo_url.strip():
        try:
            demo_row=get_url_features(demo_url)
            demo_score=float(url_model.predict_proba(demo_row)[0,1])
            if demo_score < low_cutoff:
                demo_label="LEGITIMATE (demo cutoff)"
            elif demo_score >= high_cutoff:
                demo_label="PHISHING (demo cutoff)"
            else:
                demo_label="REVIEW"
            st.metric("Model phishing score",f"{demo_score:.1%}")
            st.metric("Demo label",demo_label)
        except Exception as e:
            st.error(str(e))
elif page == "Feature CSV Scoring":
    st.subheader("Score a CSV of engineered website features")
    st.write("Upload a CSV containing the numeric feature columns from the supplied dataset.")
    selected=st.selectbox("Classifier",list(models),index=list(models).index(best_model_name))
    upload=st.file_uploader("Feature CSV",type=["csv"])
    if upload:
        data=pd.read_csv(upload)
        missing=[c for c in feature_names if c not in data.columns]
        if missing: st.error("Missing feature columns: "+", ".join(missing[:12]))
        else:
            xx=data[feature_names].apply(pd.to_numeric,errors="coerce")
            model=models[selected]; labels=model.predict(xx)
            score=model.predict_proba(xx)[:,1] if hasattr(model,"predict_proba") else model.decision_function(xx)
            out=data.copy(); out["predicted_status"]=pd.Series(labels).map({0:"legitimate",1:"phishing"}).values
            out["phishing_score"]=score
            x,y=st.columns(2); x.metric("Rows scored",f"{len(out):,}"); y.metric("Predicted phishing",f"{int((labels==1).sum()):,}")
            st.dataframe(out.head(100),use_container_width=True)
            st.download_button("Download predictions CSV",out.to_csv(index=False).encode(),"phishing_predictions.csv","text/csv")
elif page == "Model Comparison":
    st.subheader("Tuned model results")
    best=metrics.sort_values("cv_best_f1",ascending=False).iloc[0]
    a,b,c,d=st.columns(4); a.metric("CV F1 winner",best["model"]); b.metric("CV F1",f"{best['cv_best_f1']:.3f}")
    c.metric("Test F1",f"{best['test_f1_phishing']:.3f}"); d.metric("Test recall",f"{best['test_recall_phishing']:.3f}")
    fig=px.bar(metrics.sort_values("test_f1_phishing"),x="test_f1_phishing",y="model",orientation="h",color="test_roc_auc",labels={"test_f1_phishing":"Test phishing F1","model":"Model","test_roc_auc":"ROC-AUC"},title="Held-out comparison")
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(metrics,use_container_width=True,hide_index=True)
    st.download_button("Download metrics",metrics.to_csv(index=False).encode(),"phishing_model_metrics.csv","text/csv")
    st.image(str(ROOT/"figures"/"confusion_matrix.png"),caption="Confusion matrix for the CV-selected model")
else:
    st.subheader("Exploratory data analysis")
    a,b,c=st.columns(3); a.metric("Rows",f"{eda['rows']:,}"); b.metric("Missing cells",f"{eda['missing_cells']:,}"); c.metric("Numeric features",f"{eda['features_used']:,}")
    st.write("Class counts:",eda["class_counts"])
    st.image(str(ROOT/"figures"/"class_balance.png"),caption="Class balance")
    st.image(str(ROOT/"figures"/"missingness.png"),caption="Missingness check")
    st.image(str(ROOT/"figures"/"model_comparison.png"),caption="Held-out model F1 comparison")
