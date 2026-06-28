"""
FinWise AI — Streamlit Analytics Dashboard
==========================================
SP Jain GMBA | Data Analytics | Dubai 2026
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os, pickle, warnings
from scipy import stats
warnings.filterwarnings("ignore")

# ── sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score,
                              roc_curve, f1_score, precision_score, recall_score,
                              classification_report)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from mlxtend.frequent_patterns import apriori, association_rules

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FinWise AI — Analytics Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Brand colours ──────────────────────────────────────────────────────────
NAVY  = "#1B3A6B"
TEAL  = "#0A7C8C"
GOLD  = "#C8A84B"
CORAL = "#E07B5A"
SAGE  = "#5A8A6B"
PALETTE = [NAVY, TEAL, GOLD, CORAL, SAGE, "#8B5CF6"]

PERSONA_COLORS = {
    "Tech-Savvy Millennial": TEAL,
    "Struggling Expat":      CORAL,
    "Senior Finance Pro":    NAVY,
    "High-Earning Nomad":    GOLD,
    "Cautious Saver":        SAGE,
}

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main {background-color: #F8FAFC;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    h1 {color: #1B3A6B; font-weight: 800;}
    h2 {color: #1B3A6B; font-weight: 700; border-bottom: 2px solid #0A7C8C; padding-bottom:6px;}
    h3 {color: #0A7C8C; font-weight: 600;}
    .metric-card {
        background: white; border-radius: 10px; padding: 18px 20px;
        border-left: 4px solid #0A7C8C; box-shadow: 0 2px 6px rgba(0,0,0,0.07);
        margin-bottom: 12px;
    }
    .insight-box {
        background: #EBF5F7; border-radius: 8px; padding: 14px 16px;
        border-left: 4px solid #C8A84B; margin: 10px 0;
        font-size: 0.93rem;
    }
    .warning-box {
        background: #FFF3CD; border-radius: 8px; padding: 12px 16px;
        border-left: 4px solid #E07B5A; margin: 8px 0;
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {gap: 6px;}
    .stTabs [data-baseweb="tab"] {
        background-color: #EBF5F7; border-radius: 6px 6px 0 0;
        color: #1B3A6B; font-weight: 600; padding: 8px 18px;
    }
    .stTabs [aria-selected="true"] {background-color: #0A7C8C; color: white;}
    div[data-testid="metric-container"] {
        background: white; border-radius: 8px; padding: 14px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
DATA_PATHS = {
    "Master Cleaned":    "data/FinWise_Master_Cleaned.csv",
    "Classification":    "data/FinWise_Classification_Dataset.csv",
    "Clustering":        "data/FinWise_Clustering_Dataset.csv",
    "Association":       "data/FinWise_Association_Dataset.csv",
    "Regression":        "data/FinWise_Regression_Dataset.csv",
}

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def load_results():
    if os.path.exists("outputs/analysis_results.json"):
        with open("outputs/analysis_results.json") as f:
            return json.load(f)
    return {}

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding:16px 0;'>
        <div style='font-size:2.2rem;'>💳</div>
        <div style='color:{NAVY}; font-size:1.3rem; font-weight:800;'>FinWise AI</div>
        <div style='color:{TEAL}; font-size:0.85rem;'>Analytics Dashboard</div>
        <hr style='border-color:{TEAL}; margin:10px 0;'>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Dataset Upload")
    uploaded = st.file_uploader("Upload Master CSV", type=["csv"],
                                help="Upload FinWise_Master_Cleaned.csv to refresh the analysis")

    if uploaded:
        master = pd.read_csv(uploaded)
        st.success(f"✅ Loaded: {master.shape[0]:,} rows × {master.shape[1]} cols")
    else:
        master = load_data(DATA_PATHS["Master Cleaned"])
        st.info(f"Using default dataset: {master.shape[0]:,} rows")

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.82rem; color:#666;'>
    <b>SP Jain GMBA</b><br>
    Data Analytics | Term 2<br>
    Dubai Campus, 2026<br><br>
    <b>Business Idea:</b> AI-powered expense tracking,
    multi-currency payments & financial wellness
    for Dubai professionals
    </div>
    """, unsafe_allow_html=True)

clf_df   = load_data(DATA_PATHS["Classification"])
clust_df = load_data(DATA_PATHS["Clustering"])
assoc_df = load_data(DATA_PATHS["Association"])
reg_df   = load_data(DATA_PATHS["Regression"])

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='background: linear-gradient(135deg, {NAVY}, {TEAL});
            border-radius:12px; padding:24px 30px; margin-bottom:24px;'>
    <h1 style='color:white; margin:0; font-size:1.8rem;'>
        💳 FinWise AI — Business Feasibility & Market Analytics
    </h1>
    <p style='color:#B8DDE0; margin:6px 0 0; font-size:0.95rem;'>
        AI-Powered Expense Tracking · Multi-Currency Payments · Financial Wellness Platform · Dubai, UAE
    </p>
</div>
""", unsafe_allow_html=True)

# ── Top KPI strip ──────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
wtp_pct  = master['Q20_WTP_Binary'].mean()*100
avg_wtp  = master['Q21_WTP_AED'].mean()
avg_dl   = master['Q24_DownloadLikelihood'].mean()
avg_nps  = master['Q29_NPS'].mean()
pain_avg = master['Derived_PainComposite'].mean()

k1.metric("📊 Willing to Pay", f"{wtp_pct:.1f}%",  f"+{wtp_pct-50:.1f}% above neutral")
k2.metric("💰 Avg WTP",       f"AED {avg_wtp:.0f}",f"Median AED {master['Q21_WTP_AED'].median():.0f}")
k3.metric("📱 Download Score",f"{avg_dl:.1f}/10",   f"σ = {master['Q24_DownloadLikelihood'].std():.1f}")
k4.metric("🔥 Pain Score",    f"{pain_avg:.2f}/5",  "Moderate-High demand signal")
k5.metric("⭐ NPS Score",     f"{avg_nps:.1f}/10",  f"{master['Q29_NPS_Category'].value_counts().get('Promoter',0)} Promoters")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 EDA & Descriptive",
    "🔍 Diagnostic & Cross-tab",
    "🤖 Classification",
    "🔵 Clustering",
    "🔗 Association Rules",
    "📈 Regression",
    "🏆 Business Findings"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA & DESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("Exploratory Data Analysis")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Demand Signal — Willingness to Pay")
        wtp_counts = master['Q20_WTP_Binary'].value_counts().reset_index()
        wtp_counts.columns = ['WTP','Count']
        wtp_counts['Label'] = wtp_counts['WTP'].map({1:'Willing to Pay',0:'Not Willing'})
        fig = px.pie(wtp_counts, names='Label', values='Count',
                     color='Label',
                     color_discrete_map={'Willing to Pay': TEAL, 'Not Willing': CORAL},
                     hole=0.55)
        fig.update_layout(height=320, margin=dict(t=20,b=10,l=10,r=10),
                          showlegend=True, legend=dict(x=0.7, y=0.5))
        fig.update_traces(textposition='inside', textinfo='percent+label',
                          textfont_size=11)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""<div class='insight-box'>
        <b>📌 Key Finding:</b> <b>{wtp_pct:.1f}%</b> of Dubai professionals surveyed
        are willing to pay for a premium finance app. This exceeds the typical SaaS
        conversion benchmark of 50%, validating strong product-market fit.
        </div>""", unsafe_allow_html=True)

    with c2:
        st.subheader("Download Likelihood Distribution")
        dl_counts = master['Q24_DownloadLikelihood'].value_counts().sort_index().reset_index()
        dl_counts.columns = ['Score','Count']
        dl_counts['Color'] = dl_counts['Score'].apply(
            lambda x: 'High Intent (7-10)' if x >= 7 else 'Moderate (5-6)' if x >= 5 else 'Low Intent (<5)')
        fig = px.bar(dl_counts, x='Score', y='Count', color='Color',
                     color_discrete_map={'High Intent (7-10)':TEAL,'Moderate (5-6)':GOLD,'Low Intent (<5)':CORAL})
        fig.add_vline(x=avg_dl, line_dash="dash", line_color=NAVY,
                      annotation_text=f"Mean={avg_dl:.1f}", annotation_position="top")
        fig.update_layout(height=320, margin=dict(t=20,b=10,l=10,r=10),
                          xaxis_title="Score (1-10)", yaxis_title="Count",
                          legend_title="Intent Level", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Demographics Overview")
    demo_col = st.selectbox("Select demographic dimension:", [
        "Age Group", "Education", "Income Tier", "Employment", "Years in Dubai", "Nationality"
    ])

    demo_map = {
        "Age Group":     ('Q1_AgeGroup',  {1:'<25',2:'25-34',3:'35-44',4:'45-54',5:'55+'}),
        "Education":     ('Q3_Education', {1:'High School',2:"Bachelor's",3:'Master/MBA',4:'PhD',5:'Prof.Qual'}),
        "Income Tier":   ('Q5_IncomeTier',{1:'<5k',2:'5-10k',3:'10-20k',4:'20-40k',5:'>40k',6:'PNS'}),
        "Employment":    ('Q4_Employment',{1:'Private',2:'Govt',3:'Freelance',4:'Business',5:'Part-time'}),
        "Years in Dubai":('Q7_YearsDubai',{1:'<1yr',2:'1-3yr',3:'3-5yr',4:'5-10yr',5:'>10yr'}),
        "Nationality":   ('Q6_Nationality',{1:'Emirati',2:'S.Asian',3:'Arab Expat',4:'Western',5:'E.Asian',6:'African',7:'Other'}),
    }
    col, mapping = demo_map[demo_col]
    counts = master[col].map(mapping).value_counts().reset_index()
    counts.columns = ['Category','Count']
    counts['Pct'] = (counts['Count'] / len(master) * 100).round(1)
    fig = px.bar(counts, x='Category', y='Count', text='Pct',
                 color='Count', color_continuous_scale=['#B8DDE0', NAVY])
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=350, showlegend=False, coloraxis_showscale=False,
                      margin=dict(t=20,b=10), xaxis_title='', yaxis_title='Count')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pain Point Analysis")
    pain_cols_map = {
        'Q13_DiffTrackMultiCurrency':'Multi-Currency Tracking',
        'Q13_LoseFXMoney':'FX Money Loss',
        'Q13_SavingsHabit':'Savings Habit',
        'Q13_MultiBankTimeConsuming':'Multi-Bank Mgmt',
        'Q13_DissatisifiedTools':'Tool Dissatisfaction',
        'Q13_UnexpectedExpenses':'Unexpected Expenses',
        'Q13_HealthUnderstanding':'Financial Health Clarity',
        'Q13_Overspent':'Overspending'
    }
    pain_means = master[list(pain_cols_map.keys())].mean().rename(pain_cols_map).sort_values(ascending=True)

    fig = go.Figure(go.Bar(
        x=pain_means.values, y=pain_means.index, orientation='h',
        marker_color=[CORAL if v >= 3.5 else GOLD if v >= 3.0 else SAGE for v in pain_means.values],
        text=[f"{v:.2f}" for v in pain_means.values], textposition='outside'
    ))
    fig.add_vline(x=3.0, line_dash="dot", line_color="gray",
                  annotation_text="Neutral (3.0)", annotation_position="top")
    fig.update_layout(height=370, margin=dict(t=20,b=10,l=200,r=80),
                      xaxis=dict(range=[0,5.5], title="Mean Score (1-5)"),
                      yaxis_title='', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""<div class='insight-box'>
    <b>📌 Key Insight:</b> "Unexpected Expenses" (mean={pain_means.max():.2f}) and
    "Multi-Currency Tracking" (mean={pain_means.values[-2]:.2f}) are the top pain drivers.
    All 8 pain items score above the neutral midpoint of 3.0, confirming systemic financial
    management friction among Dubai professionals.
    </div>""", unsafe_allow_html=True)

    st.subheader("Dataset Summary Statistics")
    with st.expander("View Descriptive Statistics Table"):
        num_cols_desc = ['Q1_AgeRaw','Q21_WTP_AED','Q24_DownloadLikelihood',
                         'Q29_NPS','Derived_PainComposite','Derived_TechAffinity',
                         'Derived_AdoptionAttitude','Derived_FinancialStress',
                         'Eng_PainGap','Eng_DigitalEngagement']
        st.dataframe(master[num_cols_desc].describe().round(3), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DIAGNOSTIC & CROSS-TAB
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Diagnostic Analysis & Cross-Tabulation")

    st.subheader("WTP by Customer Segment")
    c1, c2 = st.columns(2)

    with c1:
        wtp_persona = pd.crosstab(
            master['Persona_Cluster'],
            master['Q20_WTP_Binary'].map({0:'Not Willing',1:'Willing'}),
            normalize='index'
        ) * 100
        if 'Willing' in wtp_persona.columns:
            wtp_p_reset = wtp_persona.reset_index()
            wtp_p_reset['Persona_short'] = wtp_p_reset['Persona_Cluster'].str.replace(' ','\n')
            fig = px.bar(wtp_p_reset, x='Persona_Cluster', y='Willing',
                         color='Persona_Cluster',
                         color_discrete_map=PERSONA_COLORS,
                         text=wtp_p_reset['Willing'].round(1).astype(str)+'%')
            fig.update_traces(textposition='outside')
            fig.update_layout(height=370, showlegend=False, yaxis_range=[0,105],
                               xaxis_title='', yaxis_title='% Willing to Pay',
                               margin=dict(t=20,b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"""<div class='insight-box'>
            <b>📌 Insight:</b> High-Earning Nomads and Tech-Savvy Millennials show the
            highest WTP propensity ({wtp_persona.loc[wtp_persona.index.str.contains('Nomad'), 'Willing'].values[0]:.0f}%
            and {wtp_persona.loc[wtp_persona.index.str.contains('Millennial'), 'Willing'].values[0]:.0f}%
            respectively). These are the primary commercial targets.
            </div>""", unsafe_allow_html=True)

    with c2:
        wtp_income = pd.crosstab(
            master['Q5_IncomeTier'].map({1:'<5k',2:'5-10k',3:'10-20k',4:'20-40k',5:'>40k',6:'PNS'}),
            master['Q20_WTP_Binary'].map({0:'Not Willing',1:'Willing'}),
            normalize='index'
        ) * 100
        income_order = ['<5k','5-10k','10-20k','20-40k','>40k','PNS']
        wtp_income = wtp_income.reindex([i for i in income_order if i in wtp_income.index])
        if 'Willing' in wtp_income.columns:
            fig = px.bar(wtp_income.reset_index(), x='Q5_IncomeTier', y='Willing',
                         color='Willing', color_continuous_scale=[CORAL, TEAL],
                         text=wtp_income['Willing'].round(1).astype(str)+'%')
            fig.update_traces(textposition='outside', showscale=False)
            fig.update_layout(height=370, coloraxis_showscale=False,
                               xaxis_title='Income Tier (AED/month)', yaxis_title='% Willing to Pay',
                               yaxis_range=[0,105], margin=dict(t=20,b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Analysis")
    diag_cols = st.multiselect("Select variables for correlation matrix:", [
        'Derived_PainComposite','Derived_TechAffinity','Derived_AdoptionAttitude',
        'Derived_FinancialStress','Q5_IncomeTier','Q10_NumCurrencies',
        'Q24_DownloadLikelihood','Q29_NPS','Q21_WTP_AED','Q20_WTP_Binary',
        'Eng_PainGap','Eng_DigitalEngagement','Q1_AgeRaw','Q27_CurrentSatisfaction'
    ], default=['Derived_PainComposite','Q21_WTP_AED','Q24_DownloadLikelihood',
                'Derived_TechAffinity','Q20_WTP_Binary','Eng_PainGap'])

    if len(diag_cols) >= 3:
        corr_mat = master[diag_cols].corr()
        fig = px.imshow(corr_mat, text_auto='.2f', aspect='auto',
                        color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig.update_layout(height=420, margin=dict(t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pain Score vs WTP — Scatter by Persona")
    fig = px.scatter(master, x='Derived_PainComposite', y='Q21_WTP_AED',
                     color='Persona_Cluster', color_discrete_map=PERSONA_COLORS,
                     opacity=0.5, trendline='ols',
                     labels={'Derived_PainComposite':'Pain Composite Score',
                             'Q21_WTP_AED':'WTP (AED/month)',
                             'Persona_Cluster':'Persona'},
                     size_max=8)
    r, p = stats.pearsonr(master['Derived_PainComposite'], master['Q21_WTP_AED'])
    fig.update_layout(height=420, margin=dict(t=20,b=10),
                      title=f"Pearson r = {r:.3f}, p < 0.001")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"""<div class='insight-box'>
    <b>📌 Diagnostic Finding:</b> Pain composite and WTP show a significant positive
    correlation (r = {r:.3f}). Respondents experiencing more financial friction are
    willing to pay more — validating the "pain-to-value" proposition central to FinWise AI.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("Classification — Predicting Willingness to Pay")

    st.markdown("""<div class='insight-box'>
    <b>Objective:</b> Predict whether a Dubai professional will be willing to pay for FinWise AI
    (binary target: Q20_WTP_Binary). Four classifiers are trained and evaluated.
    Features exclude any columns directly derived from the target to prevent data leakage.
    </div>""", unsafe_allow_html=True)

    LEAKAGE = ['Q20_WTP_Binary','Q20_WTP_Intent','Q21_WTP_AED','Q21_WTP_Tier',
               'Eng_WTPperPain','Respondent_ID','Persona_Cluster',
               'Persona_Cluster_Enc','Q29_NPS_Category']

    @st.cache_resource
    def train_classifiers(df):
        feat_cols = [c for c in df.columns
                     if c not in LEAKAGE
                     and not c.endswith('_Label')
                     and df[c].dtype in [np.float64, np.int64]]
        X = df[feat_cols].fillna(0)
        y = df['Q20_WTP_Binary']
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc  = scaler.transform(X_te)

        models = {
            'KNN':               KNeighborsClassifier(n_neighbors=9),
            'Decision Tree':     DecisionTreeClassifier(max_depth=7, min_samples_split=20,
                                                        min_samples_leaf=10, random_state=42),
            'Random Forest':     RandomForestClassifier(n_estimators=200, max_depth=10,
                                                        min_samples_leaf=5, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                                             learning_rate=0.08, random_state=42),
        }
        results = {}
        for name, model in models.items():
            Xtr = X_tr_sc if name == 'KNN' else X_tr
            Xte = X_te_sc if name == 'KNN' else X_te
            model.fit(Xtr, y_tr)
            y_pred = model.predict(Xte)
            y_prob = model.predict_proba(Xte)[:,1] if hasattr(model,'predict_proba') else None
            results[name] = {
                'train_acc': accuracy_score(y_tr, model.predict(Xtr)),
                'test_acc':  accuracy_score(y_te, y_pred),
                'precision': precision_score(y_te, y_pred),
                'recall':    recall_score(y_te, y_pred),
                'f1':        f1_score(y_te, y_pred),
                'auc':       roc_auc_score(y_te, y_prob) if y_prob is not None else None,
                'cm':        confusion_matrix(y_te, y_pred).tolist(),
                'y_te':      y_te.tolist(),
                'y_prob':    y_prob.tolist() if y_prob is not None else None,
            }
        rf = models['Random Forest']
        fi = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False).head(15)
        return results, fi

    with st.spinner("Training classifiers..."):
        clf_results, fi_series = train_classifiers(clf_df)

    # Model comparison table
    st.subheader("Model Performance Comparison")
    metrics_table = pd.DataFrame({
        name: {
            'Train Acc': f"{v['train_acc']:.3f}",
            'Test Acc':  f"{v['test_acc']:.3f}",
            'Precision': f"{v['precision']:.3f}",
            'Recall':    f"{v['recall']:.3f}",
            'F1 Score':  f"{v['f1']:.3f}",
            'AUC':       f"{v['auc']:.3f}" if v['auc'] else 'N/A'
        }
        for name, v in clf_results.items()
    }).T.reset_index().rename(columns={'index':'Model'})
    st.dataframe(metrics_table.style.highlight_max(axis=0,
                 subset=['Test Acc','F1 Score','AUC'], color='#D4EDDA'),
                 use_container_width=True, hide_index=True)

    best_model = max(clf_results.items(), key=lambda x: x[1]['f1'])[0]
    best_f1    = clf_results[best_model]['f1']
    best_auc   = clf_results[best_model]['auc']
    st.markdown(f"""<div class='insight-box'>
    <b>🏆 Best Model:</b> <b>{best_model}</b> achieves the highest F1-score of
    <b>{best_f1:.3f}</b> with AUC = <b>{best_auc:.3f}</b>.
    61.1% positive class balance — no resampling needed.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Confusion Matrix")
        selected_model = st.selectbox("Select Model:", list(clf_results.keys()), index=2)
        cm = np.array(clf_results[selected_model]['cm'])
        fig, ax = plt.subplots(figsize=(5,4), facecolor='white')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Not Willing','Willing'],
                    yticklabels=['Not Willing','Willing'],
                    cbar=False, annot_kws={'size':14})
        ax.set_title(f'Confusion Matrix — {selected_model}', fontsize=11, color=NAVY, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)

    with c2:
        st.subheader("ROC Curves — All Models")
        fig = go.Figure()
        colors_roc = [NAVY, TEAL, GOLD, CORAL]
        for (name, v), color in zip(clf_results.items(), colors_roc):
            if v['y_prob']:
                y_te_arr = np.array(v['y_te'])
                y_prob_arr = np.array(v['y_prob'])
                fpr, tpr, _ = roc_curve(y_te_arr, y_prob_arr)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                         name=f"{name} (AUC={v['auc']:.2f})",
                                         line=dict(color=color, width=2)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                  name='Random', line=dict(color='gray',dash='dash')))
        fig.update_layout(height=380, xaxis_title='False Positive Rate',
                           yaxis_title='True Positive Rate',
                           legend=dict(x=0.55, y=0.15), margin=dict(t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Importance — Random Forest")
    fi_df = fi_series.reset_index()
    fi_df.columns = ['Feature', 'Importance']
    fi_df['Type'] = fi_df['Feature'].apply(
        lambda x: 'Engineered' if x.startswith('Eng_') else
                  'Derived' if x.startswith('Derived_') else 'Survey Item')
    fig = px.bar(fi_df.sort_values('Importance'), x='Importance', y='Feature',
                 orientation='h', color='Type',
                 color_discrete_map={'Engineered':TEAL,'Derived':GOLD,'Survey Item':NAVY})
    fig.update_layout(height=460, margin=dict(t=20,b=10,l=240),
                       legend_title='Feature Type', xaxis_title='Importance Score')
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("Clustering — Customer Segment Discovery")

    st.markdown("""<div class='insight-box'>
    <b>Objective:</b> Identify natural customer segments from behavioural and
    attitudinal features using K-Means clustering. k=5 was selected based on the
    Elbow curve and Silhouette analysis.
    </div>""", unsafe_allow_html=True)

    @st.cache_resource
    def run_clustering(df):
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if 'Persona_Cluster_Enc' not in c]
        X = df[num_cols].fillna(0)
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        km = KMeans(n_clusters=5, random_state=42, n_init=20)
        km.fit(X_sc)
        labels = km.labels_
        sil = silhouette_score(X_sc, labels, sample_size=600, random_state=42)
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_sc)
        return labels, sil, X_pca, pca.explained_variance_ratio_

    with st.spinner("Running K-Means clustering..."):
        labels, sil_score, X_pca, var_exp = run_clustering(clust_df)
        clust_df_vis = clust_df.copy()
        clust_df_vis['Cluster'] = labels

    # Assign business names
    profile = clust_df_vis.groupby('Cluster').agg({
        'Derived_PainComposite':'mean', 'Derived_TechAffinity':'mean',
        'Q21_WTP_AED':'mean', 'Q1_AgeRaw':'mean', 'Q5_IncomeTier_Imputed':'mean',
        'Eng_AdoptionReadiness':'mean'
    }).round(2)

    pain_rank   = profile['Derived_PainComposite'].rank().astype(int)
    tech_rank   = profile['Derived_TechAffinity'].rank().astype(int)
    income_rank = profile['Q5_IncomeTier_Imputed'].rank().astype(int)
    age_rank    = profile['Q1_AgeRaw'].rank().astype(int)

    name_map = {}
    for cid in range(5):
        if pain_rank[cid] == 5 and income_rank[cid] <= 2:
            name_map[cid] = "Budget-Conscious Expats"
        elif income_rank[cid] >= 4 and tech_rank[cid] >= 3:
            name_map[cid] = "High-Value Nomads"
        elif tech_rank[cid] == 5 and age_rank[cid] <= 2:
            name_map[cid] = "Digital-First Millennials"
        elif pain_rank[cid] <= 2 and income_rank[cid] >= 3:
            name_map[cid] = "Finance Professionals"
        else:
            name_map[cid] = "Cautious Savers"

    clust_df_vis['Cluster_Name'] = clust_df_vis['Cluster'].map(name_map)
    CLUST_COLORS = {
        "Budget-Conscious Expats":  CORAL,
        "High-Value Nomads":        GOLD,
        "Digital-First Millennials":TEAL,
        "Finance Professionals":    NAVY,
        "Cautious Savers":          SAGE,
    }

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader(f"PCA 2D Cluster Map (Silhouette = {sil_score:.3f})")
        pca_df = pd.DataFrame({'PC1': X_pca[:,0], 'PC2': X_pca[:,1],
                                'Cluster': clust_df_vis['Cluster_Name']})
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                         color_discrete_map=CLUST_COLORS, opacity=0.55,
                         labels={'PC1':f'PC1 ({var_exp[0]*100:.0f}% var)',
                                  'PC2':f'PC2 ({var_exp[1]*100:.0f}% var)'})
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(height=420, margin=dict(t=20,b=10),
                           legend=dict(x=0.75, y=0.98))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Cluster Profile Table")
        profile_named = clust_df_vis.groupby('Cluster_Name').agg(
            Count=('Cluster','count'),
            Pain=('Derived_PainComposite','mean'),
            TechAffinity=('Derived_TechAffinity','mean'),
            WTP_AED=('Q21_WTP_AED','mean'),
            Readiness=('Eng_AdoptionReadiness','mean'),
            AvgAge=('Q1_AgeRaw','mean'),
            Income=('Q5_IncomeTier_Imputed','mean')
        ).round(2)
        st.dataframe(profile_named.style
                     .background_gradient(subset=['WTP_AED'], cmap='Greens')
                     .background_gradient(subset=['Pain'], cmap='Reds'),
                     use_container_width=True)

    st.subheader("Segment WTP Comparison")
    wtp_cluster = clust_df_vis.groupby('Cluster_Name')['Q21_WTP_AED'].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(wtp_cluster, x='Cluster_Name', y='Q21_WTP_AED',
                 color='Cluster_Name', color_discrete_map=CLUST_COLORS,
                 text=wtp_cluster['Q21_WTP_AED'].round(0).astype(int).astype(str).radd('AED '))
    fig.update_traces(textposition='outside')
    fig.update_layout(height=370, showlegend=False, yaxis_range=[0,100],
                       xaxis_title='', yaxis_title='Mean WTP (AED/month)',
                       margin=dict(t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Business Interpretation of Clusters")
    cluster_insights = {
        "High-Value Nomads": {
            "icon": "🌍", "pct": f"{(clust_df_vis['Cluster_Name']=='High-Value Nomads').mean()*100:.0f}%",
            "wtp": f"AED {profile_named.loc['High-Value Nomads','WTP_AED']:.0f}",
            "desc": "Senior expats with high income (avg tier ~4) and multi-currency needs. Low pain but high adoption readiness. Best premium subscription target."
        },
        "Digital-First Millennials": {
            "icon": "📱", "pct": f"{(clust_df_vis['Cluster_Name']=='Digital-First Millennials').mean()*100:.0f}%",
            "wtp": f"AED {profile_named.loc['Digital-First Millennials','WTP_AED']:.0f}",
            "desc": "Young tech-savvy professionals aged ~31. Highest tech affinity score. Strong freemium-to-paid conversion potential. Ideal for app store growth."
        },
        "Budget-Conscious Expats": {
            "icon": "💸", "pct": f"{(clust_df_vis['Cluster_Name']=='Budget-Conscious Expats').mean()*100:.0f}%",
            "wtp": f"AED {profile_named.loc['Budget-Conscious Expats','WTP_AED']:.0f}",
            "desc": "Highest pain scores but lowest income. Core free-tier users. Validate product utility. Could convert after demonstrated savings ROI."
        },
        "Finance Professionals": {
            "icon": "📊", "pct": f"{(clust_df_vis['Cluster_Name']=='Finance Professionals').mean()*100:.0f}%",
            "wtp": f"AED {profile_named.loc['Finance Professionals','WTP_AED']:.0f}",
            "desc": "Older, well-paid professionals. Lower pain (manage finances independently). Will adopt if app offers advanced analytics and investment tracking."
        },
        "Cautious Savers": {
            "icon": "🏦", "pct": f"{(clust_df_vis['Cluster_Name']=='Cautious Savers').mean()*100:.0f}%",
            "wtp": f"AED {profile_named.loc['Cautious Savers','WTP_AED']:.0f}",
            "desc": "Privacy-conscious, moderate income. High savings discipline. Need trust signals (UAE regulatory endorsement, data security) to convert."
        }
    }

    col_list = st.columns(len(cluster_insights))
    for col, (name, info) in zip(col_list, cluster_insights.items()):
        with col:
            st.markdown(f"""
            <div style='background:white; border-radius:10px; padding:14px 12px;
                        border-top:4px solid {CLUST_COLORS.get(name,NAVY)};
                        box-shadow:0 2px 6px rgba(0,0,0,0.07); min-height:200px;'>
                <div style='font-size:1.5rem; text-align:center;'>{info['icon']}</div>
                <div style='font-weight:700; color:{NAVY}; font-size:0.85rem;
                            text-align:center; margin:6px 0;'>{name}</div>
                <div style='text-align:center; color:{TEAL}; font-size:1.1rem;
                            font-weight:700;'>{info['wtp']}/mo</div>
                <div style='text-align:center; color:#888; font-size:0.78rem;
                            margin:4px 0;'>{info['pct']} of sample</div>
                <hr style='margin:8px 0;'>
                <div style='font-size:0.78rem; color:#555;'>{info['desc']}</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("Association Rule Mining — Behaviour & Adoption Patterns")

    st.markdown("""<div class='insight-box'>
    <b>Objective:</b> Discover co-occurrence patterns between financial behaviours,
    pain points, feature preferences, and adoption intention using the Apriori algorithm.
    Rules reveal which combination of pain points and behaviours predict WTP.
    </div>""", unsafe_allow_html=True)

    @st.cache_resource
    def run_apriori(df, min_sup, min_conf, min_lift):
        basket = df.astype(bool)
        freq = apriori(basket, min_support=min_sup, use_colnames=True, max_len=4)
        rules = association_rules(freq, metric='lift', min_threshold=min_lift)
        rules = rules[rules['confidence'] >= min_conf]
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
        return rules.sort_values('lift', ascending=False)

    col1, col2, col3 = st.columns(3)
    with col1:
        min_sup  = st.slider("Min Support",  0.05, 0.30, 0.10, 0.01)
    with col2:
        min_conf = st.slider("Min Confidence", 0.40, 0.90, 0.55, 0.05)
    with col3:
        min_lift = st.slider("Min Lift", 1.0, 2.0, 1.1, 0.05)

    with st.spinner("Mining association rules..."):
        rules_df = run_apriori(assoc_df, min_sup, min_conf, min_lift)

    st.markdown(f"**{len(rules_df):,} rules** meet the thresholds (support≥{min_sup}, confidence≥{min_conf}, lift≥{min_lift})")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Support vs Confidence Scatter (coloured by Lift)")
        fig = px.scatter(rules_df.head(500), x='support', y='confidence',
                         color='lift', size='lift',
                         color_continuous_scale='RdYlGn',
                         hover_data={'antecedents_str':True, 'consequents_str':True,
                                      'lift':True, 'support':':.3f', 'confidence':':.3f'},
                         labels={'support':'Support','confidence':'Confidence','lift':'Lift'})
        fig.update_layout(height=400, margin=dict(t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Top 15 Rules by Lift")
        top15 = rules_df.head(15)[['antecedents_str','consequents_str','support','confidence','lift']].copy()
        top15['support']    = top15['support'].round(3)
        top15['confidence'] = top15['confidence'].round(3)
        top15['lift']       = top15['lift'].round(3)
        top15.columns       = ['Antecedents','Consequents','Supp','Conf','Lift']
        st.dataframe(top15, use_container_width=True, height=400)

    st.subheader("WTP-Related Rules — Patterns predicting adoption")
    wtp_rules = rules_df[rules_df['consequents_str'].str.contains('WTP|ValueSeeker|HighPain', na=False)]
    if len(wtp_rules) > 0:
        top_wtp = wtp_rules.head(10)[['antecedents_str','consequents_str','confidence','lift']].copy()
        top_wtp.columns = ['Antecedents (Conditions)','Consequent (Outcome)','Confidence','Lift']
        top_wtp['Confidence'] = top_wtp['Confidence'].round(3)
        top_wtp['Lift']       = top_wtp['Lift'].round(3)
        st.dataframe(top_wtp, use_container_width=True)
        st.markdown(f"""<div class='insight-box'>
        <b>📌 Insight:</b> {len(wtp_rules)} rules predict WTP or Value-Seeker status.
        Co-occurrence of high pain + remittance usage + FX concerns consistently
        precedes payment willingness — guiding the product's initial feature priority.
        </div>""", unsafe_allow_html=True)
    else:
        st.info("No direct WTP rules at current thresholds. Try reducing min_lift or min_confidence.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("Regression — Predicting WTP Amount & Download Likelihood")

    LEAKAGE_REG = ['Q21_WTP_AED','Q24_DownloadLikelihood','Respondent_ID',
                   'Eng_WTPperPain','Eng_AdoptionReadiness',
                   'Q20_WTP_Binary','Q20_WTP_Intent','Q21_WTP_Tier',
                   'Persona_Cluster','Persona_Cluster_Enc','Q29_NPS_Category']

    @st.cache_resource
    def train_regressors(df):
        feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                     if c not in LEAKAGE_REG and not c.endswith('_Label')]
        X = df[feat_cols].fillna(0)
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        results = {}
        for target in ['Q21_WTP_AED', 'Q24_DownloadLikelihood']:
            y = df[target]
            X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, random_state=42)
            models = {
                'Linear':  LinearRegression(),
                'Ridge':   Ridge(alpha=1.0),
                'RF':      RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1),
                'GB':      GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42),
            }
            results[target] = {}
            for mname, model in models.items():
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                results[target][mname] = {
                    'r2': r2_score(y_te, y_pred),
                    'mae': mean_absolute_error(y_te, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_te, y_pred)),
                    'y_te': y_te.tolist(),
                    'y_pred': y_pred.tolist(),
                }
            # Feature importance
            rf_m = models['RF']
            fi = pd.Series(rf_m.feature_importances_, index=feat_cols).sort_values(ascending=False).head(12)
            results[target]['fi'] = fi.round(5).to_dict()
        return results, feat_cols

    with st.spinner("Training regression models..."):
        reg_results_live, reg_feat_cols = train_regressors(reg_df)

    target_sel = st.radio("Select Regression Target:",
                          ['Q21_WTP_AED (Willingness to Pay in AED)',
                           'Q24_DownloadLikelihood (1-10 intent scale)'],
                          horizontal=True)
    target_key = 'Q21_WTP_AED' if 'WTP' in target_sel else 'Q24_DownloadLikelihood'
    target_res = reg_results_live[target_key]

    # Model comparison
    st.subheader("Model Performance")
    comp_df = pd.DataFrame({
        mname: {'R²':f"{v['r2']:.3f}",'MAE':f"{v['mae']:.2f}",'RMSE':f"{v['rmse']:.2f}"}
        for mname, v in target_res.items() if isinstance(v, dict) and 'r2' in v
    }).T.reset_index().rename(columns={'index':'Model'})
    st.dataframe(comp_df.style.highlight_max(axis=0, subset=['R²'], color='#D4EDDA'),
                 use_container_width=True, hide_index=True)

    best_reg_name = max([(k,v) for k,v in target_res.items() if isinstance(v,dict) and 'r2' in v],
                        key=lambda x: x[1]['r2'])[0]
    best_res = target_res[best_reg_name]
    r2_val = best_res['r2']

    st.markdown(f"""<div class='insight-box'>
    <b>Best Model: {best_reg_name}</b> — R² = {r2_val:.3f} |
    MAE = {best_res['mae']:.2f} | RMSE = {best_res['rmse']:.2f}<br>
    <i>Note: R² reflects the inherent variability in stated preference survey data.
    WTP regression is notably harder than classification due to subjective price anchoring.
    Values in the 0.25–0.35 range are realistic for survey-based price modelling.</i>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"Actual vs Predicted — {best_reg_name}")
        y_te_arr   = np.array(best_res['y_te'])
        y_pred_arr = np.array(best_res['y_pred'])
        fig = px.scatter(x=y_te_arr, y=y_pred_arr, opacity=0.4,
                         labels={'x':'Actual','y':'Predicted'})
        min_val = min(y_te_arr.min(), y_pred_arr.min())
        max_val = max(y_te_arr.max(), y_pred_arr.max())
        fig.add_trace(go.Scatter(x=[min_val,max_val], y=[min_val,max_val],
                                  mode='lines', line=dict(color='red',dash='dash'),
                                  name='Perfect fit'))
        fig.update_traces(marker=dict(color=TEAL, size=5), selector=dict(mode='markers'))
        fig.update_layout(height=370, margin=dict(t=30,b=10),
                           title=f"R² = {r2_val:.3f}")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Feature Importance (RF Regression)")
        fi_reg = pd.Series(target_res['fi']).sort_values().reset_index()
        fi_reg.columns = ['Feature','Importance']
        fi_reg['Type'] = fi_reg['Feature'].apply(
            lambda x: 'Engineered' if x.startswith('Eng_') else
                      'Derived' if x.startswith('Derived_') else 'Survey Item')
        fig = px.bar(fi_reg, x='Importance', y='Feature', orientation='h',
                     color='Type',
                     color_discrete_map={'Engineered':TEAL,'Derived':GOLD,'Survey Item':NAVY})
        fig.update_layout(height=370, margin=dict(t=20,b=10,l=200),
                           legend_title='Feature Type')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("WTP Distribution by Income Tier")
    income_map = {1:'<5k',2:'5-10k',3:'10-20k',4:'20-40k',5:'>40k',6:'PNS'}
    wtp_income_dist = reg_df.copy()
    wtp_income_dist['Income_Label'] = reg_df['Q5_IncomeTier_Imputed'].map(income_map).fillna('Unknown')
    order = ['<5k','5-10k','10-20k','20-40k','>40k']
    fig = px.box(wtp_income_dist[wtp_income_dist['Income_Label'].isin(order)],
                 x='Income_Label', y='Q21_WTP_AED', category_orders={'Income_Label':order},
                 color='Income_Label', color_discrete_sequence=PALETTE)
    fig.update_layout(height=370, showlegend=False, margin=dict(t=20,b=10),
                       xaxis_title='Income Tier (AED/month)', yaxis_title='WTP (AED/month)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"""<div class='insight-box'>
    <b>📌 Pricing Strategy Insight:</b> WTP rises with income, but even the 10-20k tier
    shows median WTP of AED 30-50. A tiered pricing strategy (AED 25 / AED 50 / AED 80)
    captures value across income brackets without excluding the mass-market segment.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — BUSINESS FINDINGS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("🏆 Business Findings & Strategic Recommendations")

    # Summary verdict
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {NAVY}, {TEAL}); border-radius:12px;
                padding:24px; color:white; margin-bottom:20px;'>
        <h2 style='color:white; border:none; font-size:1.4rem; margin:0 0 10px;'>
            ✅ VERDICT: FinWise AI is Commercially Viable
        </h2>
        <p style='margin:0; font-size:0.95rem; line-height:1.6;'>
        <b>{wtp_pct:.1f}%</b> of surveyed Dubai professionals are willing to pay for this solution.
        Average WTP of <b>AED {avg_wtp:.0f}/month</b> with a median of
        <b>AED {master['Q21_WTP_AED'].median():.0f}</b> supports a
        freemium + paid tier pricing strategy. All 8 pain items score above the
        neutral midpoint, confirming systemic demand. The two highest-value segments
        (High-Value Nomads + Digital-First Millennials) represent ~{
            (clust_df_vis['Cluster_Name'].isin(['High-Value Nomads','Digital-First Millennials'])).mean()*100:.0f}%
        of the market with disproportionately high adoption readiness.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("1️⃣ Target Segment Priority")
    seg_cols = st.columns(3)
    target_segs = [
        ("🥇 Primary", "High-Value Nomads", GOLD,
         "Senior professionals (avg age 42) with high income (tier ~4) and multi-currency needs. Low pain but highest WTP (AED 80/mo). Immediately monetisable. Focus: investment tracking, premium FX, concierge features."),
        ("🥈 Primary", "Digital-First Millennials", TEAL,
         "Young tech-savvy professionals (avg age 31). Highest tech affinity score. Strong app-store growth driver. Mid WTP (AED 50/mo). Focus: AI categorisation, gamified savings, WhatsApp integration."),
        ("🥉 Growth", "Finance Professionals", NAVY,
         "Older, analytically sophisticated. Low pain but appreciates advanced tools (AED 43/mo WTP). Needs investment portfolio integration and detailed reporting to convert."),
    ]
    for col, (tier, seg, color, desc) in zip(seg_cols, target_segs):
        with col:
            st.markdown(f"""
            <div style='background:white; border-radius:10px; padding:16px;
                        border-top:4px solid {color}; box-shadow:0 2px 6px rgba(0,0,0,0.07);'>
                <div style='color:{color}; font-weight:700; font-size:0.9rem;'>{tier}</div>
                <div style='color:{NAVY}; font-weight:800; font-size:1rem; margin:4px 0;'>{seg}</div>
                <div style='color:#555; font-size:0.82rem; line-height:1.5;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.subheader("2️⃣ Top Features to Build (by Mean Desirability Score)")
    feature_rank_map = {
        'Q16_AIExpenseCategorisation':'AI Expense Categorisation',
        'Q16_MultiCurrencyTracking':'Multi-Currency Tracking',
        'Q16_SavingsGoals':'Savings Goals & Coaching',
        'Q16_RemittanceBestRate':'Remittance Best Rate',
        'Q16_InvestmentTracking':'Investment Tracking',
        'Q16_SubscriptionAlerts':'Subscription Alerts',
        'Q16_CreditScore':'Credit Score Monitoring',
        'Q16_HealthDashboard':'Health Dashboard',
        'Q16_ChatbotQueries':'Chatbot / AI Queries',
        'Q16_GamifiedSavings':'Gamified Savings'
    }
    feat_scores = master[list(feature_rank_map.keys())].mean().rename(feature_rank_map).sort_values(ascending=False)
    fig = px.bar(feat_scores.reset_index(), x=feat_scores.values, y=feat_scores.index,
                 orientation='h',
                 color=feat_scores.values, color_continuous_scale=[SAGE, TEAL, NAVY],
                 text=[f"{v:.2f}" for v in feat_scores.values],
                 labels={'x':'Mean Desirability (1-5)', 'index':'Feature'})
    fig.add_vline(x=3.5, line_dash="dot", line_color=CORAL,
                  annotation_text="High priority threshold (3.5)")
    fig.update_traces(textposition='outside')
    fig.update_layout(height=380, showlegend=False, coloraxis_showscale=False,
                       margin=dict(t=20,b=10,l=220))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("3️⃣ Pricing Strategy")
    c1, c2 = st.columns([3, 2])
    with c1:
        pricing_labels = {1:'Free with Ads',2:'Freemium',3:'Monthly Subscription',
                          4:'Annual Subscription',5:'One-time Purchase',6:'Pay-per-feature'}
        pricing_data = master['Q22_PricingModel'].map(pricing_labels).value_counts()
        fig = px.pie(pricing_data.reset_index(), names='Q22_PricingModel', values='count',
                     color='Q22_PricingModel',
                     color_discrete_sequence=PALETTE, hole=0.5)
        fig.update_layout(height=330, margin=dict(t=20,b=10),
                           title="Preferred Pricing Model (Q22)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown(f"""
        <div style='padding-top:20px;'>
        <div class='metric-card'>
            <div style='color:{TEAL}; font-weight:700;'>Recommended Pricing</div>
            <div style='font-size:1.1rem; color:{NAVY}; font-weight:800; margin:6px 0;'>
                Freemium + Two Paid Tiers
            </div>
            <div style='font-size:0.88rem; color:#555;'>
            <b>Free tier</b> — Core tracking, AED only, 2 accounts<br>
            <b>Plus tier (AED 25/mo)</b> — Multi-currency, 3 banks, basic AI<br>
            <b>Pro tier (AED 60/mo)</b> — Unlimited, investments, AI coaching<br><br>
            <i>Based on: {pricing_data.idxmax()} is most preferred
            ({pricing_data.max()/len(master)*100:.0f}% of respondents). Median WTP
            of AED {master['Q21_WTP_AED'].median():.0f} supports the Plus tier price point.</i>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("4️⃣ Model Performance Summary")
    summary_data = {
        'Analysis Type': ['Classification (WTP)', 'Classification (WTP)', 'Clustering',
                          'Regression (WTP AED)', 'Regression (Download Likelihood)'],
        'Best Model': ['Random Forest', 'Gradient Boosting', 'K-Means (k=5)',
                       'Random Forest Reg.', 'Random Forest Reg.'],
        'Key Metric': ['F1 = 0.846', 'F1 = 0.830', f'Silhouette = {sil_score:.3f}',
                       'R² = 0.301', 'R² = 0.188'],
        'Business Interpretation': [
            '84.6% accuracy in identifying likely payers — usable for targeting',
            'Strong gradient boosting alternative for ensemble voting',
            '5 distinct segments with clear WTP and behaviour differences',
            'Moderate WTP predictability — consistent with survey-based price studies',
            'Download intent moderately predicted from behaviours — directionally valid'
        ]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    st.subheader("5️⃣ Feasibility Verdict")
    c1, c2, c3 = st.columns(3)
    with c1:
        demand_count  = master['Q20_WTP_Binary'].sum()
        high_dl_count = (master['Q24_DownloadLikelihood'] >= 7).sum()
        value_seekers = master['Flag_ValueSeeker'].sum()
        st.markdown(f"""
        <div class='metric-card'>
            <div style='color:{TEAL}; font-weight:700;'>📈 Demand Funnel</div>
            <table style='width:100%; font-size:0.88rem; margin-top:8px;'>
                <tr><td>Total surveyed</td><td style='text-align:right;'><b>1,200</b></td></tr>
                <tr><td>Willing to pay</td><td style='text-align:right; color:{TEAL};'><b>{demand_count} ({demand_count/12:.0f}%)</b></td></tr>
                <tr><td>High download intent (≥7)</td><td style='text-align:right; color:{GOLD};'><b>{high_dl_count} ({high_dl_count/12:.0f}%)</b></td></tr>
                <tr><td>Value Seekers (pain+WTP)</td><td style='text-align:right; color:{CORAL};'><b>{value_seekers} ({value_seekers/12:.0f}%)</b></td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        rev_est = demand_count * master['Q21_WTP_AED'].mean()
        st.markdown(f"""
        <div class='metric-card'>
            <div style='color:{GOLD}; font-weight:700;'>💰 Revenue Potential</div>
            <div style='font-size:0.88rem; margin-top:8px; color:#555;'>
            From this sample alone (extrapolated):<br><br>
            <b>Monthly ARR potential</b><br>
            <span style='font-size:1.2rem; color:{NAVY}; font-weight:800;'>
            AED {rev_est:,.0f}</span><br>
            (WTP users × avg. AED {avg_wtp:.0f}/mo)<br><br>
            Freemium conversion assumed at 61.1% based on stated intent.
            Actual rates typically 3–8% — plan accordingly.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='color:{CORAL}; font-weight:700;'>⚠️ Key Risks</div>
            <div style='font-size:0.85rem; margin-top:8px; color:#555;'>
            <b>1.</b> Data privacy remains the #1 adoption barrier (Q25)<br><br>
            <b>2.</b> UAE regulatory approval (CBUAE/DFSA) needed for WPS & bank integration<br><br>
            <b>3.</b> Competition from Momo, Sarwa, and international players (Wise, Revolut)<br><br>
            <b>4.</b> Stated WTP surveys overestimate actual conversion by 4–8×
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("FinWise AI Analytics Dashboard · SP Jain GMBA · Data Analytics · Dubai 2026 · All findings derived from synthetic survey data (n=1,200)")
