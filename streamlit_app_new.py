import streamlit as st
import pandas as pd
import numpy as np
import html
import re
import os
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION & THEMING
# ==========================================
st.set_page_config(
    page_title="SerGene Strategic Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- BI MAPPINGS (The "Brain" of the Modality Hierarchy) ---
MODALITY_GROUPS = {
    "Cell Therapy": ["CAR-T", "TCR", "TILs", "NK Cells", "Tregs", "MSCs", "iPSCs", "gamma delta T cells", "γδ T cells", "Cell Therapy"],
    "Gene Therapy/Editing": ["CRISPR", "Base Editing", "Prime Editing", "Gene Editing", "Gene Therapy"],
    "RNA Therapeutics": ["mRNA", "siRNA", "RNAi", "miRNA", "ASO", "Antisense", "Aptamer", "RNA"],
    "Biologics": ["Antibody", "Bispecific", "ADC", "Multi-specific", "Peptide", "Biologics"],
    "Small Molecule": ["Small Molecule", "Protein Degrader", "Oral"]
}

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================
def parse_currency(val_str):
    """
    Normalizes mixed currency strings (e.g., '$1.5B', '$50M', '750 million') 
    into a numeric float representing Millions (USD).
    """
    if not val_str or pd.isna(val_str) or str(val_str).lower() in ["nan", "", "n/a"]:
        return 0.0
    try:
        # Standardize formatting
        clean_val = str(val_str).replace('$', '').replace(',', '').strip().lower()
        match = re.search(r'([\d.]+)\s?([bm])', clean_val)
        if match:
            num = float(match.group(1))
            unit = match.group(2).upper()
            if unit == 'B':
                return num * 1000
            return num
        return float(clean_val)
    except:
        return 0.0

# ==========================================
# 3. PREMIUM UI STYLING (CSS)
# ==========================================
st.markdown("""
    <style>
    /* Global Styles */
    .main { background-color: #f8fafc; }
    .stApp { background-color: #f8fafc; }
    
    /* Card Container */
    .deal-card {
        background-color: white;
        padding: 2.5rem;
        border-radius: 1.5rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .deal-card:hover {
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        border-color: #cbd5e1;
    }
    
    /* Typography */
    .date-badge {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.075em;
        margin-bottom: 0.75rem;
    }
    .parent-tag {
        background-color: #eff6ff;
        color: #1e40af;
        padding: 0.35rem 0.85rem;
        border-radius: 0.75rem;
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        border: 1px solid #bfdbfe;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .source-link {
        color: #2563eb;
        text-decoration: none;
        font-weight: 800;
        font-size: 1.5rem;
        line-height: 1.2;
    }
    .source-link:hover { text-decoration: underline; color: #1d4ed8; }
    
    .summary-text {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.6;
        margin: 1.25rem 0;
    }
    
    /* Modality Tags */
    .tag {
        display: inline-block;
        background-color: #f1f5f9;
        color: #475569;
        padding: 0.3rem 0.75rem;
        border-radius: 0.6rem;
        font-size: 0.7rem;
        font-weight: 700;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid #e2e8f0;
        text-transform: uppercase;
    }
    
    /* Financial Bar */
    .ratio-bar-container {
        height: 12px;
        background-color: #f1f5f9;
        border-radius: 6px;
        margin-top: 10px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }
    
    /* AI Synthesis Box */
    .ai-strategy-box {
        background-color: #f0f9ff;
        border-left: 6px solid #0ea5e9;
        padding: 1.75rem;
        border-radius: 0.75rem;
        margin: 2rem 0;
    }
    
    /* Lock Screen for Non-Premium */
    .lock-screen {
        background-color: #fff1f2;
        color: #be123c;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        border: 1px solid #fecdd3;
        font-weight: 700;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 4. DATA PIPELINE (REFINE & TRANSFORM)
# ==========================================
@st.cache_data
def load_and_refine_data():
    """
    Loads raw Arrow data and performs the 'Chef' transformation:
    Normalizes modalities, parses currency, and formats dates.
    """
    if not os.path.exists("sg_intel_assets.arrow"):
        return pd.DataFrame()

    df = pd.read_feather("sg_intel_assets.arrow") 
    
    # Pre-processing Dates
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(by='Date', ascending=False)
    
    refined_rows = []
    for _, row in df.iterrows():
        # A. MODALITY DRILL-DOWN
        tags = row.get('ModalityTags', [])
        if not isinstance(tags, (list, np.ndarray)): 
            tags = []
        
        # Determine Broad Category
        parent = "Other"
        normalized_tags = [str(t).lower().strip() for t in tags]
        for group_name, group_keywords in MODALITY_GROUPS.items():
            lower_kws = [k.lower().strip() for k in group_keywords]
            if any(t in lower_kws for t in normalized_tags):
                parent = group_name
                break
        
        # B. FINANCIAL NORMALIZATION
        val_m = parse_currency(row.get('DealValue', ''))
        up_m = parse_currency(row.get('Upfront', ''))
        ratio = (up_m / val_m) if val_m > 0 else 0.0

        # C. RECORD BUILDING
        refined_rows.append({
            'ID': row.get('ID', 'N/A'),
            'Date': row.get('Date'),
            'DisplayDate': row.get('Date').strftime('%b %d, %Y') if pd.notnull(row.get('Date')) else "N/A",
            'ParentModality': parent,
            'SubModalities': tags,
            'TA': row.get('TA', 'Other/General'),
            'Category': row.get('Category', 'N/A'),
            'TotalValueM': val_m,
            'UpfrontM': up_m,
            'UpfrontRatio': ratio,
            'DisplayValue': str(row.get('DealValue', 'N/A')),
            'PartnerA': str(row.get('PartnerA', 'N/A')),
            'PartnerB': str(row.get('PartnerB', 'N/A')),
            'Score': row.get('Score', 0),
            'Insight': str(row.get('Insight', '')),
            'Title': str(row.get('Title', '')),
            'Summary': str(row.get('Summary', '')),
            'Link': str(row.get('Link', '#'))
        })
    
    return pd.DataFrame(refined_rows)

# ==========================================
# 5. APP LOGIC & INTERFACE
# ==========================================
try:
    df_master = load_and_refine_data()

    # --- SIDEBAR: AUTH & FILTERS ---
    st.sidebar.title("SerGene Intelligence")
    
    # 5.1 Authentication Logic
    with st.sidebar.expander("🔐 Secure Client Access", expanded=True):
        try:
            SECRET_PWD = st.secrets["access_password"]
        except:
            SECRET_PWD = "SerGene2024"
        
        user_pwd = st.text_input("Enter Access Code", type="password")
        is_pro = (user_pwd == SECRET_PWD)
        if is_pro:
            st.success("Full Premium Access Active")
        else:
            st.info("Guest Mode: 20 Results Limit")

    st.sidebar.divider()

    # 5.2 Date Range Filter
    if not df_master.empty:
        min_d = df_master['Date'].min().to_pydatetime()
        max_d = df_master['Date'].max().to_pydatetime()
        date_sel = st.sidebar.date_input("Date Range", value=(min_d, max_d))
    
    # 5.3 Modality Filters
    all_parents = ["All"] + sorted(df_master['ParentModality'].unique().tolist())
    sel_parent = st.sidebar.selectbox("Broad Modality", all_parents)
    
    all_subs = sorted(list(set([t for sub in df_master['SubModalities'] for t in sub])))
    sel_subs = st.sidebar.multiselect("Specific Cell Types / Platforms", all_subs)

    # 5.4 Search Database
    search_term = st.sidebar.text_input("🔍 Search Competitive Intel")

    # --- 6. DATA FILTERING ENGINE ---
    filtered_df = df_master.copy()
    
    # Date Filter
    if isinstance(date_sel, (list, tuple)) and len(date_sel) == 2:
        sd, ed = date_sel
        filtered_df = filtered_df[(filtered_df['Date'].dt.date >= sd) & (filtered_df['Date'].dt.date <= ed)]
    
    # Modality Filter
    if sel_parent != "All":
        filtered_df = filtered_df[filtered_df['ParentModality'] == sel_parent]
    
    # Sub-Modality Filter
    if sel_subs:
        filtered_df = filtered_df[filtered_df['SubModalities'].apply(lambda x: any(s in x for s in sel_subs))]
    
    # Search Filter
    if search_term:
        filtered_df = filtered_df[
            filtered_df['Insight'].str.contains(search_term, case=False) | 
            filtered_df['Title'].str.contains(search_term, case=False) |
            filtered_df['PartnerA'].str.contains(search_term, case=False)
        ]

    # Enforcement of Access Levels
    display_df = filtered_df if is_pro else filtered_df.head(20)

    # --- 7. MAIN DASHBOARD CONTENT ---
    st.title("Strategic Deal Intelligence Stream")
    
    if not is_pro:
        st.markdown('<div class="lock-screen">🔒 PREVIEW MODE: Showing 20 most recent assets. Contact admin for full data access.</div>', unsafe_allow_html=True)

    # High-Level Metrics
    met1, met2, met3 = st.columns(3)
    met1.metric("Active Deal Flow", len(filtered_df))
    met2.metric("Filtered Market Vol.", f"${filtered_df['TotalValueM'].sum()/1000:.1f}B")
    
    v_ratios = filtered_df[filtered_df['UpfrontRatio'] > 0]['UpfrontRatio']
    a_ratio = v_ratios.mean() if not v_ratios.empty else 0
    met3.metric("Avg. Cash Intensity", f"{a_ratio:.1%}")

    # --- MARKET ANALYTICS (EXPANDER) ---
    st.divider()
    with st.expander("📊 Market Analytics & Landscape Mapping", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Investment by Modality**")
            st.bar_chart(filtered_df['ParentModality'].value_counts())
        with c2:
            st.write("**Therapeutic Area Priority**")
            st.bar_chart(filtered_df['TA'].value_counts())
        
        st.write("**Most Active Partners (Originators/Acquirers)**")
        st.bar_chart(filtered_df['PartnerA'].value_counts().head(10), horizontal=True)

    # --- AI STRATEGY BRIEF SECTION ---
    if st.button("🪄 Generate AI Strategic Executive Summary"):
        with st.spinner("Synthesizing market movements..."):
            st.markdown(f"""
                <div class="ai-strategy-box">
                    <h3 style="margin-top:0; color:#0369a1;">🤖 SerGene Executive Summary</h3>
                    <p>Current landscape analysis of <strong>{len(filtered_df)} assets</strong> shows a dominant focus in 
                    <strong>{filtered_df['TA'].mode()[0] if not filtered_df.empty else 'N/A'}</strong> indications. 
                    The modality focus is primarily on <strong>{sel_parent if sel_parent != 'All' else 'Multi-platform Technologies'}</strong>.</p>
                    <p>Financial intensity for this segment averages <strong>{a_ratio:.1%} upfront</strong>, 
                    signaling a market preference for <strong>{'risk-sharing milestone structures' if a_ratio < 0.2 else 'high-conviction cash acquisitions'}</strong>.</p>
                </div>
            """, unsafe_allow_html=True)

    # --- 8. THE DEAL CARDS LOOP ---
    # Define a clean HTML template using .format() to avoid curly-brace errors
    CARD_TEMPLATE = """
    <div class="deal-card">
        <div style="display: flex; justify-content: space-between; align-items: start; gap: 2.5rem;">
            <div style="flex: 2;">
                <div class="date-badge">{d_date} | {ta} • {cat}</div>
                <span class="parent-tag">{p_mod}</span>
                <h2 style="margin-top: 1rem;">
                    <a href="{link}" target="_blank" class="source-link">{insight}</a>
                </h2>
                <div style="font-weight: 700; color: #0f172a; font-size: 1.1rem; margin-bottom: 0.5rem; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.5rem;">
                    {title}
                </div>
                <p class="summary-text">{summary}</p>
                <div style="margin-top: 1.5rem;">{tags}</div>
            </div>
            
            <div style="flex: 1; border-left: 2px solid #f1f5f9; padding-left: 2.5rem; min-width: 280px;">
                <div style="margin-bottom: 2rem;">
                    <p style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em;">Total Deal Value</p>
                    <p style="font-size: 1.85rem; font-weight: 900; color: #059669; margin: 0;">{value}</p>
                </div>
                
                <div style="margin-bottom: 2rem;">
                    <p style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em;">Cash Upfront Ratio ({r_pct}%)</p>
                    <div class="ratio-bar-container">
                        <div style="height:100%; width:{r_pct}%; background-color:{r_color}; border-radius:6px;"></div>
                    </div>
                </div>
                
                <div style="margin-bottom: 2rem;">
                    <p style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em;">Primary Partner</p>
                    <p style="font-weight: 800; color: #0f172a; font-size: 1.15rem; margin: 0; line-height:1.2;">{pA}</p>
                    <p style="color: #64748b; font-size: 0.85rem; margin-top: 0.25rem;">{pB}</p>
                </div>
                
                <div style="margin-top: 2.5rem;">
                    <a href="{link}" target="_blank" style="text-decoration: none; color: white; background-color: #2563eb; padding: 0.75rem 1.5rem; border-radius: 0.75rem; font-size: 0.8rem; font-weight: 800; display: block; text-align: center;">View Original Intelligence Source</a>
                </div>
            </div>
        </div>
    </div>
    """

    for _, row in display_df.iterrows():
        # Visual color logic for the financial ratio
        r_pct = round(row['UpfrontRatio'] * 100, 1)
        r_color = "#10b981" if r_pct > 25 else "#f59e0b" # Green for solid deals, Orange for riskier ones
        
        # Build the tags string
        tags_html = "".join([f'<span class="tag">{html.escape(str(t))}</span>' for t in row['SubModalities']])
        
        # Inject the data into the card template
        st.markdown(CARD_TEMPLATE.format(
            d_date=row['DisplayDate'],
            ta=row['TA'],
            cat=row['Category'],
            p_mod=row['ParentModality'],
            link=row['Link'],
            insight=html.escape(row['Insight']),
            title=html.escape(row['Title']),
            summary=html.escape(row['Summary']),
            tags=tags_html,
            value=html.escape(row['DisplayValue']),
            ratio_pct=r_pct,
            ratio_color=r_color,
            pA=html.escape(row['PartnerA']),
            pB=html.escape(row['PartnerB'])
        ), unsafe_allow_html=True)

except Exception as e:
    st.error(f"⚠️ SerGene BI Module Initialization Error: {e}")
