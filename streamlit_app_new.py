import streamlit as st
import pandas as pd
import numpy as np
import html
import re
import os
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="SerGene Strategic Intelligence",
    page_icon="🧬",
    layout="wide"
)

# --- BI HIERARCHY DEFINITION ---
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
    """Parses strings like '$1.5B' or '$50M' into float Millions (USD)."""
    if not val_str or pd.isna(val_str) or str(val_str).lower() in ["nan", "", "n/a"]:
        return 0.0
    try:
        clean_val = str(val_str).replace('$', '').replace(',', '').strip().lower()
        match = re.search(r'([\d.]+)\s?([bm])', clean_val)
        if match:
            num = float(match.group(1))
            unit = match.group(2).upper()
            if unit == 'B': return num * 1000
            return num
        return float(clean_val)
    except: return 0.0

# ==========================================
# 3. ADVANCED UI STYLING (CSS)
# ==========================================
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stApp { background-color: #f8fafc; }
    
    .deal-card {
        background-color: white; padding: 2.5rem; border-radius: 1.5rem;
        border: 1px solid #e2e8f0; margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
    .date-badge {
        color: #64748b; font-size: 0.75rem; font-weight: 800;
        text-transform: uppercase; letter-spacing: 0.075em; margin-bottom: 0.75rem;
    }
    .parent-tag {
        background-color: #eff6ff; color: #1e40af; padding: 0.35rem 0.85rem;
        border-radius: 0.75rem; font-size: 0.75rem; font-weight: 800;
        text-transform: uppercase; border: 1px solid #bfdbfe;
        display: inline-block; margin-bottom: 1rem;
    }
    .source-link { color: #2563eb; text-decoration: none; font-weight: 800; font-size: 1.5rem; }
    .source-link:hover { text-decoration: underline; color: #1d4ed8; }
    
    .summary-text { color: #475569; font-size: 0.95rem; line-height: 1.6; margin: 1.25rem 0; }
    
    .tag {
        display: inline-block; background-color: #f1f5f9; color: #475569;
        padding: 0.3rem 0.75rem; border-radius: 0.6rem; font-size: 0.7rem;
        font-weight: 700; margin-right: 0.5rem; margin-bottom: 0.5rem;
        border: 1px solid #e2e8f0; text-transform: uppercase;
    }
    .ratio-bar-container {
        height: 12px; background-color: #f1f5f9; border-radius: 6px;
        margin-top: 10px; overflow: hidden; border: 1px solid #e2e8f0;
    }
    .ai-strategy-box {
        background-color: #f0f9ff; border-left: 6px solid #0ea5e9;
        padding: 1.75rem; border-radius: 0.75rem; margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 4. DATA REFINERY
# ==========================================
@st.cache_data
def load_and_refine_data():
    if not os.path.exists("sg_intel_assets.arrow"):
        return pd.DataFrame()
    
    df = pd.read_feather("sg_intel_assets.arrow") 
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(by='Date', ascending=False)
    
    refined_rows = []
    for _, row in df.iterrows():
        # 1. Initialize tags from the existing list
        tags = []
        raw_tags = row.get('ModalityTags')
        if isinstance(raw_tags, (list, np.ndarray)):
            tags = [str(t).strip() for t in raw_tags if str(t).lower() != 'nan']
        
        # 2. THE UNIFIED HARVESTER
        for col_name in row.index:
            val = row[col_name]
            col_lower = str(col_name).lower().strip()
            
            # Check if the cell is 'Positive' (Number > 0 or string 'Yes')
            is_positive = False
            try:
                if float(val) > 0: is_positive = True
            except:
                if str(val).lower().strip() in ['yes', 'y', 'true', '1']: is_positive = True
            
            # If positive, we map the column name to a professional Tag
            if is_positive:
                if "msc" in col_lower:
                    tags.append("MSCs")
                
                elif "ipsc" in col_lower:
                    tags.append("iPSCs")
                
                # This line solves the Seagen (English) & Takeda (Greek) mismatch
                elif any(x in col_lower for x in ["gamma", "delta", "γ", "δ"]):
                    tags.append("γδ T cells")

        # 3. Final cleanup of tags
        tags = list(set([t for t in tags if t]))
        
        # 4. Determine Broad Modality for Filtering
        parent = "Other"
        norm_tags = [t.lower() for t in tags]
        for group_name, keywords in MODALITY_GROUPS.items():
            lower_kws = [k.lower() for k in keywords]
            if any(t in lower_kws for t in norm_tags):
                parent = group_name
                break
        
        # 5. Financials
        val_m = parse_currency(row.get('DealValue', ''))
        up_m = parse_currency(row.get('Upfront', ''))
        ratio = (up_m / val_m) if val_m > 0 else 0.0

        refined_rows.append({
            'ID': row.get('ID', 'N/A'),
            'Date': row.get('Date'),
            'DisplayDate': row.get('Date').strftime('%b %d, %Y') if pd.notnull(row.get('Date')) else "N/A",
            'ParentModality': parent,
            'SubModalities': tags, 
            'TA': row.get('TA', 'Other/General'),
            'Category': row.get('Category', 'N/A'),
            'TotalValueM': val_m,
            'UpfrontRatio': ratio,
            'DisplayValue': str(row.get('DealValue', 'N/A')),
            'PartnerA': str(row.get('PartnerA', 'N/A')),
            'PartnerB': str(row.get('PartnerB', 'N/A')),
            'Insight': str(row.get('Insight', '')),
            'Title': str(row.get('Title', '')),
            'Summary': str(row.get('Summary', '')),
            'Link': str(row.get('Link', '#'))
        })
    return pd.DataFrame(refined_rows)

# ==========================================
# 5. UI & FILTER LOGIC
# ==========================================
try:
    df_master = load_and_refine_data()
    if df_master.empty:
        st.warning("Database not found or empty.")
        st.stop()

    # Sidebar
    st.sidebar.title("SerGene Intelligence")
    
    # 5.1 Date Range
    min_d = df_master['Date'].min().to_pydatetime()
    max_d = df_master['Date'].max().to_pydatetime()
    date_sel = st.sidebar.date_input("Date Range", value=(min_d, max_d))

    # 5.2 Modality Filters
    all_parents = ["All"] + sorted(df_master['ParentModality'].unique().tolist())
    sel_parent = st.sidebar.selectbox("Broad Modality", all_parents)
    
    all_subs = sorted(list(set([t for sub in df_master['SubModalities'] for t in sub])))
    sel_subs = st.sidebar.multiselect("Specific Sub-Modalities", all_subs)

    # 5.3 Search
    search_term = st.sidebar.text_input("🔍 Search Database")

    # Filter Application
    f_df = df_master.copy()
    if isinstance(date_sel, (list, tuple)) and len(date_sel) == 2:
        sd, ed = date_sel
        f_df = f_df[(f_df['Date'].dt.date >= sd) & (f_df['Date'].dt.date <= ed)]
    if sel_parent != "All":
        f_df = f_df[f_df['ParentModality'] == sel_parent]
    if sel_subs:
        f_df = f_df[f_df['SubModalities'].apply(lambda x: any(s in x for s in sel_subs))]
    if search_term:
        f_df = f_df[f_df['Insight'].str.contains(search_term, case=False) | f_df['Title'].str.contains(search_term, case=False)]

    # ==========================================
    # 6. DASHBOARD & ANALYTICS
    # ==========================================
    st.title("Strategic Deal Intelligence Stream")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Active Deals", len(f_df))
    m2.metric("Market Volume", f"${f_df['TotalValueM'].sum()/1000:.1f}B")
    valid_r = f_df[f_df['UpfrontRatio'] > 0]['UpfrontRatio']
    avg_r = valid_r.mean() if not valid_r.empty else 0
    m3.metric("Avg. Upfront Ratio", f"{avg_r:.1%}")

    # Market Visualizations
    st.divider()
    with st.expander("📈 Market Analytics Expandable View", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Modality Distribution**")
            st.bar_chart(f_df['ParentModality'].value_counts(), color="#3b82f6")
        with c2:
            st.write("**Top Therapeutic Areas**")
            st.bar_chart(f_df['TA'].value_counts(), color="#10b981")
        st.write("**Most Active Strategic Partners**")
        st.bar_chart(f_df['PartnerA'].value_counts().head(10), horizontal=True, color="#f59e0b")

    # AI Brief
    if st.button("🪄 Generate AI Market Brief"):
        st.markdown(f"""
            <div class="ai-strategy-box">
                <h3 style="margin-top:0;">🤖 SerGene AI Strategy Brief</h3>
                <p>Analyzing <strong>{len(f_df)} deals</strong>. Market leader in this view: 
                <strong>{f_df['PartnerA'].mode()[0] if not f_df.empty else 'N/A'}</strong>. 
                Focus Area: <strong>{f_df['TA'].mode()[0] if not f_df.empty else 'N/A'}</strong>.</p>
            </div>
        """, unsafe_allow_html=True)

    # ==========================================
    # 7. DEAL CARDS ENGINE
    # ==========================================
    # FIXED: Placeholders r_pct and r_color now match formatting args exactly
    CARD_TEMPLATE = """
    <div class="deal-card">
        <div style="display: flex; justify-content: space-between; align-items: start; gap: 2.5rem;">
            <div style="flex: 2;">
                <div class="date-badge">{d_date} | {ta} • {cat}</div>
                <span class="parent-tag">{p_mod}</span>
                <h2 style="margin-top: 1rem;">
                    <a href="{link}" target="_blank" class="source-link">{insight}</a>
                </h2>
                <div style="font-weight: 700; color: #0f172a; font-size: 1.1rem; margin-bottom: 0.5rem; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.5rem;">{title}</div>
                <p class="summary-text">{summary}</p>
                <div style="margin-top: 1.5rem;">{tags}</div>
            </div>
            <div style="flex: 1; border-left: 2px solid #f1f5f9; padding-left: 2.5rem; min-width: 280px;">
                <div style="margin-bottom: 2rem;">
                    <p style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase;">Total Deal Value</p>
                    <p style="font-size: 1.85rem; font-weight: 900; color: #059669; margin: 0;">{value}</p>
                </div>
                <div style="margin-bottom: 2rem;">
                    <p style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase;">Upfront Ratio ({r_pct}%)</p>
                    <div class="ratio-bar-container">
                        <div style="height:100%; width:{r_pct}%; background-color:{r_color}; border-radius:6px;"></div>
                    </div>
                </div>
                <div style="margin-bottom: 2rem;">
                    <p style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase;">Partners</p>
                    <p style="font-weight: 800; color: #0f172a; font-size: 1.15rem; margin: 0;">{pA}</p>
                    <p style="color: #64748b; font-size: 0.85rem; margin-top: 0.25rem;">{pB}</p>
                </div>
                <div style="margin-top: 2.5rem;">
                    <a href="{link}" target="_blank" style="text-decoration: none; color: white; background-color: #2563eb; padding: 0.75rem 1.5rem; border-radius: 0.75rem; font-size: 0.8rem; font-weight: 800; display: block; text-align: center;">View Full Intelligence Source</a>
                </div>
            </div>
        </div>
    </div>
    """

    for _, row in f_df.iterrows():
        rpct = round(row['UpfrontRatio'] * 100, 1)
        rcol = "#10b981" if rpct > 25 else "#f59e0b"
        tags_h = "".join([f'<span class="tag">{html.escape(str(t))}</span>' for t in row['SubModalities']])
        
        st.markdown(CARD_TEMPLATE.format(
            d_date=row['DisplayDate'], ta=row['TA'], cat=row['Category'],
            p_mod=row['ParentModality'], link=row['Link'],
            insight=html.escape(row['Insight']), title=html.escape(row['Title']),
            summary=html.escape(row['Summary']), tags=tags_h,
            value=html.escape(row['DisplayValue']), r_pct=rpct, r_color=rcol,
            pA=html.escape(row['PartnerA']), pB=html.escape(row['PartnerB'])
        ), unsafe_allow_html=True)

except Exception as e:
    st.error(f"SerGene BI Module Initialization Error: {e}")
