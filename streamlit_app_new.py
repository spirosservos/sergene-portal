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

def smart_format_company(name):
    if not name or pd.isna(name) or str(name).lower() == 'nan': return "N/A"
    text = str(name).strip()
    words = text.split()
    formatted_words = [word.capitalize() if word.islower() else word for word in words]
    return " ".join(formatted_words)

# ==========================================
# 3. PREMIUM UI STYLING
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
    .blurred-card {
        filter: blur(8px); opacity: 0.5; pointer-events: none; user-select: none;
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
    .cta-banner {
        background-color: #fef2f2; border: 2px dashed #ef4444; 
        padding: 2.5rem; border-radius: 1.5rem; text-align: center; 
        margin-top: 2rem; margin-bottom: 5rem;
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
        raw_tags = row.get('ModalityTags')
        tags = []
        if isinstance(raw_tags, (list, np.ndarray)):
            tags = [re.sub(r'([a-z])([A-Z])', r'\1 \2', str(t)).strip() for t in raw_tags]
        
        for col_name in row.index:
            val = row[col_name]
            col_l = str(col_name).lower().strip()
            is_hit = False
            try:
                if float(val) > 0: is_hit = True
            except:
                if str(val).lower().strip() in ['yes', 'y', 'true', '1']: is_hit = True
            
            if is_hit:
                if "msc" in col_l: tags.append("MSCs")
                elif "ipsc" in col_l: tags.append("iPSCs")
                elif any(x in col_l for x in ["gamma", "delta", "γ", "δ"]): tags.append("γδ T cells")

        tags = list(set([t for t in tags if t and str(t).lower() != 'nan']))
        parent = "Other"
        norm_tags = [t.lower() for t in tags]
        for group_name, keywords in MODALITY_GROUPS.items():
            lower_kws = [k.lower() for k in keywords]
            if any(t in lower_kws for t in norm_tags):
                parent = group_name
                break
        
        val_m = parse_currency(row.get('DealValue', ''))
        up_m = parse_currency(row.get('Upfront', ''))
        ratio = (up_m / val_m) if val_m > 0 else 0.0

        refined_rows.append({
            'ID': row.get('ID', 'N/A'),
            'Date': row.get('Date'),
            'DisplayDate': row.get('Date').strftime('%b %d, %Y') if pd.notnull(row.get('Date')) else "N/A",
            'ParentModality': parent,
            'SubModalities': tags,
            'TA': str(row.get('TA', 'Other/General')).strip(),
            'Stage': str(row.get('Stage', 'Pre-clinical')).strip(),
            'Category': row.get('Category', 'N/A'),
            'TotalValueM': val_m,
            'UpfrontRatio': ratio,
            'DisplayValue': str(row.get('DealValue', 'N/A')),
            'PartnerA': smart_format_company(row.get('PartnerA')),
            'PartnerB': smart_format_company(row.get('PartnerB')),
            'Insight': str(row.get('Insight', '')),
            'Title': str(row.get('Title', '')),
            'Summary': str(row.get('Summary', '')),
            'Link': str(row.get('Link', '#'))
        })
    return pd.DataFrame(refined_rows)

# ==========================================
# 5. UI & FILTER ENGINE
# ==========================================
try:
    df_master = load_and_refine_data()
    
    # --- AUTH SETTINGS ---
    is_authenticated = False 
    GLOBAL_PREVIEW_LIMIT = 5
    BLUR_LIMIT = 3

    st.sidebar.title("SerGene Intelligence")
    
    # Sidebar Filters (Always use df_master to show all options)
    date_sel = st.sidebar.date_input("Date Range", value=(df_master['Date'].min(), df_master['Date'].max()))
    sel_tas = st.sidebar.multiselect("Therapeutic Area", sorted(df_master['TA'].unique().tolist()))
    sel_stages = st.sidebar.multiselect("Development Stage", sorted(df_master['Stage'].unique().tolist()))
    sel_parent = st.sidebar.selectbox("Broad Modality", ["All"] + sorted(df_master['ParentModality'].unique().tolist()))
    
    # Sub-Modality Harvesting for Filter
    all_subs = sorted(list(set([t for sub in df_master['SubModalities'] for t in sub])))
    sel_subs = st.sidebar.multiselect("Specific Platforms / Cell Types", all_subs)
    
    search_term = st.sidebar.text_input("🔍 Search Database")

    # 5.1 APPLY GLOBAL FILTERS (TO FULL DATABASE)
    # We do this so the Charts and Metrics show the REAL market volume
    f_df = df_master.copy()
    if isinstance(date_sel, (list, tuple)) and len(date_sel) == 2:
        sd, ed = date_sel
        f_df = f_df[(f_df['Date'].dt.date >= sd) & (f_df['Date'].dt.date <= ed)]
    if sel_tas: f_df = f_df[f_df['TA'].isin(sel_tas)]
    if sel_stages: f_df = f_df[f_df['Stage'].isin(sel_stages)]
    if sel_parent != "All": f_df = f_df[f_df['ParentModality'] == sel_parent]
    if sel_subs: f_df = f_df[f_df['SubModalities'].apply(lambda x: any(s in x for s in sel_subs))]
    if search_term: f_df = f_df[f_df['Insight'].str.contains(search_term, case=False) | f_df['Title'].str.contains(search_term, case=False)]

    # ==========================================
    # 6. DASHBOARD & ANALYTICS (Hidden Gem)
    # ==========================================
    st.title("Strategic Deal Intelligence Stream")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Strategic Assets", len(f_df)) # Shows full filtered depth
    m2.metric("Market Volume Analysed", f"${f_df['TotalValueM'].sum()/1000:.1f}B")
    
    valid_r = f_df[f_df['UpfrontRatio'] > 0]['UpfrontRatio']
    avg_r = valid_r.mean() if not valid_r.empty else 0
    m3.metric("Avg. Upfront Ratio", f"{avg_r:.1%}")

    # Charts: These now accurately show trends for the WHOLE selection
    st.divider()
    with st.expander("📈 Market Trends & Competitive Landscape", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**Modality Mix**")
            st.bar_chart(f_df['ParentModality'].value_counts(), color="#3b82f6")
        with c2:
            st.write("**Therapeutic Focus**")
            st.bar_chart(f_df['TA'].value_counts(), color="#10b981")
        with c3:
            st.write("**Development Stage**")
            st.bar_chart(f_df['Stage'].value_counts(), color="#6366f1")

    # ==========================================
    # 7. TEASER SPLICING & RENDERING
    # ==========================================
    
    # Final Slicing for the Stream
    if not is_authenticated:
        # We only show the Top 5 of whatever is filtered
        visible_df = f_df.head(GLOBAL_PREVIEW_LIMIT)
        # We blur the NEXT 3 from the master list (or filtered list)
        blurred_df = f_df.iloc[GLOBAL_PREVIEW_LIMIT : GLOBAL_PREVIEW_LIMIT + BLUR_LIMIT]
    else:
        visible_df = f_df
        blurred_df = pd.DataFrame()

    # 7.1 HANDLING EMPTY RESULTS
    if not is_authenticated and visible_df.empty and not f_df.empty:
        st.warning("⚠️ These specific results are reserved for Premium Subscribers.")
        st.info("The graphics above confirm active deal flow in this segment, but the individual records are currently locked.")

    CARD_HTML = """
    <div class="deal-card {extra_class}">
        <div style="display: flex; justify-content: space-between; align-items: start; gap: 2.5rem;">
            <div style="flex: 2;">
                <div class="date-badge">{d_date} | {ta} • {stage}</div>
                <span class="parent-tag">{p_mod}</span>
                <h2 style="margin-top: 1rem;"><a href="{link}" target="_blank" class="source-link">{insight}</a></h2>
                <div style="font-weight: 700; color: #0f172a; font-size: 1.1rem; margin-bottom: 0.5rem; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.5rem;">{title}</div>
                <p class="summary-text">{summary}</p>
                <div style="margin-top: 1rem;">{tags}</div>
            </div>
            <div style="flex: 1; border-left: 2px solid #f1f5f9; padding-left: 2.5rem; min-width: 280px;">
                <p style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase;">Total Deal Value</p>
                <p style="font-size: 1.85rem; font-weight: 900; color: #059669; margin: 0;">{value}</p>
                <div style="margin-top: 2rem;">
                    <p style="font-weight: 800; color: #0f172a; font-size: 1.15rem; margin: 0;">{pA}</p>
                    <p style="color: #64748b; font-size: 0.85rem;">{pB}</p>
                </div>
            </div>
        </div>
    </div>
    """

    # Render Visible
    for _, row in visible_df.iterrows():
        tags_h = "".join([f'<span class="tag">{html.escape(str(t))}</span>' for t in row['SubModalities']])
        st.markdown(CARD_HTML.format(
            extra_class="", d_date=row['DisplayDate'], ta=row['TA'], stage=row['Stage'],
            p_mod=row['ParentModality'], link=row['Link'], insight=html.escape(row['Insight']),
            title=html.escape(row['Title']), summary=html.escape(row['Summary']), tags=tags_h,
            value=html.escape(row['DisplayValue']), pA=html.escape(row['PartnerA']), pB=html.escape(row['PartnerB'])
        ), unsafe_allow_html=True)

    # Render Blurred
    if not is_authenticated:
        for _, row in blurred_df.iterrows():
            st.markdown(CARD_HTML.format(
                extra_class="blurred-card", d_date=row['DisplayDate'], ta=row['TA'], stage=row['Stage'],
                p_mod=row['ParentModality'], link="#", insight="[LOCKED INSIGHT]",
                title=html.escape(row['Title']), summary=html.escape(row['Summary']), tags="",
                value="$$$,$$$", pA="[LOCKED]", pB="[LOCKED]"
            ), unsafe_allow_html=True)

        # 7.2 CTA BANNER
        st.markdown(f"""
            <div class="cta-banner">
                <h2 style="color: #991b1b; margin-top: 0;">🔒 Strategic Intelligence Access</h2>
                <p style="font-size: 1.1rem; color: #b91c1c; margin-bottom: 1.5rem;">
                    Analyze the full <b>{len(df_master)} proprietary records</b>. Unlock TA/Stage filters 
                    and the <b>AI Executive Brief</b> generator.
                </p>
                <a href="mailto:info@sergene.com?subject=Strategic Access" 
                   style="text-decoration: none; color: white; background-color: #ef4444; 
                   padding: 1rem 2rem; border-radius: 0.75rem; font-weight: 800;">
                   Request Client Password
                </a>
            </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"BI Module Error: {e}")
