import streamlit as st
import pandas as pd
import numpy as np
import html
import re
from datetime import datetime

# 1. Page Configuration
st.set_page_config(page_title="SerGene Strategic Intelligence", layout="wide")

# --- BI MAPPINGS ---
MODALITY_GROUPS = {
    "Cell Therapy": ["CAR-T", "TCR", "TILs", "NK Cells", "Tregs", "MSCs", "iPSCs", "gamma delta T cells", "γδ T cells", "Cell Therapy"],
    "Gene Therapy/Editing": ["CRISPR", "Base Editing", "Prime Editing", "Gene Editing", "Gene Therapy"],
    "RNA Therapeutics": ["mRNA", "siRNA", "RNAi", "miRNA", "ASO", "Antisense", "Aptamer", "RNA"],
    "Biologics": ["Antibody", "Bispecific", "ADC", "Multi-specific", "Peptide", "Biologics"],
    "Small Molecule": ["Small Molecule", "Protein Degrader", "Oral"]
}

def parse_currency(val_str):
    if not val_str or pd.isna(val_str) or val_str == "" or val_str == "nan":
        return 0.0
    try:
        match = re.search(r'([\d.]+)\s?([BMbm])', str(val_str))
        if match:
            num = float(match.group(1))
            unit = match.group(2).upper()
            return num * 1000 if unit == 'B' else num
        return 0.0
    except: return 0.0

# 2. Refined Styles
st.markdown("""
    <style>
    .deal-card {
        background-color: white;
        padding: 2.25rem;
        border-radius: 1.5rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
    .parent-tag {
        background-color: #eff6ff;
        color: #1e40af;
        padding: 0.3rem 0.8rem;
        border-radius: 0.6rem;
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        border: 1px solid #bfdbfe;
        letter-spacing: 0.05em;
    }
    .source-link { color: #3b82f6; text-decoration: none; font-weight: 800; }
    .source-link:hover { text-decoration: underline; color: #2563eb; }
    .tag {
        display: inline-block;
        background-color: #f8fafc;
        color: #64748b;
        padding: 0.25rem 0.7rem;
        border-radius: 0.6rem;
        font-size: 0.7rem;
        font-weight: 700;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid #e2e8f0;
        text-transform: uppercase;
    }
    .ratio-bar-bg {
        height: 10px;
        background-color: #f1f5f9;
        border-radius: 5px;
        margin-top: 8px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load & Refine Data
@st.cache_data
def load_and_refine_data():
    df = pd.read_feather("sg_intel_assets.arrow") 
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(by='Date', ascending=False)
    
    refined_rows = []
    for _, row in df.iterrows():
        tags = row.get('ModalityTags', [])
        if not isinstance(tags, (list, np.ndarray)): tags = []
        
        parent = "Other"
        for p_mod, keywords in MODALITY_GROUPS.items():
            if any(tag in keywords for tag in tags):
                parent = p_mod
                break
        
        val_m = parse_currency(row.get('DealValue', ''))
        up_m = parse_currency(row.get('Upfront', ''))
        ratio = (up_m / val_m) if val_m > 0 else 0.0

        refined_rows.append({
            'ID': row.get('ID'),
            'Date': row.get('Date'),
            'ParentModality': parent,
            'SubModalities': tags,
            'TotalValueM': val_m,
            'UpfrontRatio': ratio,
            'DisplayValue': row.get('DealValue', 'N/A'),
            'PartnerA': row.get('PartnerA', 'N/A'),
            'PartnerB': row.get('PartnerB', 'N/A'),
            'Insight': row.get('Insight', ''),
            'Title': row.get('Title', ''),
            'Summary': row.get('Summary', ''),
            'Link': row.get('Link', '#')
        })
    return pd.DataFrame(refined_rows)

try:
    df = load_and_refine_data()

    # 4. Sidebar Filters
    st.sidebar.title("SerGene Intelligence")
    
    min_date = df['Date'].min().to_pydatetime()
    max_date = df['Date'].max().to_pydatetime()
    selected_dates = st.sidebar.date_input("Date Range", value=(min_date, max_date))

    all_parents = ["All"] + sorted(df['ParentModality'].unique().tolist())
    selected_parent = st.sidebar.selectbox("Broad Modality", all_parents)
    
    all_tags = sorted(list(set([t for sublist in df['SubModalities'] for t in sublist])))
    selected_subs = st.sidebar.multiselect("Specific Cell Types / Platforms", all_tags)

    # 5. Apply Filters
    filtered_df = df.copy()
    if isinstance(selected_dates, (list, tuple)) and len(selected_dates) == 2:
        start_d, end_d = selected_dates
        filtered_df = filtered_df[(filtered_df['Date'].dt.date >= start_d) & (filtered_df['Date'].dt.date <= end_d)]
    
    if selected_parent != "All":
        filtered_df = filtered_df[filtered_df['ParentModality'] == selected_parent]
    
    if selected_subs:
        filtered_df = filtered_df[filtered_df['SubModalities'].apply(lambda x: any(s in x for s in selected_subs))]

    # 6. Dashboard Header
    st.title("Strategic Deal Stream")
    m1, m2, m3 = st.columns(3)
    m1.metric("Active Deals", len(filtered_df))
    m2.metric("Market Volume", f"${filtered_df['TotalValueM'].sum()/1000:.1f}B")
    
    valid_ratios = filtered_df[filtered_df['UpfrontRatio'] > 0]['UpfrontRatio']
    avg_r = valid_ratios.mean() if not valid_ratios.empty else 0
    m3.metric("Avg. Upfront Ratio", f"{avg_r:.1%}")

    # 7. Card Template (Separated to avoid SyntaxErrors)
    CARD_TEMPLATE = """
    <div class="deal-card">
        <div style="display: flex; justify-content: space-between; align-items: start; gap: 2rem;">
            <div style="flex: 2;">
                <span class="parent-tag">{parent_mod}</span>
                <h2 style="margin-top: 1rem;">
                    <a href="{link}" target="_blank" class="source-link">{insight}</a>
                </h2>
                <div style="font-weight: 700; color: #0f172a; font-size: 1.15rem; margin-bottom: 1rem;">{title}</div>
                <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">{summary}</p>
                <div style="margin-top: 1.5rem;">{tags}</div>
            </div>
            <div style="flex: 1; border-left: 2px solid #f1f5f9; padding-left: 2rem;">
                <div>
                    <p style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase;">Total Deal Value</p>
                    <p style="font-size: 1.8rem; font-weight: 900; color: #059669; margin: 0;">{value}</p>
                </div>
                <div style="margin-top: 1.5rem;">
                    <p style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase;">Cash Upfront Ratio ({ratio_pct}%)</p>
                    <div class="ratio-bar-bg">
                        <div style="height:10px; width:{ratio_pct}%; background-color:{ratio_color}; border-radius:5px;"></div>
                    </div>
                </div>
                <div style="margin-top: 1.5rem;">
                    <p style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase;">Partners</p>
                    <p style="font-weight: 800; color: #0f172a; font-size: 1.15rem; margin: 0;">{pA}</p>
                    <p style="color: #64748b; font-size: 0.85rem;">{pB}</p>
                </div>
                <div style="margin-top: 2rem;">
                    <a href="{link}" target="_blank" style="text-decoration: none; color: white; background-color: #3b82f6; padding: 0.7rem 1.2rem; border-radius: 0.75rem; font-size: 0.8rem; font-weight: 800;">View Original Source</a>
                </div>
            </div>
        </div>
    </div>
    """

    for _, row in filtered_df.iterrows():
        r_pct = round(row['UpfrontRatio'] * 100, 1)
        r_color = "#10b981" if r_pct > 25 else "#f59e0b"
        tags_html = "".join([f'<span class="tag">{html.escape(t)}</span>' for t in row['SubModalities']])
        
        # Inject values into the template
        st.markdown(CARD_TEMPLATE.format(
            parent_mod=row['ParentModality'],
            link=str(row['Link']),
            insight=html.escape(str(row['Insight'])),
            title=html.escape(str(row['Title'])),
            summary=html.escape(str(row['Summary'])),
            tags=tags_html,
            value=html.escape(str(row['DisplayValue'])),
            ratio_pct=r_pct,
            ratio_color=r_color,
            pA=html.escape(str(row['PartnerA'])),
            pB=html.escape(str(row['PartnerB']))
        ), unsafe_allow_html=True)

except Exception as e:
    st.error(f"BI Module Error: {e}")
