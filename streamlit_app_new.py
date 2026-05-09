import streamlit as st
import pandas as pd
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
        padding: 2rem;
        border-radius: 1.25rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .parent-tag {
        background-color: #eff6ff;
        color: #1e40af;
        padding: 0.25rem 0.75rem;
        border-radius: 0.5rem;
        font-size: 0.7rem;
        font-weight: 800;
        text-transform: uppercase;
        border: 1px solid #bfdbfe;
    }
    .ratio-bar-bg {
        height: 10px;
        background-color: #f1f5f9;
        border-radius: 5px;
        margin-top: 8px;
    }
    .ratio-fill {
        height: 10px;
        border-radius: 5px;
    }
    .tag {
        display: inline-block;
        background-color: #f8fafc;
        color: #64748b;
        padding: 0.2rem 0.6rem;
        border-radius: 0.5rem;
        font-size: 0.65rem;
        font-weight: 700;
        margin-right: 0.4rem;
        border: 1px solid #e2e8f0;
        text-transform: uppercase;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load & Refine Data
@st.cache_data
def load_and_refine_data():
    # Load your specific Arrow file
    df = pd.read_feather("sg_intel_assets.arrow") 
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(by='Date', ascending=False)
    
    refined_rows = []
    for _, row in df.iterrows():
        # Modality Grouping
        tags = row.get('ModalityTags', [])
        parent = "Other"
        for p_mod, keywords in MODALITY_GROUPS.items():
            if any(tag in keywords for tag in tags):
                parent = p_mod
                break
        
        # Financial Ratios
        val_m = parse_currency(row.get('DealValue', ''))
        up_m = parse_currency(row.get('Upfront', ''))
        ratio = (up_m / val_m) if val_m > 0 else 0.0

        refined_rows.append({
            'ParentModality': parent,
            'SubModalities': tags,
            'TotalValueM': val_m,
            'UpfrontRatio': ratio,
            'DisplayValue': row.get('DealValue', 'N/A'),
            'PartnerA': row.get('PartnerA', 'N/A'),
            'PartnerB': row.get('PartnerB', 'N/A'),
            'Score': row.get('Score', 0),
            'Insight': row.get('Insight', ''),
            'Title': row.get('Title', ''),
            'Summary': row.get('Summary', ''),
            'Date': row.get('Date'),
            'Link': row.get('Link', '#')
        })
    return pd.DataFrame(refined_rows)

try:
    df = load_and_refine_data()

    # 4. Sidebar BI Filters
    st.sidebar.title("SerGene Intelligence")
    
    all_parents = ["All"] + sorted(df['ParentModality'].unique().tolist())
    selected_parent = st.sidebar.selectbox("Broad Modality", all_parents)
    
    max_val = int(df['TotalValueM'].max()) if not df.empty else 1000
    min_val_filter = st.sidebar.slider("Min Deal Value ($M)", 0, max_val, 0)

    # 5. Apply Filters
    filtered_df = df.copy()
    if selected_parent != "All":
        filtered_df = filtered_df[filtered_df['ParentModality'] == selected_parent]
    filtered_df = filtered_df[filtered_df['TotalValueM'] >= min_val_filter]

    # 6. Top Metrics
    st.title("Strategic Deal Stream")
    m1, m2, m3 = st.columns(3)
    m1.metric("Active Deals", len(filtered_df))
    m2.metric("Market Volume", f"${filtered_df['TotalValueM'].sum()/1000:.1f}B")
    
    valid_ratios = filtered_df[filtered_df['UpfrontRatio'] > 0]['UpfrontRatio']
    avg_r = valid_ratios.mean() if not valid_ratios.empty else 0
    m3.metric("Avg. Upfront Ratio", f"{avg_r:.1%}")

    # 7. Card Loop
    for _, row in filtered_df.iterrows():
        r_pct = row['UpfrontRatio'] * 100
        r_color = "#10b981" if r_pct > 25 else "#f59e0b"
        
        tags_html = "".join([f'<span class="tag">{html.escape(t)}</span>' for t in row['SubModalities']])
        
        # We build the HTML string carefully to avoid syntax errors
        card_html = f"""
        <div class="deal-card">
            <div style="display: flex; justify-content: space-between; align-items: start; gap: 2rem;">
                <div style="flex: 2;">
                    <span class="parent-tag">{row['ParentModality']}</span>
                    <h2 style="color: #3b82f6; font-size: 1.4rem; font-weight: 800; margin-top: 1rem;">{html.escape(str(row['Insight']))}</h2>
                    <div style="font-weight: 700; color: #0f172a; font-size: 0.95rem; margin-bottom: 0.75rem;">{html.escape(str(row['Title']))}</div>
                    <p style="color: #475569; font-size: 0.9rem; line-height: 1.5;">{html.escape(str(row['Summary']))}</p>
                    <div style="margin-top: 1.25rem;">{tags_html}</div>
                </div>
                <div style="flex: 1; border-left: 1px solid #f1f5f9; padding-left: 1.5rem;">
                    <div>
                        <p style="font-size: 0.65rem; font-weight: 800; color: #94a3b8; text-transform: uppercase;">Total Deal Value</p>
                        <p style="font-size: 1.6rem; font-weight: 900; color: #059669; margin: 0;">{html.escape(str(row['DisplayValue']))}</p>
                    </div>
                    <div style="margin-top: 1.25rem;">
                        <p style="font-size: 0.65rem; font-weight: 800; color: #94a3b8; text-transform: uppercase;">Upfront Ratio ({r_pct:.1f}%)</p>
                        <div class="ratio-bar-bg"><div style="height:10px; width:{r_pct}%; background-color:{r_color}; border-radius:5px;"></div></div>
                    </div>
                    <div style="margin-top: 1.25rem;">
                        <p style="font-size: 0.65rem; font-weight: 800; color: #94a3b8; text-transform: uppercase;">Partners</p>
                        <p style="font-weight: 800; color: #0f172a; font-size: 1rem; margin: 0;">{html.escape(str(row['PartnerA']))}</p>
                        <p style="color: #64748b; font-size: 0.8rem; margin: 0;">{html.escape(str(row['PartnerB']))}</p>
                    </div>
                </div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

except Exception as e:
    st.error(f"BI Module Error: {e}")
