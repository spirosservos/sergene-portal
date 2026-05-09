import streamlit as st
import pandas as pd
import html
import re
from datetime import datetime

# 1. Page Configuration
st.set_page_config(page_title="SerGene Strategic Intelligence", layout="wide")

# --- BI MAPPINGS (The "Brain" of the transformation) ---
MODALITY_GROUPS = {
    "Cell Therapy": ["CAR-T", "TCR", "TILs", "NK Cells", "Tregs", "MSCs", "iPSCs", "gamma delta T cells", "γδ T cells", "Cell Therapy"],
    "Gene Therapy/Editing": ["CRISPR", "Base Editing", "Prime Editing", "Gene Editing", "Gene Therapy"],
    "RNA Therapeutics": ["mRNA", "siRNA", "RNAi", "miRNA", "ASO", "Antisense", "Aptamer", "RNA"],
    "Biologics": ["Antibody", "Bispecific", "ADC", "Multi-specific", "Peptide", "Biologics"],
    "Small Molecule": ["Small Molecule", "Protein Degrader", "Oral"]
}

def parse_currency(val_str):
    """Converts strings like '$1.5B' or '$50M' into floats."""
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
    .ratio-bar {
        height: 10px;
        background-color: #f1f5f9;
        border-radius: 5px;
        margin-top: 8px;
        overflow: hidden;
    }
    .ratio-fill {
        height: 100%;
        border-radius: 5px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 900;
        color: #0f172a;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load & Transform Data
@st.cache_data
def load_and_refine_data():
    # Load from Feather (Arrow) format
    df = pd.read_feather("sg_intel_assets.arrow") 
    
    # A. Clean Dates
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(by='Date', ascending=False)
    
    # B. BI Transformation Layer
    refined_rows = []
    for _, row in df.iterrows():
        # 1. Hierarchical Modality 
        # Check against the list in 'ModalityTags'
        tags = row.get('ModalityTags', [])
        parent = "Other"
        for p_mod, keywords in MODALITY_GROUPS.items():
            if any(tag in keywords for tag in tags):
                parent = p_mod
                break
        
        # 2. Financial Normalization
        # Using keys from your previous webapp script (DealValue, Upfront)
        val_m = parse_currency(row.get('DealValue', ''))
        up_m = parse_currency(row.get('Upfront', '')) # Check if Upfront exists in arrow
        ratio = (up_m / val_m) if val_m > 0 else 0.0

        refined_rows.append({
            'ID': row.get('ID'),
            'Date': row.get('Date'),
            'ParentModality': parent,
            'SubModalities': tags, # Keep original tags for the card
            'TotalValueM': val_m,
            'UpfrontM': up_m,
            'UpfrontRatio': ratio,
            'DisplayValue': row.get('DealValue', 'N/A'),
            'PartnerA': row.get('PartnerA', 'N/A'),
            'PartnerB': row.get('PartnerB', 'N/A'),
            'Score': row.get('Score', 0),
            'Insight': row.get('Insight', ''),
            'Title': row.get('Title', ''),
            'Summary': row.get('Summary', ''),
            'Category': row.get('Category', 'Other'),
            'Link': row.get('Link', '#')
        })
    
    return pd.DataFrame(refined_rows)

try:
    df = load_and_refine_data()

    # 4. Sidebar BI Filters
    st.sidebar.title("SerGene Intelligence")
    
    # Simple Access Check
    with st.sidebar.expander("🔐 Access", expanded=False):
        password = st.text_input("Code", type="password")
        is_auth = (password == "SerGene2024")

    # BI Filters
    all_parents = ["All"] + sorted(df['ParentModality'].unique().tolist())
    selected_parent = st.sidebar.selectbox("Broad Modality", all_parents)
    
    # Range Slider for Big Deals
    max_val = int(df['TotalValueM'].max()) if not df.empty else 1000
    min_val_filter = st.sidebar.slider("Min Deal Value ($M)", 0, max_val, 0)

    # 5. Filter Application
    filtered_df = df.copy()
    if selected_parent != "All":
        filtered_df = filtered_df[filtered_df['ParentModality'] == selected_parent]
    filtered_df = filtered_df[filtered_df['TotalValueM'] >= min_val_filter]

    # Limit view for guests
    display_df = filtered_df if is_auth else filtered_df.head(20)

    # 6. BI Header Metrics (Summary of Current View)
    st.title("Strategic Deal Stream")
    m1, m2, m3 = st.columns(3)
    m1.metric("Deals Found", len(filtered_df))
    m2.metric("Market Volume", f"${filtered_df['TotalValueM'].sum()/1000:.1f}B")
    
    # Calculate Avg Upfront Ratio for deals that have financial data
    valid_ratios = filtered_df[filtered_df['UpfrontRatio'] > 0]['UpfrontRatio']
    avg_r = valid_ratios.mean() if not valid_ratios.empty else 0
    m3.metric("Avg. Cash Intensity", f"{avg_r:.1%}")

    # 7. Card Display
    for _, row in display_df.iterrows():
        # Visual Logic for Ratio Bar
        r_pct = row['UpfrontRatio'] * 100
        r_color = "#10b981" if r_pct > 25 else "#f59e0b" # Green if high cash, Orange if risky
        
        tags_html = " ".join([f'<span class="tag">{html.escape(t)}</span>' for t in row['SubModalities']])
        
        st.markdown(f"""
        <div class="deal-card">
            <div style="display: flex; justify-content: space-between; align-items: start; gap: 2rem;">
                <div style="flex: 2;">
                    <span class="parent-tag">{row['ParentModality']}</span>
                    <h2 style="color: #3b82f6; font-size: 1.5rem; font-weight: 800; margin-top: 1rem; margin-bottom: 0.5rem;">{row['Insight']}</h2>
                    <div style="font-weight: 700; color: #0f172a; margin-bottom: 1rem;">{row['Title']}</div>
                    <p style="color: #475569; font-size: 0.95rem; line-height: 1.6;">{row['Summary']}</p>
                    <div style="margin-top: 1.5rem;">{tags_html}</div>
                </div>
                <div style="flex: 1; border-left: 2px solid #f1f5f9; padding-left: 2rem;">
                    <div>
                        <p style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase;">Total Deal Value</p>
                        <p style="font-size: 1.8rem; font-weight: 900; color: #059669;">{row['DisplayValue']}</p>
                    </div>
                    
                    <div style="margin-top: 1.5rem;">
                        <p style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase;">Cash Upfront Ratio ({r_pct:.1f}%)</p>
                        <div class="ratio-bar">
                            <div class="ratio-fill" style="width: {r_pct}%; background-color: {r_color};"></div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 1.5rem;">
                        <p style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase;">Partners</p>
                        <p style="font-weight: 800; color: #0f172a; font-size: 1.1rem;">{row['PartnerA']}</p>
                        <p style="color: #
