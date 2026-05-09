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
        padding: 2rem;
        border-radius: 1rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .parent-tag {
        background-color: #eff6ff;
        color: #1e40af;
        padding: 0.2rem 0.6rem;
        border-radius: 0.4rem;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        border: 1px solid #bfdbfe;
    }
    .ratio-bar {
        height: 8px;
        background-color: #f1f5f9;
        border-radius: 4px;
        margin-top: 5px;
    }
    .ratio-fill {
        height: 8px;
        background-color: #10b981;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load & Transform Data
@st.cache_data
def load_and_refine_data():
    # Load your raw database (using CSV as per your earlier file)
    df = pd.read_csv("Biotech_Deals_Database.xlsx - Sheet1.csv") 
    
    # A. Clean Dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values(by='Date', ascending=False)
    
    # B. BI Transformation Layer (The "Chef")
    refined_rows = []
    for _, row in df.iterrows():
        # 1. Hierarchical Modality
        parent = "Other"
        sub_tags = []
        for p_mod, keywords in MODALITY_GROUPS.items():
            found = [k for k in keywords if str(row.get(k, '')).strip().lower() == 'yes']
            if found:
                parent = p_mod
                # Merge Synonyms for display
                sub_tags = [("Gamma Delta" if "delta" in s.lower() else s) for s in found]
                break
        
        # 2. Financial Normalization
        val_m = parse_currency(row.get('Deal Value', ''))
        up_m = parse_currency(row.get('Upfront', ''))
        ratio = (up_m / val_m) if val_m > 0 else 0.0

        refined_rows.append({
            'ID': row.get('ID'),
            'Date': row.get('Date'),
            'ParentModality': parent,
            'SubModalities': list(set(sub_tags)),
            'TotalValueM': val_m,
            'UpfrontM': up_m,
            'UpfrontRatio': ratio,
            'DisplayValue': row.get('Deal Value', 'N/A'),
            'PartnerA': row.get('Partner A', 'N/A'),
            'PartnerB': row.get('Partner B', 'N/A'),
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
    
    # Filter 1: Parent Modality (The Big Picture)
    all_parents = ["All"] + sorted(df['ParentModality'].unique().tolist())
    selected_parent = st.sidebar.selectbox("Broad Modality", all_parents)
    
    # Filter 2: Value Range
    max_val = int(df['TotalValueM'].max())
    value_range = st.sidebar.slider("Min Deal Value ($M)", 0, max_val, 0)

    # 5. Filter Application
    filtered_df = df.copy()
    if selected_parent != "All":
        filtered_df = filtered_df[filtered_df['ParentModality'] == selected_parent]
    filtered_df = filtered_df[filtered_df['TotalValueM'] >= value_range]

    # 6. BI Header Metrics
    st.title("Strategic Deal Stream")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Deals", len(filtered_df))
    col2.metric("Total Value Managed", f"${filtered_df['TotalValueM'].sum()/1000:.1f}B")
    avg_ratio = filtered_df[filtered_df['UpfrontRatio'] > 0]['UpfrontRatio'].mean()
    col3.metric("Avg. Upfront Ratio", f"{avg_ratio:.1%}")

    # 7. Card Display
    for _, row in filtered_df.iterrows():
        # Financial logic for display
        ratio_pct = row['UpfrontRatio'] * 100
        ratio_color = "#10b981" if ratio_pct > 20 else "#f59e0b"
        
        subs_html = " ".join([f'<span class="tag">{s}</span>' for s in row['SubModalities']])
        
        st.markdown(f"""
        <div class="deal-card">
            <div style="display: flex; justify-content: space-between;">
                <div style="flex: 2;">
                    <span class="parent-tag">{row['ParentModality']}</span>
                    <h2 style="color: #1d4ed8; margin-top: 10px; font-size: 1.5rem;">{row['Insight']}</h2>
                    <p style="font-weight: 700; font-size: 0.9rem;">{row['Title']}</p>
                    <p style="color: #64748b; font-size: 0.85rem;">{row['Summary']}</p>
                    <div style="margin-top: 10px;">{subs_html}</div>
                </div>
                <div style="flex: 1; border-left: 1px solid #e2e8f0; padding-left: 20px; text-align: right;">
                    <p style="font-size: 0.7rem; color: #94a3b8; font-weight: 800;">TOTAL DEAL VALUE</p>
                    <p style="font-size: 1.8rem; font-weight: 900; color: #0f172a;">{row['DisplayValue']}</p>
                    
                    <p style="font-size: 0.7rem; color: #94a3b8; font-weight: 800; margin-top: 15px;">CASH UPFRONT RATIO ({ratio_pct:.1f}%)</p>
                    <div class="ratio-bar">
                        <div class="ratio-fill" style="width: {ratio_pct}%; background-color: {ratio_color};"></div>
                    </div>
                    
                    <div style="margin-top: 20px;">
                        <p style="font-size: 0.7rem; color: #94a3b8; font-weight: 800;">PARTNERS</p>
                        <p style="font-size: 0.9rem; font-weight: 700;">{row['PartnerA']}</p>
                        <p style="font-size: 0.8rem; color: #64748b;">{row['PartnerB']}</p>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error loading stream: {e}")
