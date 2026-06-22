import streamlit as st
import pandas as pd
import numpy as np
import html
import re
import os
import textwrap  
from datetime import datetime
from google import genai 
import plotly.express as px  

# ==========================================
# 1. PAGE CONFIG & AI INITIALIZATION
# ==========================================
st.set_page_config(
    page_title="SerGene Strategic Intelligence",
    page_icon="🧬",
    layout="wide"
)

# AI Init
GENAI_KEY = st.secrets.get("GEMINI_API_KEY")
if GENAI_KEY:
    ai_client = genai.Client(api_key=GENAI_KEY)
    AI_MODEL = "gemini-3.1-flash-lite-preview"
else:
    st.error("AI Configuration Error: Gemini API Key not found.")

# Reconfigured Modality Classes aligned to strategic hierarchy
MODALITY_GROUPS = {
    "Gene Therapy/Editing": ["CRISPR", "Base Editing", "Prime Editing", "Gene Editing", "Gene Therapy", "AAV", "Lentivirus", "Lenti", "Alternative & General Vectors"],
    "Cell Therapy": ["CAR-T", "TCR", "TILs", "NK Cells", "Tregs", "MSCs", "iPSCs", "GammaDelta"],
    "RNA Therapeutics": ["mRNA", "siRNA", "RNAi", "miRNA", "ASO", "Antisense", "Aptamer", "RNA", "ASO / Antisense"],
    "Immunotherapies": ["Oncolytic Virus", "Immuno-oncology"],
    "Biologics": ["Antibody", "Bispecific", "ADC", "Multi-specific", "Peptide", "Biologics", "Exosomes"],
    "Small Molecule": ["Small Molecule", "Protein Degrader", "Oral"]
}

CELL_THERAPY_TAGS = ["CAR-T", "TCR", "TILs", "NK Cells", "Tregs", "MSCs", "iPSCs", "GammaDelta"]

MODALITY_ORDER = [
    "Gene Therapy/Editing",
    "Cell Therapy",
    "RNA Therapeutics",
    "Immunotherapies",
    "Biologics",
    "Small Molecule",
    "Emerging Platforms & Conjugates"
]

PLATFORM_ORDER = [
    'CRISPR', 'Gene Editing', 'Base Editing', 'Prime Editing', 'Gene Therapy', 'AAV', 'Lentivirus', 'Lenti', 'Alternative & General Vectors',
    'RNA', 'mRNA', 'siRNA', 'RNAi', 'miRNA', 'ASO / Antisense', 'Aptamer',
    'Oncolytic Virus', 'Immuno-oncology',
    'Biologics', 'Antibody', 'Bispecific', 'Multi-specific', 'ADC', 'Peptide', 'Exosomes',
    'Small Molecule', 'Protein Degrader', 'Oral',
    'LNP', 'Nanoparticle', 'Radiopharmaceutical', 'GLP-1', 'Incretin'
]

LOGICAL_STAGES = ["Preclinical", "Phase 1", "Phase 2", "Phase 3", "Marketed", "Terminated"]
MATRIX_STAGES = ["Preclinical", "Phase 1", "Phase 2", "Phase 3"]

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
        
        val = float(clean_val)
        if val > 10000:
            return val / 1000000.0
        return val
    except: return 0.0

def smart_format_company(name):
    if not name or pd.isna(name) or str(name).lower() == 'nan': return "N/A"
    text = str(name).strip()
    words = text.split()
    formatted_words = [word.capitalize() if word.islower() else word for word in words]
    return " ".join(formatted_words)

@st.cache_data
def load_funding_vcs():
    excel_path = "reference_data.xlsx"
    if os.path.exists(excel_path):
        try:
            ref_df = pd.read_excel(excel_path)
            if 'VC' in ref_df.columns:
                return set(ref_df['VC'].dropna().astype(str).str.strip().tolist())
        except Exception as e:
            st.error(f"Error reading reference_data.xlsx: {e}")
    return set()

def extract_individual_companies(df):
    all_entities = set()
    excluded_placeholders = ['n/a', 'nan', '', 'locked', 'unknown']
    for col in ['PartnerA', 'PartnerB']:
        if col in df.columns:
            for cell in df[col].dropna().astype(str):
                parts = [p.strip() for p in cell.split(',')]
                for p in parts:
                    if p and p.lower() not in excluded_placeholders:
                        all_entities.add(p)
    return sorted(list(all_entities), key=lambda s: s.lower())

def reset_all_filters_callback():
    state_resets = {
        "directory_alphabet_selector": "All",
        "selected_company_dropdown": "None",
        "selected_vc_dropdown": "None",
        "sidebar_parents": [],
        "sidebar_platforms": [],
        "sidebar_cells": [],
        "sidebar_tas": [],
        "sidebar_stages": [],
        "sidebar_search": ""
    }
    for key, def_val in state_resets.items():
        st.session_state[key] = def_val

# ==========================================
# 3. CSS STYLING
# ==========================================
st.markdown("""
    <style>
    .main, .stApp { background-color: #f8fafc; }
    
    div[data-baseweb="select"] {
        border: 2px solid #10b981 !important;
        border-radius: 0.5rem;
    }
    
    /* 👈 NEW: Enforces green coloring and increases font size across directory navigation tabs */
    button[data-baseweb="tab"], button[data-baseweb="tab"] span, button[data-baseweb="tab"] div {
        color: #10b981 !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
    }
    
    .deal-card {
        background-color: white; padding: 2.5rem; border-radius: 1.5rem;
        border: 1px solid #e2e8f0; margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
    .blurred-card { filter: blur(8px); opacity: 0.5; pointer-events: none; }
    .blur-financials { filter: blur(5px); opacity: 0.3; pointer-events: none; user-select: none; }
    .date-badge { color: #64748b; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; margin-bottom: 0.75rem; }
    .parent-tag {
        background-color: #eff6ff; color: #1e40af; padding: 0.35rem 0.85rem;
        border-radius: 0.75rem; font-size: 0.75rem; font-weight: 800;
        text-transform: uppercase; border: 1px solid #bfdbfe; display: inline-block; margin-bottom: 1rem;
    }
    .source-link { color: #2563eb; text-decoration: none; font-weight: 800; font-size: 1.5rem; }
    .summary-text { color: #475569; font-size: 0.95rem; line-height: 1.6; margin: 1.25rem 0; }
    .tag {
        display: inline-block; background-color: #f1f5f9; color: #475569;
        padding: 0.3rem 0.75rem; border-radius: 0.6rem; font-size: 0.7rem;
        font-weight: 700; margin-right: 0.5rem; border: 1px solid #e2e8f0;
    }
    .ratio-bar-container { height: 12px; background-color: #f1f5f9; border-radius: 6px; margin-top: 5px; overflow: hidden; border: 1px solid #e2e8f0; }
    .ai-strategy-box { background-color: #f0f9ff; border-left: 6px solid #0ea5e9; padding: 1.75rem; border-radius: 0.75rem; margin: 2rem 0; }
    .cta-banner { background-color: #fef2f2; border: 2px dashed #ef4444; padding: 2.5rem; border-radius: 1.5rem; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 4. DATA REFINERY (BULLETPROOF SEARCH)
# ==========================================
@st.cache_data
def load_and_refine_data():
    if not os.path.exists("sg_intel_assets.arrow"):
        return pd.DataFrame()
    df = pd.read_feather("sg_intel_assets.arrow") 
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(by='Date', ascending=False)
    
    TECH_COLUMNS = [
        'Small Molecule', 'Biologics', 'Protein Degrader', 'Peptide', 'GLP-1', 'Incretin', 
        'RNA', 'mRNA', 'siRNA', 'RNAi', 'miRNA', 'ASO', 'Antisense', 'Aptamer', 
        'CRISPR', 'Gene Therapy', 'Gene Editing', 'Base Editing', 'Prime Editing', 
        'AAV', 'Lentivirus', 'Lenti', 'Oncolytic Virus', 'Vector', 'Tregs', 'TCR', 
        'CAR-T', 'Cell Therapy', 'NK Cells', 'TILs', 'ADC', 'Antibody', 'Bispecific', 
        'Exosomes', 'LNP', 'Oral', 'Radiopharmaceutical', 'Immuno-oncology', 
        'Multi-specific', 'Nanoparticle', 'MSCs', 'iPSCs', 'gamma delta T cells', 'γδ T cells'
    ]
    
    tech_columns_clean = [c.strip().lower() for c in TECH_COLUMNS]
    
    refined_rows = []
    for _, row in df.iterrows():
        raw_row_flags = []
        for col_name in row.index:
            col_name_clean = str(col_name).strip().lower()
            val = row[col_name]
            
            if isinstance(val, (list, np.ndarray)):
                is_positive_signal = any(
                    str(i).strip().lower() in ['yes', 'y', 'true', '1'] or 
                    (isinstance(i, (int, float)) and i > 0) 
                    for i in val if pd.notna(i)
                )
            else:
                if pd.isna(val):
                    continue
                clean_val = str(val).strip().lower()
                is_positive_signal = False
                if clean_val in ['yes', 'y', 'true', '1']:
                    is_positive_signal = True
                else:
                    try:
                        if float(clean_val) > 0:
                            is_positive_signal = True
                    except (ValueError, TypeError):
                        pass
            
            if is_positive_signal:
                if "nk" in col_name_clean:
                    raw_row_flags.append("NK Cells")
                elif any(x in col_name_clean for x in ["gamma", "delta", "γ", "δ"]):
                    raw_row_flags.append("gamma delta T cells")
                elif col_name_clean in tech_columns_clean:
                    matched_idx = tech_columns_clean.index(col_name_clean)
                    raw_row_flags.append(TECH_COLUMNS[matched_idx])

        tags = []
        has_specific_viral = any(x in raw_row_flags for x in ['AAV', 'Lentivirus', 'Lenti'])

        for flag in raw_row_flags:
            if flag == 'Vector':
                if not has_specific_viral:
                    tags.append("Alternative & General Vectors")
            elif flag in ['ASO', 'Antisense']:
                tags.append("ASO / Antisense")
            elif "msc" in flag.lower():
                tags.append("MSCs")
            elif "ipsc" in flag.lower():
                tags.append("iPSCs")
            elif any(x in flag.lower() for x in ["gamma", "delta", "γ", "δ"]):
                tags.append("GammaDelta")
            else:
                tags.append(flag)

        tags = list(set([t for t in tags if t and str(t).lower() != 'nan']))
        cell_types_extracted = [t for t in tags if t in CELL_THERAPY_TAGS]
        platforms_extracted = [t for t in tags if t not in CELL_THERAPY_TAGS]
        
        parent = "Emerging Platforms & Conjugates"  
        norm_tags = [t.lower() for t in tags]
        for group_name, keywords in MODALITY_GROUPS.items():
            if any(k.lower() in norm_tags for k in keywords):
                parent = group_name
                break
        
        val_m = parse_currency(row.get('DealValue', ''))
        up_m = parse_currency(row.get('Upfront', ''))
        
        ratio = (up_m / val_m) if val_m > 0 else 0.0
        if ratio > 1.0:
            ratio = 1.0

        row_values = []
        for val in row.values:
            if isinstance(val, (list, np.ndarray)):
                row_values.extend([str(i) for i in val if pd.notna(i)])
            elif pd.notna(val):
                row_values.append(str(val))
        
        blob = " ".join(row_values).lower().replace('\xa0', ' ')
        blob = " ".join(blob.split())

        raw_stage = str(row.get('Stage', 'Preclinical')).strip()
        if "pre" in raw_stage.lower(): cleaned_stage = "Preclinical"
        elif "market" in raw_stage.lower() or "commerc" in raw_stage.lower(): cleaned_stage = "Marketed"
        elif "term" in raw_stage.lower(): cleaned_stage = "Terminated"
        else: cleaned_stage = raw_stage

        refined_rows.append({
            'Row_ID': row.name, 
            'Date': row.get('Date'),
            'Date_Obj': row.get('Date').date() if pd.notnull(row.get('Date')) else None,
            'DisplayDate': row.get('Date').strftime('%b %d, %Y') if pd.notnull(row.get('Date')) else "N/A",
            'ParentModality': parent,
            'SubModalities': tags,
            'CellTypes': cell_types_extracted,
            'Platforms': platforms_extracted,
            'Category': str(row.get('Category', 'Partnership/R&D')).strip(),
            'TA': str(row.get('TA', 'Other/General')).strip(),
            'TargetDisease': str(row.get('Target Disease', row.get('TargetDisease', 'N/A'))),
            'Stage': cleaned_stage,
            'TotalValueM': val_m,
            'UpfrontRatio': ratio,
            'DisplayValue': str(row.get('DealValue', 'N/A')),
            'PartnerA': smart_format_company(row.get('PartnerA')),
            'PartnerB': smart_format_company(row.get('PartnerB')),
            'Insight': str(row.get('Insight', '')),
            'Title': str(row.get('Title', '')),
            'Summary': str(row.get('Summary', '')),
            'Link': str(row.get('Link', '#')),
            'SearchBlob': blob
        })
    return pd.DataFrame(refined_rows)

# ==========================================
# 5. UI, AUTHENTICATION & FILTERING
# ==========================================
df_master = load_and_refine_data()

if df_master.empty or 'Date_Obj' not in df_master.columns:
    st.warning("⚠️ Application Out of Sync: Streamlit Cloud is holding an older data cache snapshot.")
    if st.button("🔄 Clear Server Cache Memory & Rebuild Database", key="critical_force_cache_clear_btn"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

try:
    def log_audit_event(client_tag, action, details=""):
        log_file = "sergene_audit_log.csv"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ip_address = "Unknown"
        try:
            if hasattr(st, "context") and st.context.headers:
                x_forwarded = st.context.headers.get("x-forwarded-for")
                if x_forwarded: ip_address = x_forwarded.split(",")[0].strip()
                else: ip_address = st.context.headers.get("remote-addr", "Unknown")
        except Exception: pass
            
        log_entry = pd.DataFrame([{"Timestamp": timestamp, "Client_Tag": client_tag, "Action": action, "IP_Address": ip_address, "Details": details}])
        try:
            if not os.path.exists(log_file): log_entry.to_csv(log_file, index=False)
            else: log_entry.to_csv(log_file, mode='a', header=False, index=False)
        except Exception: pass 

    df_master = df_master.dropna(subset=['Date_Obj']).sort_values('Date_Obj', ascending=False)
    
    st.sidebar.title("🧬 SerGene Intelligence")
    st.sidebar.button("🔄 Reset All System Filters", on_click=reset_all_filters_callback, use_container_width=True)
    st.sidebar.markdown("---")

    st.sidebar.subheader("📅 Select Timeframe")
    min_db = df_master['Date_Obj'].min()
    max_db = df_master['Date_Obj'].max()
    date_sel = st.sidebar.date_input("Date Range", value=(min_db, max_db), min_value=min_db, max_value=max(max_db, datetime.now().date()))
    st.sidebar.divider()

    if "download_count" not in st.session_state: st.session_state["download_count"] = 0
    if "is_authenticated" not in st.session_state: st.session_state["is_authenticated"] = False
    if "active_client_tag" not in st.session_state: st.session_state["active_client_tag"] = "Guest"

    with st.sidebar.expander("🔑 Client Access", expanded=False):
        raw_keys = st.secrets.get("client_keys", "")
        valid_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        legacy_pass = st.secrets.get("access_password")
        if legacy_pass: valid_keys.append(legacy_pass)

        password_input = st.text_input("Enter Access Code", type="password", key="access_credential_input")
        if password_input in valid_keys and password_input != "":
            if not st.session_state["is_authenticated"] or st.session_state["active_client_tag"] != password_input:
                st.session_state["is_authenticated"] = True
                st.session_state["active_client_tag"] = password_input
                log_audit_event(password_input, "Login Success", "Client authenticated via sidebar.")
        elif password_input:
            if st.session_state["is_authenticated"]: log_audit_event(st.session_state["active_client_tag"], "De-authenticated", "Incorrect key layer override attempt.")
            st.session_state["is_authenticated"] = False
            st.session_state["active_client_tag"] = "Guest"

        if st.session_state["is_authenticated"]: st.success("Access Verified")
        else:
            st.markdown("---")
            st.caption("Contact Support for Code:")
            st.code("spiros@sergenebio.co.uk")

    is_authenticated = st.session_state["is_authenticated"]
    client_tag = st.session_state["active_client_tag"]
    st.sidebar.divider()

    existing_parents = df_master['ParentModality'].unique().tolist()
    sorted_parents_options = [m for m in MODALITY_ORDER if m in existing_parents] + [m for m in existing_parents if m not in MODALITY_ORDER]
    sel_parents = st.sidebar.multiselect("Modality Class", sorted_parents_options, key="sidebar_parents")
    
    all_platforms = list(set([p for sub in df_master['Platforms'] for p in sub]))
    sorted_platform_options = [p for p in PLATFORM_ORDER if p in all_platforms] + [p for p in all_platforms if p not in PLATFORM_ORDER]
    sel_platforms = st.sidebar.multiselect("Platforms & Delivery", sorted_platform_options, key="sidebar_platforms")

    sel_cells = st.sidebar.multiselect("Cell Types", CELL_THERAPY_TAGS, key="sidebar_cells")
    sel_tas = st.sidebar.multiselect("Therapeutic Area", sorted(df_master['TA'].unique().tolist()), key="sidebar_tas")
    sel_stages = st.sidebar.multiselect("Development Stage", sorted(df_master['Stage'].unique().tolist()), key="sidebar_stages")
    search_term = st.sidebar.text_input("🔍 Search Everything (Deep Scan)", key="sidebar_search")

    if is_authenticated and client_tag == "SPIROS-VIP":
        st.sidebar.markdown("---")
        st.sidebar.subheader("🛡️ System Administration")
        if os.path.exists("sergene_audit_log.csv"):
            with open("sergene_audit_log.csv", "rb") as f:
                st.sidebar.download_button(label="📥 Download Master Audit Logs", data=f, file_name="Master_Security_Audit_Log.csv", mime="text/csv", key="admin_audit_log_download_btn")

    stats_df = df_master.copy()
    if isinstance(date_sel, (list, tuple)) and len(date_sel) == 2:
        stats_df = stats_df[(stats_df['Date_Obj'] >= date_sel[0]) & (stats_df['Date_Obj'] <= date_sel[1])]
    if len(sel_parents) > 0: stats_df = stats_df[stats_df['ParentModality'].isin(sel_parents)]
    if len(sel_platforms) > 0: stats_df = stats_df[stats_df['Platforms'].apply(lambda x: any(s in x for s in sel_platforms))]
    if len(sel_cells) > 0: stats_df = stats_df[stats_df['CellTypes'].apply(lambda x: any(s in x for s in sel_cells))]
    if len(sel_tas) > 0: stats_df = stats_df[stats_df['TA'].isin(sel_tas)]
    if len(sel_stages) > 0: stats_df = stats_df[stats_df['Stage'].isin(sel_stages)]
    if search_term: stats_df = stats_df[stats_df['SearchBlob'].str.contains(search_term.lower(), na=False)]

    # ==========================================
    # 5.5 TOP-LEVEL NETWORK DIRECTORY & FILTERS
    # ==========================================
    funding_vcs = load_funding_vcs()
    funding_vcs_lower = {vc.lower() for vc in funding_vcs}

    all_individual_entities = extract_individual_companies(df_master)

    regular_companies = [c for c in all_individual_entities if c.lower() not in funding_vcs_lower]
    available_vcs = [c for c in all_individual_entities if c.lower() in funding_vcs_lower]

    st.markdown("### 🏢 Network Directory")
    
    col_dir_title, col_dir_reset = st.columns([4, 1])
    with col_dir_reset:
        st.button("🔄 Reset Directory Filters", on_click=reset_all_filters_callback, key="inline_directory_reset_btn", use_container_width=True)

    dir_tab1, dir_tab2 = st.tabs(["Alphabetical Company Index", "Investors & Funding Firms"])

    with dir_tab1:
        alphabet_options = ["All", "0-9"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        selected_letter = st.radio(
            "Filter directory by starting character:",
            alphabet_options,
            horizontal=True,
            key="directory_alphabet_selector"
        )
        
        if selected_letter == "All":
            dropdown_companies = regular_companies
        elif selected_letter == "0-9":
            dropdown_companies = [c for c in regular_companies if c and c[0].isdigit()]
        else:
            dropdown_companies = [c for c in regular_companies if c and c.upper().startswith(selected_letter)]
            
        selected_company = st.selectbox(
            f"Select a Company ({len(dropdown_companies)} matched):",
            ["None"] + dropdown_companies,
            key="selected_company_dropdown"
        )

    with dir_tab2:
        selected_vc = st.selectbox(
            f"Select a Venture Capital / Funding Firm ({len(available_vcs)} tracked in data):",
            ["None"] + sorted(available_vcs, key=lambda s: s.lower()),
            help="Select an investor to view every startup or asset they have funded.",
            key="selected_vc_dropdown"
        )

    def is_entity_in_cell(cell_val, target_name):
        if pd.isna(cell_val):
            return False
        return target_name.lower() in [p.strip().lower() for p in str(cell_val).split(',')]

    if "selected_company_dropdown" in st.session_state and st.session_state["selected_company_dropdown"] != "None":
        current_comp = st.session_state["selected_company_dropdown"]
        stats_df = stats_df[
            stats_df['PartnerA'].apply(lambda x: is_entity_in_cell(x, current_comp)) |
            stats_df['PartnerB'].apply(lambda x: is_entity_in_cell(x, current_comp))
        ]

    if "selected_vc_dropdown" in st.session_state and st.session_state["selected_vc_dropdown"] != "None":
        current_vc = st.session_state["selected_vc_dropdown"]
        stats_df = stats_df[
            stats_df['PartnerA'].apply(lambda x: is_entity_in_cell(x, current_vc)) |
            stats_df['PartnerB'].apply(lambda x: is_entity_in_cell(x, current_vc))
        ]

    GLOBAL_PREVIEW_LIMIT = 5
    visible_df = stats_df if is_authenticated else stats_df.head(GLOBAL_PREVIEW_LIMIT)

    # ==========================================
    # 6. DASHBOARD METRICS
    # ==========================================
    st.title("Strategic Deal Intelligence Stream")
    
    current_filtered_entities = extract_individual_companies(stats_df)
    unique_companies_count = len(current_filtered_entities)
    
    partnership_pool = stats_df[stats_df['Category'] == 'Partnership/R&D']
    mna_pool = stats_df[stats_df['Category'] == 'Financial/M&A']

    true_p_deals = partnership_pool[(partnership_pool['TotalValueM'] > 0) & (partnership_pool['UpfrontRatio'] > 0) & (partnership_pool['UpfrontRatio'] < 1.0)]
    p_upfront_total = (true_p_deals['TotalValueM'] * true_p_deals['UpfrontRatio']).sum()
    p_volume_total = true_p_deals['TotalValueM'].sum()
    macro_p_ratio = (p_upfront_total / p_volume_total) if p_volume_total > 0 else 0.0

    true_m_deals = mna_pool[(mna_pool['TotalValueM'] > 0) & (mna_pool['UpfrontRatio'] > 0) & (mna_pool['UpfrontRatio'] < 1.0)]
    m_upfront_total = (true_m_deals['TotalValueM'] * true_m_deals['UpfrontRatio']).sum()
    m_volume_total = true_m_deals['TotalValueM'].sum()
    macro_m_ratio = (m_upfront_total / m_volume_total) if m_volume_total > 0 else 0.0

    # Row 1: General Database Scope Metrics
    row1_m1, row1_m2, row1_m3 = st.columns(3)
    row1_m1.metric("Database Depth", f"{len(stats_df)} Deals")
    row1_m2.metric("Companies Tracked", f"{unique_companies_count} Unique")
    row1_m3.metric("Total Market Volume Analyzed", f"${stats_df['TotalValueM'].sum()/1000:.2f}B")

    # Row 2: Segmented Financial Volume & Cleaned Risk Ratios
    row2_m1, row2_m2, row2_m3, row2_m4 = st.columns(4)
    row2_m1.metric("Partnership Volume", f"${partnership_pool['TotalValueM'].sum()/1000:.2f}B")
    row2_m2.metric("Partnership Avg. Upfront Ratio", f"{macro_p_ratio:.1%}", help="Calculated exclusively using structured biobucks deals (excluding flat/undisclosed milestones).")
    row2_m3.metric("M&A Asset Volume", f"${mna_pool['TotalValueM'].sum()/1000:.2f}B")
    row2_m4.metric("M&A Avg. Upfront Ratio", f"{macro_m_ratio:.1%}", help="Calculated using structured earnout acquisitions.")
    
    st.divider()

    # ==========================================
    # 6.1 PREMIUM FINANCIAL METRICS ENGINE (THE STRUCTURAL MATRIX)
    # ==========================================
    st.subheader("📊 Strategic Pipeline Financial Matrix")
    
    if is_authenticated:
        matrix_records = []
        for stage_name in MATRIX_STAGES: 
            stage_slice = stats_df[stats_df['Stage'] == stage_name]
            
            # --- METRIC A: Partnership / Licensing Metrics ---
            licensing_slice = stage_slice[stage_slice['Category'] == 'Partnership/R&D']
            l_count = len(licensing_slice)
            
            valid_l_financials = licensing_slice[licensing_slice['TotalValueM'] > 0]
            avg_l_total = f"${valid_l_financials['TotalValueM'].mean():.1f}M" if not valid_l_financials.empty else "—"
            
            upfront_dollars = licensing_slice['TotalValueM'] * licensing_slice['UpfrontRatio']
            avg_l_upfront = f"${upfront_dollars[licensing_slice['TotalValueM'] > 0].mean():.1f}M" if not valid_l_financials.empty else "—"
            
            struct_l = licensing_slice[(licensing_slice['TotalValueM'] > 0) & (licensing_slice['UpfrontRatio'] > 0) & (licensing_slice['UpfrontRatio'] < 1.0)]
            struct_l_upfront = (struct_l['TotalValueM'] * struct_l['UpfrontRatio']).sum()
            struct_l_total = struct_l['TotalValueM'].sum()
            l_ratio_pct = f"{(struct_l_upfront / struct_l_total * 100):.1f}%" if struct_l_total > 0 else "—"
            
            # --- METRIC B: Financial / M&A Metrics ---
            mna_slice = stage_slice[stage_slice['Category'] == 'Financial/M&A']
            m_count = len(mna_slice)
            
            valid_m_financials = mna_slice[mna_slice['TotalValueM'] > 0]
            avg_mna_total = f"${valid_m_financials['TotalValueM'].mean():.1f}M" if not valid_m_financials.empty else "—"
            
            struct_m = mna_slice[(mna_slice['TotalValueM'] > 0) & (mna_slice['UpfrontRatio'] > 0) & (mna_slice['UpfrontRatio'] < 1.0)]
            struct_m_upfront = (struct_m['TotalValueM'] * struct_m['UpfrontRatio']).sum()
            struct_m_total = struct_m['TotalValueM'].sum()
            m_ratio_pct = f"{(struct_m_upfront / struct_m_total * 100):.1f}%" if struct_m_total > 0 else "—"

            matrix_records.append({
                "Clinical Development Stage": stage_name,
                "Partnership Count": l_count,
                "Avg Licensing Upfront": avg_l_upfront,
                "Avg Licensing Total Value": avg_l_total,
                "Avg Licensing Upfront Ratio": l_ratio_pct,
                "M&A Count": m_count,
                "Avg M&A Total Value": avg_mna_total,
                "Avg M&A Upfront Ratio": m_ratio_pct
            })
            
        matrix_df = pd.DataFrame(matrix_records)
        st.dataframe(
            matrix_df, 
            column_config={
                "Clinical Development Stage": st.column_config.TextColumn(help="Standardized asset progression stage."),
                "Partnership Count": st.column_config.NumberColumn(format="%d"),
                "Avg Licensing Upfront": st.column_config.TextColumn(),
                "Avg Licensing Total Value": st.column_config.TextColumn(),
                "Avg Licensing Upfront Ratio": st.column_config.TextColumn(help="True ratio calculated excluding 100% upfront milestones anomalies."),
                "M&A Count": st.column_config.NumberColumn(format="%d"),
                "Avg M&A Total Value": st.column_config.TextColumn(),
                "Avg M&A Upfront Ratio": st.column_config.TextColumn(help="True ratio calculated excluding standard all-cash clean acquisitions."),
            },
            use_container_width=True, 
            hide_index=True
        )
    else:
        st.info("🔒 Premium Analytics Layer: Pipeline asset financial summaries, upfront ratios, and M&A benchmark matrices are locked for guest instances.")
    
    st.divider()

    # ==========================================
    # 6.2 PLOTLY MASTER TIMELINE
    # ==========================================
    if not stats_df.empty:
        timeline_df = stats_df.copy()
        if not timeline_df.empty:
            current_available_order = [o for o in MODALITY_ORDER if o in timeline_df['ParentModality'].unique()]
            timeline_df = timeline_df.sort_values(['Date_Obj', 'ParentModality'], ascending=[True, True])
            timeline_df['cum_idx'] = timeline_df.groupby('Date_Obj').cumcount()
            timeline_df['col_shift'] = timeline_df['cum_idx'] // 6
            timeline_df['stack_y'] = (timeline_df['cum_idx'] % 6) + 1
            timeline_df['Plot_DateTime'] = pd.to_datetime(timeline_df['Date_Obj']) + timeline_df['col_shift'] * pd.Timedelta(hours=5)

            # Normalizes dot scaling weight smoothly using clipped logarithms referenced in image_137f62.png
            timeline_df['PlotSize'] = timeline_df['TotalValueM'].apply(lambda x: np.log1p(min(float(x), 1000.0)) + 1.0 if float(x) > 0 else 1.0)

            hover_meta_list = []
            for _, r in timeline_df.iterrows():
                if not is_authenticated: text_html = "" 
                else:
                    clean_insight = str(r['Insight']).strip()
                    if clean_insight in ["", "nan", "NaN", "N/A"]:
                        wrapped_insight = "Undisclosed"
                    else:
                        wrapped_insight = "<br>".join(textwrap.wrap(html.escape(clean_insight), width=70))
                    
                    clean_value = str(r['DisplayValue']).strip()
                    display_val = html.escape(r['DisplayValue']) if clean_value not in ["", "nan", "NaN", "N/A"] else "Undisclosed"
                    
                    clean_disease = str(r['TargetDisease']).strip()
                    target_dis = html.escape(r['TargetDisease']) if clean_disease not in ["", "nan", "NaN", "N/A"] else "Undisclosed"
                    
                    text_html = (
                        f"<span style='font-family:Arial, sans-serif; line-height:1.6; color:#0f172a;'>"
                        f"<b style='color:#2563eb;'>DATE:</b> {r['DisplayDate']}<br>"
                        f"<b style='color:#059669;'>PARTNERS:</b> {html.escape(r['PartnerA'])} & {html.escape(r['PartnerB'])}<br>"
                        f"<b style='color:#d97706;'>VALUE:</b> {display_val}<br>"
                        f"<b style='color:#7c3aed;'>CLASS:</b> {html.escape(r['ParentModality'])}<br>"
                        f"<b style='color:#0284c7;'>TARGET:</b> {target_dis}<br><br>"
                        f"<b style='color:#dc2626;'>STRATEGIC INSIGHT:</b><br><i style='color:#334155;'>{wrapped_insight}</i></span>"
                    )
                hover_meta_list.append(text_html)
                
            timeline_df['HoverHTML'] = hover_meta_list
            fig_timeline = px.scatter(
                timeline_df, x='Plot_DateTime', y='stack_y', color='ParentModality', 
                size='PlotSize', size_max=22, 
                custom_data=['HoverHTML'],
                color_discrete_map={
                    "Gene Therapy/Editing": "#3b82f6", "Cell Therapy": "#10b981", "RNA Therapeutics": "#6366f1",
                    "Immunotherapies": "#ec4899", "Biologics": "#f59e0b", "Small Molecule": "#b91c1c",
                    "Emerging Platforms & Conjugates": "#64748b"
                },
                category_orders={"ParentModality": current_available_order}
            )
            if is_authenticated: 
                fig_timeline.update_traces(marker=dict(opacity=0.85, line=dict(width=1.5, color='#ffffff')), hovertemplate="%{customdata[0]}<extra></extra>")
            else: 
                fig_timeline.update_traces(marker=dict(opacity=0.85, line=dict(width=1.5, color='#ffffff')), hoverinfo="skip", hovertemplate=None)
            
            fig_timeline.update_layout(
                title=dict(text="🧬 Interactive Deal Intelligence Master Timeline", font=dict(size=16, color='#1e293b', weight='bold')),
                plot_bgcolor='#ffffff', paper_bgcolor='rgba(0,0,0,0)', hovermode='closest', 
                hoverlabel=dict(align="left"), 
                xaxis=dict(title=None, showgrid=True, gridcolor='#f1f5f9', type='date', rangeslider=dict(visible=True, thickness=0.04)),
                yaxis=dict(visible=False, showgrid=False, range=[0.3, 6.7]), legend=dict(title=dict(text="Modality Class"), orientation="h", yanchor="top", y=-0.35, xanchor="center", x=0.5), margin=dict(l=10, r=10, t=50, b=80), height=390  
            )
            st.plotly_chart(fig_timeline, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})
    st.divider()

    with st.expander("📈 Market Trends & Competitive Landscape", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1: st.bar_chart(stats_df['ParentModality'].value_counts(), color="#3b82f6")
        with c2: st.bar_chart(stats_df['TA'].value_counts(), color="#10b981")
        with c3: st.bar_chart(stats_df['Stage'].value_counts(), color="#6366f1")

    if is_authenticated:
        ai_ready = st.toggle("Enable AI Strategic Analysis Tool", value=False)
        if ai_ready and st.button("🪄 Generate AI Strategic Brief"):
            with st.status("🤖 Analyzing current deal flow...", expanded=True):
                deal_list = "\n".join([f"- {r['PartnerA']} & {r['PartnerB']}: {r['Insight']}" for _, r in stats_df.head(20).iterrows()])
                prompt = f"You are a Senior Biotech Strategic Analyst. Analyze these recent deals:\n{deal_list}\n\nProvide a professional 3-point summary:\n1. Biggest trend?\n2. Market risk appetite?\n3. 1-sentence 'Strategic Outlook'."
                try:
                    response = ai_client.models.generate_content(model=AI_MODEL, contents=prompt)
                    st.markdown(f'<div class="ai-strategy-box"><h3 style="margin-top:0;">🤖 Strategic Market Brief</h3><p style="white-space: pre-wrap;">{response.text}</p></div>', unsafe_allow_html=True)
                except Exception as ai_e: st.error(f"AI Error: {ai_e}")
    else:
        st.warning("🔒 AI Strategic Analysis is a Premium Feature for Clients.")

    # ==========================================
    # 6.5 EXPORT INTELLIGENCE STREAM 
    # ==========================================
    st.write("")
    st.subheader("📥 Export Intelligence Stream")
    DOWNLOAD_MAX_SESSION_CAP = 5 
    is_admin = (client_tag == "SPIROS-VIP")
    
    if not stats_df.empty:
        if is_authenticated and st.session_state["download_count"] >= DOWNLOAD_MAX_SESSION_CAP and not is_admin:
            st.error("⚠️ Session Export Limit Reached.")
        else:
            if is_admin:
                admin_df = stats_df.copy()
                export_df = pd.DataFrame({
                    'Internal Row ID': admin_df['Row_ID'], 'Date': admin_df['DisplayDate'], 'Category': admin_df['Category'], 'Modality Class': admin_df['ParentModality'],
                    'Platforms & Delivery': admin_df['Platforms'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x),
                    'Cell Types': admin_df['CellTypes'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x),
                    'Therapeutic Area': admin_df['TA'], 'Target Disease': admin_df['TargetDisease'], 'Development Stage': admin_df['Stage'],
                    'Deal Value': admin_df['DisplayValue'], 'Parsed Numeric Value ($M)': admin_df['TotalValueM'], 'Upfront Ratio': admin_df['UpfrontRatio'].round(4), 
                    'Partner A': admin_df['PartnerA'], 'Partner B': admin_df['PartnerB'], 'Strategic Insight': admin_df['Insight'], 'Source Link URL': admin_df['Link']
                })
            elif is_authenticated:
                target_records = stats_df.head(20)
                export_df = pd.DataFrame({
                    'Date': target_records['DisplayDate'], 'Category': target_records['Category'], 'Modality Class': target_records['ParentModality'],
                    'Platforms & Delivery': target_records['Platforms'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x),
                    'Cell Types': target_records['CellTypes'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x),
                    'Therapeutic Area': target_records['TA'], 'Target Disease': target_records['TargetDisease'], 'Development Stage': target_records['Stage'],
                    'Deal Value': target_records['DisplayValue'], 'Upfront Ratio': target_records['UpfrontRatio'].round(2), 
                    'Partner A': target_records['PartnerA'], 'Partner B': target_records['PartnerB'], 'Insight': target_records['Insight'], 'Source Link URL': target_records['Link']
                })
            else:
                target_records = stats_df.head(5)
                export_df = pd.DataFrame({
                    'Date': target_records['DisplayDate'], 'Modality Class': target_records['ParentModality'], 'Therapeutic Area': target_records['TA'], 
                    'Target Disease': target_records['TargetDisease'], 'Development Stage': target_records['Stage'], 'Partner A': target_records['PartnerA'], 'Partner B': target_records['PartnerB']
                })
            
            csv_payload = export_df.to_csv(index=False).encode('utf-8-sig')
            filename_stamp = datetime.now().strftime('%Y%m%d')
            
            if is_admin:
                st.download_button(label=f"📥 Download Unrestricted Master Sheet ({len(export_df)} Deals CSV)", data=csv_payload, file_name=f"SerGene_MASTER_{filename_stamp}.csv", mime="text/csv", key="cloud_admin_master_download_button")
            elif is_authenticated:
                if st.download_button(label=f"📥 Download Top {len(export_df)} Filtered Deals (CSV)", data=csv_payload, file_name=f"SerGene_Premium_{filename_stamp}.csv", mime="text/csv", key="cloud_premium_download_button"):
                    st.session_state["download_count"] += 1
                    log_audit_event(client_tag, "CSV Export Executed", f"Extracted {len(export_df)} items.")
            else:
                st.download_button(label=f"📥 Download Preview Data Extract ({len(export_df)} Deals CSV)", data=csv_payload, file_name=f"SerGene_Preview_{filename_stamp}.csv", mime="text/csv", key="cloud_preview_download_button")
    st.divider()

    # ==========================================
    # 7. DEAL CARDS DISPLAY
    # ==========================================
    CARD_HTML = """
    <div class="deal-card">
        <div style="display: flex; justify-content: space-between; align-items: start; gap: 2rem;">
            <div style="flex: 2;">
                <div class="date-badge">{d_date} | {ta} • {stage}</div>
                <div style="font-size: 0.75rem; color: #3b82f6; font-weight: bold; margin-bottom: 8px;">TARGET: {target}</div>
                <span class="parent-tag">{p_mod}</span>
                <h2><a href="{link}" target="_blank" class="source-link">{insight}</a></h2>
                <div style="font-weight: 700; color: #0f172a; font-size: 1.1rem;">{title}</div>
                <p class="summary-text">{summary}</p>
                <div>{tags}</div>
            </div>
            <div style="flex: 1; border-left: 2px solid #f1f5f9; padding-left: 2.5rem;">
                <p style="font-size: 0.7rem; color: #94a3b8;">DEAL VALUE ({cat_label})</p>
                <div class="{blur_financials_class}">
                    <p style="font-size: 1.85rem; font-weight: 900; color: #059669;">{value}</p>
                    <p style="font-size: 0.75rem; color: #059669; font-weight: 800; margin-bottom: 0;">{r_pct}% UPFRONT</p>
                    <div class="ratio-bar-container"><div style="height:100%; width:{r_pct}%; background:#10b981;"></div></div>
                </div>
                <p style="font-size: 0.7rem; color: #94a3b8; margin-top:1rem;">PARTNERS</p>
                <p style="font-weight: 800;">{pA}</p><p style="color: #64748b;">{pB}</p>
            </div>
        </div>
    </div>
    """

    if stats_df.empty:
        st.info("No matching deals found.")
    else:
        financial_blur = "" if is_authenticated else "blur-financials"
        for _, row in visible_df.iterrows():
            tags_h = "".join([f'<span class="tag">{html.escape(t)}</span>' for t in row['SubModalities']])
            r_val = int(round(row['UpfrontRatio'] * 100))
            st.markdown(CARD_HTML.format(
                d_date=row['DisplayDate'], ta=row['TA'], target=row['TargetDisease'], stage=row['Stage'], p_mod=row['ParentModality'], link=row['Link'], 
                insight=html.escape(row['Insight']), title=html.escape(row['Title']), summary=html.escape(row['Summary']), tags=tags_h, value=row['DisplayValue'], 
                r_pct=r_val, pA=row['PartnerA'], pB=row['PartnerB'], blur_financials_class=financial_blur, cat_label=row['Category']
            ), unsafe_allow_html=True)

    if not is_authenticated:
        mailto_link = "mailto:spiros@sergenebio.co.uk?subject=Access Request"
        st.markdown(f'<div class="cta-banner"><h2 style="color: #991b1b; margin-top: 0;">🔒 Unlock Full Access</h2><a href="{mailto_link}" style="text-decoration:none; color:white; background:#ef4444; padding:1rem 2rem; border-radius:0.75rem; font-weight:800; display:inline-block;">Request Access Code</a></div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"BI Module Error: {e}")
