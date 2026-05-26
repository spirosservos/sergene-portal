import streamlit as st
import pandas as pd
import numpy as np
import html
import re
import os
import textwrap  
import datetime  # Integrated for native, reliable date arithmetic on cloud environments
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
    "Cell Therapy": ["CAR-T", "TCR", "TILs", "NK Cells", "Tregs", "MSCs", "iPSCs", "gamma delta T cells", "γδ T cells", "Cell Therapy"],
    "RNA Therapeutics": ["mRNA", "siRNA", "RNAi", "miRNA", "ASO", "Antisense", "Aptamer", "RNA", "ASO / Antisense"],
    "Immunotherapies": ["Oncolytic Virus", "Immuno-oncology"],
    "Biologics": ["Antibody", "Bispecific", "ADC", "Multi-specific", "Peptide", "Biologics", "Exosomes"],
    "Small Molecule": ["Small Molecule", "Protein Degrader", "Oral"]
}

# Strict filter to classify cell types versus engineering/delivery platforms
CELL_THERAPY_TAGS = ["CAR-T", "TCR", "TILs", "NK Cells", "Tregs", "MSCs", "iPSCs", "gamma delta T cells", "γδ T cells", "Cell Therapy"]

# Strict explicit ordering arrays for filters to avoid random alphabetical sorting
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
# 3. CSS STYLING
# ==========================================
st.markdown("""
    <style>
    .main, .stApp { background-color: #f8fafc; }
    .deal-card {
        background-color: white; padding: 2.5rem; border-radius: 1.5rem;
        border: 1px solid #e2e8f0; margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
    .blurred-card { filter: blur(8px); opacity: 0.5; pointer-events: none; }
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
    # Master schema design map protects against empty file KeyError crashes on Cloud builds
    columns_template = [
        'Row_ID', 'Date', 'Date_Obj', 'DisplayDate', 'ParentModality', 'SubModalities',
        'CellTypes', 'Platforms', 'TA', 'TargetDisease', 'Stage', 'TotalValueM',
        'UpfrontRatio', 'DisplayValue', 'PartnerA', 'PartnerB', 'Insight', 'Title',
        'Summary', 'Link', 'SearchBlob'
    ]
    
    if not os.path.exists("sg_intel_assets.arrow"):
        return pd.DataFrame(columns=columns_template)
        
    df = pd.read_feather("sg_intel_assets.arrow") 
    if df.empty:
        return pd.DataFrame(columns=columns_template)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(by='Date', ascending=False)
    
    # Strict whitelist to isolate scientific modalities and prevent data leakage
    TECH_COLUMNS = [
        'Small Molecule', 'Biologics', 'Protein Degrader', 'Peptide', 'GLP-1', 'Incretin', 
        'RNA', 'mRNA', 'siRNA', 'RNAi', 'miRNA', 'ASO', 'Antisense', 'Aptamer', 
        'CRISPR', 'Gene Therapy', 'Gene Editing', 'Base Editing', 'Prime Editing', 
        'AAV', 'Lentivirus', 'Lenti', 'Oncolytic Virus', 'Vector', 'Tregs', 'TCR', 
        'CAR-T', 'Cell Therapy', 'NK Cells', 'TILs', 'ADC', 'Antibody', 'Bispecific', 
        'Exosomes', 'LNP', 'Oral', 'Radiopharmaceutical', 'Immuno-oncology', 
        'Multi-specific', 'Nanoparticle', 'MSCs', 'iPSCs', 'gamma delta T cells', 'γδ T cells'
    ]
    
    refined_rows = []
    for _, row in df.iterrows():
        # First pass: gather all raw tags detected on this row
        raw_row_flags = []
        for col_name in row.index:
            if col_name in TECH_COLUMNS:
                val = row[col_name]
                try:
                    if float(val) > 0:
                        raw_row_flags.append(col_name)
                except:
                    if str(val).lower() in ['yes', 'y', 'true', '1']:
                        raw_row_flags.append(col_name)

        # Second pass: Apply exclusivity rules to eliminate overlapping redundancies
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
                tags.append("γδ T cells")
            else:
                tags.append(flag)

        tags = list(set([t for t in tags if t and str(t).lower() != 'nan']))
        
        # Split tags cleanly between Cell Types and Engineering/Delivery Platforms
        cell_types_extracted = [t for t in tags if t in CELL_THERAPY_TAGS]
        platforms_extracted = [t for t in tags if t not in CELL_THERAPY_TAGS]
        
        # B. Modality Groups Mapping
        parent = "Emerging Platforms & Conjugates"  
        norm_tags = [t.lower() for t in tags]
        for group_name, keywords in MODALITY_GROUPS.items():
            if any(k.lower() in norm_tags for k in keywords):
                parent = group_name
                break
        
        val_m = parse_currency(row.get('DealValue', ''))
        up_m = parse_currency(row.get('Upfront', ''))
        ratio = (up_m / val_m) if val_m > 0 else 0.0

        # C. Search Blob Creation
        row_values = []
        for val in row.values:
            if isinstance(val, (list, np.ndarray)):
                row_values.extend([str(i) for i in val if pd.notna(i)])
            elif pd.notna(val):
                row_values.append(str(val))
        
        blob = " ".join(row_values).lower().replace('\xa0', ' ')
        blob = " ".join(blob.split())

        refined_rows.append({
            'Row_ID': row.name, 
            'Date': row.get('Date'),
            'Date_Obj': row.get('Date').date() if pd.notnull(row.get('Date')) else None,
            'DisplayDate': row.get('Date').strftime('%b %d, %Y') if pd.notnull(row.get('Date')) else "N/A",
            'ParentModality': parent,
            'SubModalities': tags,
            'CellTypes': cell_types_extracted,
            'Platforms': platforms_extracted,
            'TA': str(row.get('TA', 'Other/General')).strip(),
            'TargetDisease': str(row.get('Target Disease', row.get('TargetDisease', 'N/A'))),
            'Stage': str(row.get('Stage', 'Pre-clinical')).strip(),
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
try:
    df_master = load_and_refine_data()
    df_master = df_master.dropna(subset=['Date_Obj']).sort_values('Date_Obj', ascending=False)

    st.sidebar.title("🧬 SerGene Intelligence")
    st.sidebar.markdown("---")

    # 1. Select Timeframe & Date Range
    st.sidebar.subheader("📅 Select Timeframe")
    if df_master.empty:
        min_db = datetime.date.today() - datetime.timedelta(days=30)
        max_db = datetime.date.today()
    else:
        min_db = df_master['Date_Obj'].min()
        max_db = df_master['Date_Obj'].max()
        
    date_sel = st.sidebar.date_input("Date Range", value=(min_db, max_db), min_value=min_db, max_value=max(max_db, datetime.date.today()))

    st.sidebar.divider()

    # 2. Client Access
    is_authenticated = False
    with st.sidebar.expander("🔑 Client Access", expanded=False):
        secret_pass = st.secrets.get("access_password")
        password_input = st.text_input("Enter Access Code", type="password")
        if password_input == secret_pass and secret_pass:
            is_authenticated = True
            st.success("Full Access Granted")
        if not is_authenticated:
            st.markdown("---")
            st.caption("Contact Support for Code:")
            st.code("spiros@sergenebio.co.uk")

    st.sidebar.divider()

    # 3. Modality Class
    existing_parents = df_master['ParentModality'].unique().tolist() if not df_master.empty else []
    sorted_parents_options = [m for m in MODALITY_ORDER if m in existing_parents] + [m for m in existing_parents if m not in MODALITY_ORDER]
    sel_parents = st.sidebar.multiselect("Modality Class", sorted_parents_options)
    
    # 4. Platforms & Delivery Dropdown
    all_platforms = list(set([p for sub in df_master['Platforms'] for p in sub])) if not df_master.empty else []
    sorted_platform_options = [p for p in PLATFORM_ORDER if p in all_platforms] + [p for p in all_platforms if p not in PLATFORM_ORDER]
    sel_platforms = st.sidebar.multiselect("Platforms & Delivery", sorted_platform_options)

    # 5. Cell Types Dropdown
    all_cells = sorted(list(set([c for sub in df_master['CellTypes'] for c in sub]))) if not df_master.empty else []
    sel_cells = st.sidebar.multiselect("Cell Types", all_cells)

    # 6. Therapeutic Area
    sel_tas = st.sidebar.multiselect("Therapeutic Area", sorted(df_master['TA'].unique().tolist())) if not df_master.empty else []

    # 7. Development Stage
    sel_stages = st.sidebar.multiselect("Development Stage", sorted(df_master['Stage'].unique().tolist())) if not df_master.empty else []
    
    # 8. Search Everything (Deep Scan)
    search_term = st.sidebar.text_input("🔍 Search Everything (Deep Scan)")

    # --- FILTERING ENGINE ---
    stats_df = df_master.copy()

    if isinstance(date_sel, (list, tuple)) and len(date_sel) == 2:
        stats_df = stats_df[(stats_df['Date_Obj'] >= date_sel[0]) & (stats_df['Date_Obj'] <= date_sel[1])]
    
    if len(sel_parents) > 0: stats_df = stats_df[stats_df['ParentModality'].isin(sel_parents)]
    if len(sel_platforms) > 0:
        stats_df = stats_df[stats_df['Platforms'].apply(lambda x: any(s in x for s in sel_platforms))]
    if len(sel_cells) > 0:
        stats_df = stats_df[stats_df['CellTypes'].apply(lambda x: any(s in x for s in sel_cells))]
    if len(sel_tas) > 0: stats_df = stats_df[stats_df['TA'].isin(sel_tas)]
    if len(sel_stages) > 0: stats_df = stats_df[stats_df['Stage'].isin(sel_stages)]
    
    if search_term:
        stats_df = stats_df[stats_df['SearchBlob'].str.contains(search_term.lower(), na=False)]

    GLOBAL_PREVIEW_LIMIT = 5
    BLUR_LIMIT = 3
    visible_df = stats_df if is_authenticated else stats_df.head(GLOBAL_PREVIEW_LIMIT)

    # ==========================================
    # 6. DASHBOARD
    # ==========================================
    st.title("Strategic Deal Intelligence Stream")
    
    # Process and calculate unique companies based on first word extraction
    if not stats_df.empty:
        partners_combined = pd.concat([stats_df['PartnerA'], stats_df['PartnerB']]).dropna()
        excluded_placeholders = ['n/a', 'nan', '', 'locked', 'unknown']
        partners_combined = partners_combined[~partners_combined.astype(str).str.lower().isin(excluded_placeholders)]
        company_first_words = partners_combined.astype(str).apply(lambda x: x.split()[0].lower() if len(x.split()) > 0 else '')
        unique_companies_count = company_first_words[company_first_words != ''].nunique()
    else:
        unique_companies_count = 0
    
    # 4-Column Layout
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Database Depth", f"{len(stats_df)} Deals")
    m2.metric("Companies Tracked", f"{unique_companies_count} Unique")
    m3.metric("Market Volume Analysed", f"${stats_df['TotalValueM'].sum()/1000:.1f}B" if not stats_df.empty else "$0.0B")
    
    valid_r = stats_df[stats_df['UpfrontRatio'] > 0]['UpfrontRatio'] if not stats_df.empty else pd.Series()
    avg_r = valid_r.mean() if not valid_r.empty else 0
    m4.metric("Avg. Upfront Ratio", f"{avg_r:.1%}")

    st.divider()

    # ==========================================
    # 6.2 CHRONOLOGICAL NEWS GRAPHIC
    # ==========================================
    lookback_days = 30 if is_authenticated else 7
    label_text = "Month" if is_authenticated else "Week"
    
    if not stats_df.empty:
        global_latest_date = df_master['Date_Obj'].max()
        cutoff_date = global_latest_date - datetime.timedelta(days=lookback_days)
        timeline_df = stats_df[stats_df['Date_Obj'] >= cutoff_date].copy()
        
        if not timeline_df.empty:
            timeline_df = timeline_df.sort_values('Date_Obj', ascending=True)
            timeline_df['stack_y'] = timeline_df.groupby('Date_Obj').cumcount() + 1
            
            current_available_order = [o for o in MODALITY_ORDER if o in timeline_df['ParentModality'].unique()]
            timeline_df['ParentModality'] = pd.Categorical(timeline_df['ParentModality'], categories=current_available_order, ordered=True)
            timeline_df = timeline_df.sort_values(['Date_Obj', 'ParentModality'])

            hover_meta_list = []
            visible_row_ids = visible_df['Row_ID'].values if 'Row_ID' in visible_df.columns else []
            
            for _, r in timeline_df.iterrows():
                if not is_authenticated and r['Row_ID'] not in visible_row_ids:
                    text_html = (
                        "<span style='font-size:16px; font-family:Arial, sans-serif; color:#64748b; padding:10px;'>"
                        "<b>📅 DATE:</b> %{x}<br><br>"
                        "<b style='color:#ef4444;'>🔒 STATUS: Premium Client Account Required</b><br>"
                        "Activate your access code to unlock real-time dashboard analytics.</span>"
                    )
                else:
                    raw_insight = r['Insight'] if r['Insight'] else ""
                    wrapped_insight = "<br>".join(textwrap.wrap(html.escape(raw_insight), width=70))
                    
                    text_html = (
                        f"<span style='font-size:15px; font-family:Arial, sans-serif; line-height:1.6; color:#0f172a;'>"
                        f"<b style='color:#2563eb;'>📅 DATE:</b> {r['DisplayDate']}<br>"
                        f"<b style='color:#059669;'>🤝 PARTNERS:</b> {html.escape(r['PartnerA'])} & {html.escape(r['PartnerB'])}<br>"
                        f"<b style='color:#d97706;'>💰 VALUE:</b> {html.escape(r['DisplayValue'])}<br>"
                        f"<b style='color:#7c3aed;'>🧬 CLASS:</b> {html.escape(r['ParentModality'])}<br>"
                        f"<b style='color:#0284c7;'>🎯 TARGET:</b> {html.escape(r['TargetDisease'])}<br><br>"
                        f"<b style='color:#dc2626;'>💡 STRATEGIC INSIGHT:</b><br>"
                        f"<i style='color:#334155;'>{wrapped_insight}</i>"
                        f"</span>"
                    )
                hover_meta_list.append(text_html)
                
            timeline_df['HoverHTML'] = hover_meta_list
            
            fig_timeline = px.scatter(
                timeline_df,
                x='Date_Obj',
                y='stack_y',
                color='ParentModality',
                custom_data=['HoverHTML'], 
                color_discrete_map={
                    "Gene Therapy/Editing": "#3b82f6",                     
                    "Cell Therapy": "#10b981",                             
                    "RNA Therapeutics": "#6366f1",                         
                    "Immunotherapies": "#ec4899",                          
                    "Biologics": "#f59e0b",                                
                    "Small Molecule": "#b91c1c",                           
                    "Emerging Platforms & Conjugates": "#64748b"            
                },
                category_orders={"ParentModality": MODALITY_ORDER}
            )
            
            fig_timeline.update_traces(
                marker=dict(size=18, opacity=0.85, line=dict(width=1.5, color='#ffffff')),
                hovertemplate="%{customdata[0]}<extra></extra>" 
            )
            
            fig_timeline.update_layout(
                title=dict(
                    text=f"🧬 Latest Deal Intelligence Timeline (Last {label_text} of Activity)",
                    font=dict(size=16, color='#1e293b', weight='bold')
                ),
                plot_bgcolor='#ffffff',
                paper_bgcolor='rgba(0,0,0,0)',
                hovermode='closest',
                hoverdistance=3, 
                hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e2e8f0"),
                xaxis=dict(title=None, showgrid=True, gridcolor='#f1f5f9', tickfont=dict(color='#64748b', size=12), type='date'),
                yaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False),
                legend=dict(
                    title=dict(text="Modality Class", font=dict(size=12, weight='bold')),
                    orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5
                ),
                margin=dict(l=10, r=10, t=50, b=80), 
                height=320 
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info(f"No transactions recorded during the immediate 1-{label_text} timeframe window.")
    else:
        st.info("No transaction coordinates available to map trend visualizations.")

    st.divider()

    with st.expander("📈 Market Trends & Competitive Landscape", expanded=False):
        if not stats_df.empty:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("### **Modality Mix**")
                st.bar_chart(stats_df['ParentModality'].value_counts(), color="#3b82f6")
            with c2:
                st.markdown("### **Therapeutic Focus**")
                st.bar_chart(stats_df['TA'].value_counts(), color="#10b981")
            with c3:
                st.markdown("### **Development Stage**")
                st.bar_chart(stats_df['Stage'].value_counts(), color="#6366f1")
        else:
            st.info("No metric data available to generate trend plots.")

    # AI Section
    st.write("") 
    if is_authenticated:
        ai_ready = st.toggle("Enable AI Strategic Analysis Tool", value=False)
        if ai_ready:
            if st.button("🪄 Generate AI Strategic Brief"):
                with st.status("🤖 Analyzing current deal flow...", expanded=True):
                    deal_list = "\n".join([f"- {r['PartnerA']} & {r['PartnerB']}: {r['Insight']}" for _, r in stats_df.head(20).iterrows()]) if not stats_df.empty else ""
                    prompt = f"""
                    You are a Senior Biotech Strategic Analyst. Analyze these recent deals:
                    {deal_list}
                    
                    Provide a professional 3-point summary:
                    1. What is the biggest trend in this specific segment?
                    2. What does this suggest about the current market risk appetite?
                    3. A 1-sentence 'Strategic Outlook' for an investor.
                    """
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
    download_limit = 20 if is_authenticated else 5
    
    if not stats_df.empty:
        target_records = stats_df.head(download_limit)
        
        export_data = {
            'Date': target_records['DisplayDate'],
            'Modality Class': target_records['ParentModality'],
            'Platforms & Delivery': target_records['Platforms'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x),
            'Cell Types': target_records['CellTypes'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x),
            'Therapeutic Area': target_records['TA'],
            'Target Disease': target_records['TargetDisease'],
            'Development Stage': target_records['Stage'],
            'Deal Value': target_records['DisplayValue'],
            'Upfront Ratio': target_records['UpfrontRatio'].round(2), 
            'Partner A': target_records['PartnerA'],
            'Partner B': target_records['PartnerB'],
            'Insight': target_records['Insight'],
            'Title': target_records['Title'],
            'Summary': target_records['Summary'],
            'Link': target_records['Link']
        }
        
        export_df = pd.DataFrame(export_data)
        csv_payload = export_df.to_csv(index=False).encode('utf-8-sig')
        filename_stamp = datetime.now().strftime('%Y%m%d')
        
        if is_authenticated:
            st.info(f"Premium Target Active: Extracting up to {download_limit} deals based on your active sidebar criteria filters.")
            st.download_button(
                label=f"📥 Download Top {len(export_df)} Filtered Deals (CSV)", 
                data=csv_payload, 
                file_name=f"SerGene_Premium_Extract_{filename_stamp}.csv", 
                mime="text/csv"
            )
        else:
            st.warning(f"Free Version Active: Downloads are limited to a maximum of 5 deals. Activate client credentials to unlock up to 20 deals.")
            st.download_button(
                label=f"📥 Download Preview Data Extract ({len(export_df)} Deals CSV)", 
                data=csv_payload, 
                file_name=f"SerGene_Preview_Extract_{filename_stamp}.csv", 
                mime="text/csv"
            )
    else:
        st.info("No matching data entries are available to generate an extraction file.")

    st.divider()

    # ==========================================
    # 7. DEAL CARDS
    # ==========================================
    CARD_HTML = """
    <div class="deal-card {extra_class}">
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
                <p style="font-size: 0.7rem; color: #94a3b8;">DEAL VALUE</p>
                <p style="font-size: 1.85rem; font-weight: 900; color: #059669;">{value}</p>
                <p style="font-size: 0.75rem; color: #059669; font-weight: 800; margin-bottom: 0;">{r_pct}% UPFRONT</p>
                <div class="ratio-bar-container"><div style="height:100%; width:{r_pct}%; background:#10b981;"></div></div>
                <p style="font-size: 0.7rem; color: #94a3b8; margin-top:1rem;">PARTNERS</p>
                <p style="font-weight: 800;">{pA}</p><p style="color: #64748b;">{pB}</p>
            </div>
        </div>
    </div>
    """

    if stats_df.empty:
        st.info("No matching deals found for the current filters.")
    else:
        for _, row in visible_df.iterrows():
            tags_h = "".join([f'<span class="tag">{html.escape(t)}</span>' for t in row['SubModalities']])
            r_val = int(round(row['UpfrontRatio'] * 100))
            st.markdown(CARD_HTML.format(
                extra_class="", d_date=row['DisplayDate'], ta=row['TA'], target=row['TargetDisease'],
                stage=row['Stage'], p_mod=row['ParentModality'], link=row['Link'], 
                insight=html.escape(row['Insight']), title=html.escape(row['Title']), 
                summary=html.escape(row['Summary']), tags=tags_h, value=row['DisplayValue'], 
                r_pct=r_val, pA=row['PartnerA'], pB=row['PartnerB']
            ), unsafe_allow_html=True)

    # --- BLURRED CARDS & CTA BANNER ---
    if not is_authenticated:
        if not stats_df.empty and len(stats_df) > GLOBAL_PREVIEW_LIMIT:
            for _, row in stats_df.iloc[GLOBAL_PREVIEW_LIMIT : GLOBAL_PREVIEW_LIMIT + BLUR_LIMIT].iterrows():
                st.markdown(CARD_HTML.format(
                    extra_class="blurred-card", d_date=row['DisplayDate'], ta=row['TA'], target="LOCKED",
                    stage=row['Stage'], p_mod=row['ParentModality'], link="#", insight="LOCKED", 
                    title="LOCKED", summary="Unlock full access to view details.", tags="", 
                    value="$$$", r_pct=0, pA="LOCKED", pB="LOCKED"
                ), unsafe_allow_html=True)
            
        mailto_link = "mailto:spiros@sergenebio.co.uk?subject=Access Request"
        st.markdown(f"""
            <div class="cta-banner">
                <h2 style="color: #991b1b; margin-top: 0;">🔒 Unlock Full Historical Access</h2>
                <p style="font-size: 1.1rem; color: #b91c1c; margin-bottom: 1.5rem;">Analyze the full historical database and generate custom AI Strategic Briefs.</p>
                <a href="{mailto_link}" style="text-decoration:none; color:white; background:#ef4444; padding:1rem 2rem; border-radius:0.75rem; font-weight:800; font-size:1.1rem; display:inline-block;">Request Access Code</a>
                <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px dashed #fca5a5;">
                    <p style="font-size: 0.9rem; color: #7f1d1d; margin: 0;">Direct Inquiry: <b>spiros@sergenebio.co.uk</b></p>
                </div>
            </div>
        """, unsafe_allow_html=True)
            
except Exception as e:
    st.error(f"BI Module Error: {e}")
