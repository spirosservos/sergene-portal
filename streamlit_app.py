import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import io
import os
import plotly.express as px
import textwrap

# --- CONFIGURATION ---
st.set_page_config(
    page_title="SerGene Bio | Deal Intelligence Portal",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MASTER ORDER & CONSTANTS ---
MODALITIES = [
    "Small Molecule", "Biologics", "Protein Degrader", "Peptide", "GLP-1", "Incretin",
    "RNA", "mRNA", "siRNA", "RNAi", "miRNA", "ASO", "Antisense", "Aptamer",
    "CRISPR", "Gene Therapy", "Gene Editing", "Base Editing", "Prime Editing",
    "AAV", "Lentivirus", "Lenti", "Vector",
    "Tregs", "TCR", "CAR-T", "Cell Therapy", "NK Cells", "TILs",
    "ADC", "Antibody", "Bispecific", "Exosomes", "LNP", "Oral"
]

COLUMN_ORDER_PRIORITY = [
    "Date", "Title", "Partner A", "Partner B", 
    "Deal Value", "Upfront", "Milestones", "Royalties",
    "Summary", "Source"
]

# Brand Colors
SERGENE_BLUE = "#1A3D7C"
DISEASE_BROWN = "#8B4513"

# --- HELPER FOR STYLED TEXT ---
def color_text(text, color, bold=True):
    if not text or text == "" or text == "None":
        return text
    weight = "font-weight: 600;" if bold else ""
    return f'<span style="color: {color}; {weight}">{text}</span>'

# --- DATA LOADING ---
@st.cache_data(ttl=600) 
def load_data():
    try:
        if "gsheets_url" in st.secrets:
            df = pd.read_csv(st.secrets["gsheets_url"])
        else:
            file_path = "Biotech_Deals_Database.xlsx"
            if not os.path.exists(file_path): return pd.DataFrame()
            df = pd.read_excel(file_path)
        
        df['Date_Obj'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Filter_Date'] = df['Date_Obj'].dt.date
        
        rename_mapping = {
            "Lead Organization": "Partner A",
            "Partner": "Partner B",
            "Target Diseases": "Diseases", 
            "Total Deal Value": "Deal Value",
            "Upfront Payment": "Upfront",
            "Link": "Source",
            "Classification": "Type"
        }
        df = df.rename(columns=rename_mapping)
        
        text_cols = ['Title', 'Summary', 'Partner A', 'Partner B', 'Diseases', 'Type', 'Source', 'Deal Value', 'Upfront', 'Milestones', 'Royalties']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str).str.strip()
                # Zero-width space fix for older iOS math-mode crash
                df[col] = df[col].apply(lambda x: x.replace('$', '$' + '\u200b'))
                
                if col in ['Partner A', 'Partner B']:
                    df[col] = df[col].apply(lambda x: x.title() if x.islower() else x)

        all_cols = list(df.columns)
        for col in all_cols:
            if col not in ['Date', 'Date_Obj', 'Filter_Date']:
                df[col] = df[col].astype(str).replace(["nan", "0", "0.0", "None", "False"], "")

        return df
    except Exception as e:
        return pd.DataFrame()

df_raw = load_data()

# --- AUTHENTICATION SYSTEM ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        return True

    st.markdown(f"<h1 style='text-align: center; color: {SERGENE_BLUE};'>🧬 SerGene Bio | Intelligence Portal</h1>", unsafe_allow_html=True)
    
    if not df_raw.empty:
        st.markdown("### 🔍 Latest Market Intelligence (Public Preview)")
        preview_df = df_raw.sort_values('Date_Obj', ascending=False).head(10).copy()
        
        # Apply branding to public preview table (HTML style)
        if "Partner A" in preview_df.columns:
            preview_df["Partner A"] = preview_df["Partner A"].apply(lambda x: color_text(x, SERGENE_BLUE))
        if "Partner B" in preview_df.columns:
            preview_df["Partner B"] = preview_df["Partner B"].apply(lambda x: color_text(x, SERGENE_BLUE))
        if "Diseases" in preview_df.columns:
            preview_df["Diseases"] = preview_df["Diseases"].apply(lambda x: color_text(x, DISEASE_BROWN))
        
        cols_to_show = ["Date", "Title", "Partner A", "Partner B", "Deal Value"]
        existing_cols = [c for c in cols_to_show if c in preview_df.columns]
        
        # Wrap in div for scrolling
        st.markdown(f'<div style="overflow-x: auto;">{preview_df[existing_cols].to_html(escape=False, index=False)}</div>', unsafe_allow_html=True)
        st.info("💡 To access full summaries, source links, and historical data, please log in below.")

    st.divider()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("🔒 Secure Access")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if "users" in st.secrets:
            VALID_USERS = st.secrets["users"]
        else:
            VALID_USERS = {"admin": "admin123"}

        if st.button("Log In", use_container_width=True):
            if username in VALID_USERS and str(VALID_USERS[username]) == password:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("😕 Incorrect username or password")
    
    st.markdown("<p style='text-align: center; color: gray; font-size: 0.8rem;'>Contact SerGene Bio admin for access credentials.</p>", unsafe_allow_html=True)
    return False

if not check_password():
    st.stop()

# --- CUSTOM CSS ---
st.markdown(f"""
<style>
    .block-container {{ padding-top: 1rem; padding-bottom: 2rem; }}
    [data-testid="stMetric"] {{ background-color: #f8f9fa; border: 1px solid #e2e8f0; padding: 15px; border-radius: 10px; }}
    thead tr th {{ font-weight: 800 !important; background-color: #f1f5f9 !important; color: {SERGENE_BLUE} !important; }}
    .stDownloadButton button {{ background-color: {SERGENE_BLUE} !important; color: white !important; border-radius: 8px; width: 100%; }}
    
    /* Hide built-in download button in toolbar */
    [data-testid="stElementToolbar"] button[title="Download as CSV"],
    [data-testid="stElementToolbar"] button[aria-label="Download as CSV"] {{
        display: none !important;
    }}

    /* Block Copy-Paste */
    .stDataFrame, .stTable, [data-testid="stTable"], [data-testid="stDataFrame"] {{
        -webkit-user-select: none;
        -moz-user-select: none;
        -ms-user-select: none;
        user-select: none;
    }}
    
    /* Styling for Reading Mode HTML Tables */
    .reading-table-container {{
        overflow-x: auto;
        margin-top: 1rem;
    }}
    .reading-table-container table {{
        width: 100%;
        border-collapse: collapse;
    }}
</style>
<script>
document.addEventListener('contextmenu', event => event.preventDefault());
</script>
""", unsafe_allow_html=True)

# --- SIDEBAR & FILTERS ---
st.sidebar.title("🧬 SerGene Bio")
st.sidebar.write(f"Logged in as: **{st.session_state.get('username', 'User')}**")

if st.sidebar.button("Log Out"):
    st.session_state["authenticated"] = False
    st.rerun()

st.sidebar.divider()

if not df_raw.empty:
    st.sidebar.subheader("👁️ Viewing Mode")
    # UPDATED: Set "Reading Mode" as the default (index 1) so colors are visible immediately
    view_mode = st.sidebar.radio("Layout:", ["Interactive Grid", "Reading Mode"], index=1, label_visibility="collapsed")
    consolidate = st.sidebar.toggle("Consolidate Reports", value=True)

    with st.sidebar.expander("📅 Date Range", expanded=True):
        valid_dates = df_raw.loc[df_raw['Filter_Date'].notnull(), 'Filter_Date']
        UI_MIN = date(1980, 1, 1)
        UI_MAX = date(2030, 12, 31)
        default_start = valid_dates.min() if not valid_dates.empty else date(2023, 1, 1)
        default_end = valid_dates.max() if not valid_dates.empty else date.today()
        start_date = st.date_input("From", default_start, min_value=UI_MIN, max_value=UI_MAX)
        end_date = st.date_input("To", default_end, min_value=UI_MIN, max_value=UI_MAX)

    with st.sidebar.expander("🧬 Modality Filter", expanded=False):
        valid_mods = [m for m in MODALITIES if m in df_raw.columns]
        selected_mods = st.multiselect("Select Modalities", valid_mods)

    available_cols = [c for c in df_raw.columns if c not in ['Date_Obj', 'Filter_Date', 'ID', 'Sources_All']]
    pref_cols = [c for c in COLUMN_ORDER_PRIORITY if c in available_cols]
    
    # User Reordering logic
    with st.sidebar.expander("👁️ Customize View"):
        selected_columns = st.multiselect("Display Columns", available_cols, default=pref_cols)
else:
    st.sidebar.warning("No data found.")

# --- MAIN DASHBOARD ---
st.title("🧬 Deal Intelligence Portal")
search_query = st.text_input("", placeholder="🔍 Search full database...", label_visibility="collapsed")

if not df_raw.empty:
    df = df_raw.copy()
    
    # Filtering
    df = df[(df['Filter_Date'] >= start_date) & (df['Filter_Date'] <= end_date)]
    if selected_mods:
        mod_mask = df[selected_mods].apply(lambda x: x.astype(str).str.strip() != "").any(axis=1)
        df = df[mod_mask]
    if search_query:
        q = search_query.lower()
        search_cols = [c for c in ['Title', 'Summary', 'Partner A', 'Partner B', 'Diseases'] if c in df.columns]
        mask = df[search_cols].apply(lambda x: x.str.lower().str.contains(q, na=False)).any(axis=1)
        df = df[mask]

    if consolidate and not df.empty:
        df = df.sort_values('Score', ascending=False)
        agg_funcs = {col: 'first' for col in df.columns}
        if 'Source' in df.columns:
            df['Sources_All'] = df['Source']
            agg_funcs['Sources_All'] = lambda x: " | ".join(list(set([str(link) for link in x if link])))
            agg_funcs['Source'] = 'first'
        df = df.groupby(['Partner A', 'Partner B', 'Filter_Date'], as_index=False).agg(agg_funcs)

    # Export Logic
    st.sidebar.divider()
    st.sidebar.subheader("📥 Export Data")
    download_df = df.sort_values(by='Date_Obj', ascending=False).head(15).copy()
    for col in download_df.columns:
        if download_df[col].dtype == 'object':
            download_df[col] = download_df[col].apply(lambda x: str(x).replace('\u200b', '') if isinstance(x, str) else x)
    
    st.sidebar.download_button(
        label="Download Latest 15 Deals (CSV)",
        data=download_df.to_csv(index=False).encode('utf-8'),
        file_name=f"SerGene_Deals_Export_{date.today()}.csv",
        mime="text/csv"
    )

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Database Matches", len(df))
    if 'Score' in df.columns:
        scores = pd.to_numeric(df['Score'], errors='coerce').dropna()
        m2.metric("Avg Quality", round(scores.mean(), 1) if not scores.empty else 0)
    
    lead_m = df[df['Partner A'] != ""]['Partner A'].mode()
    m3.metric("Most Active", lead_m[0] if not lead_m.empty else "N/A")
    
    dis_m = df[df['Diseases'] != ""]['Diseases'].mode()
    m4.metric("Key Area", str(dis_m[0]).split(";")[0] if not dis_m.empty else "N/A")

    st.divider()

    # View Selection
    tab_data, tab_charts = st.tabs(["📊 Data Explorer", "📈 Visual Analytics"])

    with tab_data:
        if not df.empty:
            # Respect selection order exactly
            final_cols = selected_columns
            df_display = df.sort_values(by='Date_Obj', ascending=False)
            
            if view_mode == "Interactive Grid":
                # Reminder: Grid does not support branding colors
                st.dataframe(df_display[final_cols], column_config={
                    "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                    "Source": st.column_config.LinkColumn("Source", display_text="Read"),
                }, hide_index=True, use_container_width=True)
                st.caption("ℹ️ Note: Interactive Grid is for fast sorting/searching. Use 'Reading Mode' for branding colors.")
            else:
                # Reading Mode Styling (HTML)
                html_df = df_display[final_cols].copy()
                if "Partner A" in html_df.columns:
                    html_df["Partner A"] = html_df["Partner A"].apply(lambda x: color_text(x, SERGENE_BLUE))
                if "Partner B" in html_df.columns:
                    html_df["Partner B"] = html_df["Partner B"].apply(lambda x: color_text(x, SERGENE_BLUE))
                if "Diseases" in html_df.columns:
                    html_df["Diseases"] = html_df["Diseases"].apply(lambda x: color_text(x, DISEASE_BROWN))
                
                def make_links(row):
                    val = row.get('Sources_All') or row.get('Source')
                    return " , ".join([f'<a href="{l.strip()}" target="_blank">Source {i+1}</a>' for i, l in enumerate(str(val).split(" | "))]) if val else ""
                
                if "Source" in html_df.columns:
                    html_df["Source"] = df_display.apply(make_links, axis=1)
                
                # Wrapped HTML for horizontal scrolling
                st.markdown(f'<div class="reading-table-container">{html_df.to_html(escape=False, index=False)}</div>', unsafe_allow_html=True)
        else:
            st.info("No matches found for current filters.")

    with tab_charts:
        if not df.empty:
            st.subheader("🕵️ Transaction Timeline")
            timeline_df = df.copy().sort_values('Date_Obj')
            fig_timeline = px.scatter(
                timeline_df, x="Date_Obj", y="Score", color_discrete_sequence=[SERGENE_BLUE],
                hover_name="Title", hover_data={"Date_Obj": "|%Y-%m-%d", "Partner A": True, "Partner B": True},
                template="plotly_white"
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.warning("Please adjust filters to see visualizations.")

st.markdown("---")
st.caption(f"© {datetime.now().year} SerGene Bio | Deal Intelligence Platform")
