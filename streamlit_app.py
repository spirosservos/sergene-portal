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

# --- 1. AUTHENTICATION SYSTEM (SECURITY UPGRADE) ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        return True

    st.title("🔒 SerGene Bio | Secure Access")
    col1, col2 = st.columns([1, 2])
    with col1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if "users" in st.secrets:
            VALID_USERS = st.secrets["users"]
        else:
            VALID_USERS = {"admin": "admin123"}

        if st.button("Log In"):
            if username in VALID_USERS and str(VALID_USERS[username]) == password:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("😕 Incorrect username or password")
    return False

if not check_password():
    st.stop()

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    [data-testid="stMetric"] { background-color: #f8f9fa; border: 1px solid #e0e0e0; padding: 15px; border-radius: 10px; }
    thead tr th { font-weight: 800 !important; background-color: #f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)

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

# --- DATA LOADING ---
@st.cache_data(ttl=3600) 
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
        
        text_cols = ['Title', 'Summary', 'Partner A', 'Partner B', 'Diseases', 'Type', 'Source']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str).str.strip()
                if col in ['Partner A', 'Partner B']:
                    df[col] = df[col].apply(lambda x: x.title() if x.islower() else x)

        all_cols = list(df.columns)
        for col in all_cols:
            if col not in ['Date', 'Date_Obj', 'Filter_Date']:
                df[col] = df[col].astype(str).replace(["nan", "0", "0.0", "None", "False"], "")

        return df
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return pd.DataFrame()

df_raw = load_data()

# --- SIDEBAR & FILTERS ---
st.sidebar.title("🧬 SerGene Bio")
st.sidebar.write(f"Logged in: **{st.session_state.get('username', 'User')}**")

if st.sidebar.button("Log Out"):
    st.session_state["authenticated"] = False
    st.rerun()

st.sidebar.divider()

if not df_raw.empty:
    st.sidebar.subheader("👁️ Viewing Mode")
    view_mode = st.sidebar.radio("Layout:", ["Interactive Grid", "Reading Mode"], label_visibility="collapsed")
    consolidate = st.sidebar.toggle("Consolidate Reports", value=True)

    with st.sidebar.expander("📅 Date Range", expanded=True):
        UI_MIN = date(1980, 1, 1)
        UI_MAX = date(2030, 12, 31)
        valid_dates = df_raw.loc[df_raw['Filter_Date'].notnull(), 'Filter_Date']
        default_start = valid_dates.min() if not valid_dates.empty else date(2023, 1, 1)
        default_end = valid_dates.max() if not valid_dates.empty else date.today()
        start_date = st.date_input("From", default_start, min_value=UI_MIN, max_value=UI_MAX)
        end_date = st.date_input("To", default_end, min_value=UI_MIN, max_value=UI_MAX)

    with st.sidebar.expander("🧬 Modality Filter", expanded=False):
        valid_mods = [m for m in MODALITIES if m in df_raw.columns]
        selected_mods = st.multiselect("Select Modalities", valid_mods)

    available_cols = [c for c in df_raw.columns if c not in ['Date_Obj', 'Filter_Date', 'ID', 'Sources_All']]
    pref_cols = [c for c in COLUMN_ORDER_PRIORITY if c in available_cols]
    with st.sidebar.expander("👁️ Customize View"):
        selected_columns = st.multiselect("Display Columns", available_cols, default=pref_cols)
else:
    st.sidebar.warning("No data found.")

# --- MAIN DASHBOARD ---
st.title("🧬 Deal Intelligence Portal")
search_query = st.text_input("", placeholder="🔍 Search database...", label_visibility="collapsed")

# --- FILTERING & DE-DUPLICATION ---
if not df_raw.empty:
    df = df_raw.copy()
    
    # 1. Date Filtering
    df = df[(df['Filter_Date'] >= start_date) & (df['Filter_Date'] <= end_date)]

    # 2. Modality Filtering (FIXED)
    if selected_mods:
        mod_mask = df[selected_mods].apply(lambda x: x.astype(str).str.strip() != "").any(axis=1)
        df = df[mod_mask]

    # 3. Search Query
    if search_query:
        q = search_query.lower()
        search_cols = [c for c in ['Title', 'Summary', 'Partner A', 'Partner B', 'Diseases'] if c in df.columns]
        mask = df[search_cols].apply(lambda x: x.str.lower().str.contains(q, na=False)).any(axis=1)
        df = df[mask]

    # 4. Consolidation
    if consolidate and not df.empty:
        df = df.sort_values('Score', ascending=False)
        agg_funcs = {col: 'first' for col in df.columns}
        if 'Source' in df.columns:
            df['Sources_All'] = df['Source']
            agg_funcs['Sources_All'] = lambda x: " | ".join(list(set([str(link) for link in x if link])))
            agg_funcs['Source'] = 'first'
        df = df.groupby(['Partner A', 'Partner B', 'Filter_Date'], as_index=False).agg(agg_funcs)

    # --- TOP METRICS ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Unique Deals", len(df))
    if 'Score' in df.columns:
        scores = pd.to_numeric(df['Score'], errors='coerce').dropna()
        m2.metric("Avg Quality", round(scores.mean(), 1) if not scores.empty else 0)
    if 'Partner A' in df.columns:
        top_org = df[df['Partner A'] != ""]['Partner A'].mode()
        m3.metric("Most Active", top_org[0] if not top_org.empty else "N/A")
    if 'Diseases' in df.columns:
        top_dis = df[df['Diseases'] != ""]['Diseases'].mode()
        m4.metric("Key Area", str(top_dis[0]).split(";")[0] if not top_dis.empty else "N/A")

    st.divider()

    # --- TABS ---
    tab_data, tab_charts = st.tabs(["📊 Data Explorer", "📈 Visual Analytics"])

    with tab_data:
        if not df.empty:
            df = df.sort_values(by='Date_Obj', ascending=False)
            def sort_key(col):
                try: return COLUMN_ORDER_PRIORITY.index(col)
                except: return 999
            final_cols = sorted(selected_columns, key=sort_key)
            
            if view_mode == "Interactive Grid":
                st.dataframe(df[final_cols], column_config={
                    "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                    "Source": st.column_config.LinkColumn("Source", display_text="Read", width="small"),
                }, hide_index=True, use_container_width=True)
            else:
                def make_links(row):
                    val = row.get('Sources_All') or row.get('Source')
                    return " , ".join([f'<a href="{l.strip()}" target="_blank">Source {i+1}</a>' for i, l in enumerate(str(val).split(" | "))]) if val else ""
                html_df = df[final_cols].copy()
                if "Source" in html_df.columns: html_df["Source"] = df.apply(make_links, axis=1)
                st.markdown(html_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.info("No matches found for current filters.")

    with tab_charts:
        if not df.empty:
            st.subheader("🕵️ Transaction Timeline")
            timeline_df = df.copy().sort_values('Date_Obj')
            
            def wrap_summary(text, width=50):
                if not text: return ""
                short_text = text[:300] + "..." if len(text) > 300 else text
                return "<br>".join(textwrap.wrap(short_text, width=width))
                
            timeline_df['Hover_Summary'] = timeline_df['Summary'].apply(lambda x: wrap_summary(x))
            
            fig_timeline = px.scatter(
                timeline_df, 
                x="Date_Obj", 
                y="Score", 
                color="Type" if "Type" in timeline_df.columns else None,
                hover_name="Title", 
                hover_data={
                    "Date_Obj": "|%Y-%m-%d",
                    "Partner A": True,
                    "Partner B": True,
                    "Diseases": True,
                    "Deal Value": True,
                    "Hover_Summary": True,
                    "Score": False 
                },
                labels={
                    "Date_Obj": "Deal Date", 
                    "Partner A": "Lead Org", 
                    "Partner B": "Partner", 
                    "Diseases": "Diseases",
                    "Deal Value": "Value",
                    "Hover_Summary": "Summary"
                },
                template="plotly_white"
            )
            fig_timeline.update_traces(marker=dict(size=12, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
            fig_timeline.update_layout(
                hovermode='closest', 
                hoverdistance=10,
                hoverlabel=dict(bgcolor="white", font_size=12, align="left")
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            st.divider()
            
            c1, c2 = st.columns(2)
            with c1:
                volume_data = df.groupby(df['Date_Obj'].dt.to_period('M')).size().reset_index(name='Deals')
                volume_data['Date_Obj'] = volume_data['Date_Obj'].dt.to_timestamp()
                fig_trend = px.line(volume_data, x='Date_Obj', y='Deals', title="Transaction Volume Trend (Monthly)", line_shape='spline')
                st.plotly_chart(fig_trend, use_container_width=True)

                valid_orgs_plot = df[df['Partner A'] != ""]['Partner A']
                if not valid_orgs_plot.empty:
                    top_partners = valid_orgs_plot.value_counts().head(10).reset_index()
                    top_partners.columns = ['Organization', 'Count']
                    fig_partners = px.bar(top_partners, x='Count', y='Organization', orientation='h', title="Top 10 Active Strategic Partners", color='Count', color_continuous_scale='Blues')
                    fig_partners.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_partners, use_container_width=True)

            with c2:
                if 'Type' in df.columns:
                    valid_types_plot = df[df['Type'] != ""]['Type']
                    if not valid_types_plot.empty:
                        type_dist = valid_types_plot.value_counts().reset_index()
                        type_dist.columns = ['Type', 'Count']
                        fig_type = px.pie(type_dist, values='Count', names='Type', title="Transaction Type Distribution", hole=0.4)
                        st.plotly_chart(fig_type, use_container_width=True)

                mod_counts = []
                for mod in MODALITIES:
                    if mod in df.columns:
                        count = (df[mod].astype(str).str.len() > 0).sum()
                        if count > 0: mod_counts.append({'Modality': mod, 'Count': count})
                
                if mod_counts:
                    mod_df = pd.DataFrame(mod_counts).sort_values('Count', ascending=False)
                    fig_mod = px.bar(mod_df, x='Modality', y='Count', title="Prevalence of Modal Technologies", color='Count', color_continuous_scale='Viridis')
                    st.plotly_chart(fig_mod, use_container_width=True)
        else:
            st.warning("Please adjust filters to see visualizations.")

st.markdown("---")
st.caption(f"© {datetime.now().year} SerGene Bio | Deal Intelligence Platform")
