import streamlit as st
import pandas as pd
import html
from datetime import datetime

# 1. Page Configuration & Custom CSS
st.set_page_config(page_title="SerGene Intelligence", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stApp { background-color: #f8fafc; }
    .deal-card {
        background-color: white;
        padding: 2.25rem;
        border-radius: 1.5rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    .insight-text {
        color: #3b82f6;
        font-size: 1.4rem;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 0.4rem;
    }
    .deal-title {
        color: #0f172a;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 1.25rem;
        letter-spacing: -0.01em;
    }
    .summary-text {
        color: #475569;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }
    .score-badge {
        background-color: #059669;
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 0.75rem;
        font-weight: 800;
        font-size: 0.9rem;
        letter-spacing: 0.05em;
        display: inline-block;
        margin-bottom: 1rem;
    }
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
    .lock-banner {
        background-color: #fef2f2;
        color: #991b1b;
        padding: 1.25rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 800;
        border: 1px solid #fee2e2;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Load Data
@st.cache_data
def load_data():
    df = pd.read_feather("sg_intel_assets.arrow")
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # We keep the actual datetime objects for filtering, but create a string version for display
        df = df.sort_values(by='Date', ascending=False)
        df['DisplayDate'] = df['Date'].dt.strftime('%Y-%m-%d').fillna('N/A')
    return df

try:
    df = load_data()

    # 3. Sidebar Filters & Authentication
    st.sidebar.title("SerGene Intel")
    
    with st.sidebar.expander("🔐 Secure Access", expanded=True):
        try:
            CORRECT_PASSWORD = st.secrets["access_password"]
        except:
            CORRECT_PASSWORD = "SerGene2024"
            
        password_input = st.text_input("Access Code", type="password")
        is_authenticated = (password_input == CORRECT_PASSWORD)
        
        if is_authenticated:
            st.success("Full Access Granted")
        else:
            st.info("Guest Mode: Latest 20 results")

    st.sidebar.divider()
    
    # NEW: Date Range Filter
    min_date = df['Date'].min().to_pydatetime()
    max_date = df['Date'].max().to_pydatetime()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # NEW: Modalities Filter (Above Category)
    all_modalities = sorted(list(set([tag for tags in df['ModalityTags'] for tag in tags if tag])))
    selected_modalities = st.sidebar.multiselect("Filter by Modalities", all_modalities)
    
    categories = ["All"] + sorted(df['Category'].unique().tolist())
    category_filter = st.sidebar.selectbox("Category", categories)
    
    search = st.sidebar.text_input("🔍 Search Database")
    
    # 4. Filter Logic
    filtered_df = df.copy()
    
    # Apply Date Range Filter
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['Date'].dt.date >= start_date) & 
            (filtered_df['Date'].dt.date <= end_date)
        ]
    
    # Apply Modalities Filter
    if selected_modalities:
        filtered_df = filtered_df[filtered_df['ModalityTags'].apply(lambda tags: any(m in tags for m in selected_modalities))]
    
    # Apply Category Filter
    if category_filter != "All":
        filtered_df = filtered_df[filtered_df['Category'] == category_filter]
        
    # Apply Search
    if search:
        filtered_df = filtered_df[
            filtered_df['Insight'].str.contains(search, case=False) | 
            filtered_df['Title'].str.contains(search, case=False) |
            filtered_df['PartnerA'].str.contains(search, case=False)
        ]
        
    display_df = filtered_df if is_authenticated else filtered_df.head(20)

    # 5. Main Display
    st.title("Strategic Intelligence Stream")
    
    if not is_authenticated:
        st.markdown('<div class="lock-banner">🔒 Preview Mode: Showing 20 most recent assets. Enter code in sidebar for full database.</div>', unsafe_allow_html=True)
    
    st.caption(f"Displaying {len(display_df)} intelligence records")

    for _, row in display_df.iterrows():
        # Sanitize data for HTML safety
        s_insight = html.escape(str(row['Insight']))
        s_title = html.escape(str(row['Title']))
        s_summary = html.escape(str(row['Summary']))
        s_partnerA = html.escape(str(row['PartnerA']))
        s_partnerB = html.escape(str(row['PartnerB']))
        s_value = html.escape(str(row['DealValue']))
        s_date = html.escape(str(row['DisplayDate']))
        s_cat = html.escape(str(row['Category']))
        s_score = html.escape(str(row['Score']))

        tags_html = ' '.join([f'<span class="tag">{html.escape(tag)}</span>' for tag in row['ModalityTags']])
        
        st.markdown(f"""
        <div class="deal-card">
            <div style="display: flex; justify-content: space-between; align-items: start; flex-wrap: wrap; gap: 1rem;">
                <div style="flex: 2; min-width: 300px;">
                    <div class="score-badge">PT SCORE: {s_score}</div>
                    <p style="color: #94a3b8; font-size: 0.75rem; font-weight: 800; margin-top: 0.2rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">
                        {s_date} • {s_cat}
                    </p>
                    <h2 class="insight-text">{s_insight}</h2>
                    <div class="deal-title">{s_title}</div>
                    <p class="summary-text">{s_summary}</p>
                    <div style="margin-top: 1.25rem;">
                        {tags_html}
                    </div>
                </div>
                <div style="flex: 1; min-width: 250px; border-left: 2px solid #f1f5f9; padding-left: 2rem;">
                    <div style="margin-bottom: 1.5rem;">
                        <p style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">Primary Partner</p>
                        <p style="font-weight: 800; color: #0f172a; font-size: 1.15rem; line-height: 1.2;">{s_partnerA}</p>
                        <p style="font-weight: 600; color: #64748b; font-size: 0.85rem; margin-top: 0.25rem;">{s_partnerB}</p>
                    </div>
                    <div>
                        <p style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.2rem;">Deal Value</p>
                        <p style="font-size: 1.75rem; font-weight: 900; color: #059669; letter-spacing: -0.02em;">{s_value}</p>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Failed to initialize database: {e}")
