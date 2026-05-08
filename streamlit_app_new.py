import streamlit as st
import pandas as pd

# 1. Page Configuration & Custom CSS
st.set_page_config(page_title="SerGene Intelligence", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stApp { background-color: #f8fafc; }
    .deal-card {
        background-color: white;
        padding: 2rem;
        border-radius: 1.5rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .insight-text {
        color: #0f172a;
        font-size: 1.35rem;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 0.25rem;
    }
    .deal-title {
        color: #3b82f6;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }
    .summary-text {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    .score-badge {
        background-color: #059669;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .tag {
        display: inline-block;
        background-color: #f1f5f9;
        color: #475569;
        padding: 0.2rem 0.6rem;
        border-radius: 0.5rem;
        font-size: 0.7rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
        border: 1px solid #e2e8f0;
        text-transform: uppercase;
    }
    .lock-banner {
        background-color: #fef2f2;
        color: #991b1b;
        padding: 1rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        border: 1px solid #fee2e2;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Load Data
@st.cache_data
def load_data():
    # Streamlit handles Arrow IPC streams perfectly
    df = pd.read_feather("sg_intel_assets.arrow")
    # Ensure date sorting so latest is on top
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(by='Date', ascending=False)
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d').fillna('N/A')
    return df

try:
    df = load_data()

    # 3. Sidebar Filters & Authentication
    st.sidebar.title("SerGene Intel")
    
    # Password Protection Logic
    with st.sidebar.expander("🔐 Secure Access", expanded=True):
        password = st.text_input("Access Code", type="password")
        is_authenticated = (password == "SerGene2024")
        
        if is_authenticated:
            st.success("Full Access Granted")
        else:
            st.info("Guest Mode: Showing latest 20 results")

    st.sidebar.divider()
    search = st.sidebar.text_input("🔍 Search Insights or Partners")
    
    categories = ["All"] + sorted(df['Category'].unique().tolist())
    category_filter = st.sidebar.selectbox("Category", categories)
    
    # 4. Filter & Slicing Logic
    filtered_df = df.copy()
    
    # Apply Search
    if search:
        filtered_df = filtered_df[
            filtered_df['Insight'].str.contains(search, case=False) | 
            filtered_df['Title'].str.contains(search, case=False) |
            filtered_df['PartnerA'].str.contains(search, case=False)
        ]
    
    # Apply Category Filter
    if category_filter != "All":
        filtered_df = filtered_df[filtered_df['Category'] == category_filter]
        
    # Apply Guest Limit (Only show top 20 if not authenticated)
    display_df = filtered_df if is_authenticated else filtered_df.head(20)

    # 5. Main Display
    st.title("Strategic Intelligence Stream")
    
    if not is_authenticated:
        st.markdown('<div class="lock-banner">🔒 Preview Mode: Showing 20 most recent assets. Enter code in sidebar for full database.</div>', unsafe_allow_html=True)
    
    st.caption(f"Displaying {len(display_df)} records")

    for _, row in display_df.iterrows():
        with st.container():
            # Modal tags formatting
            tags_html = ' '.join([f'<span class="tag">{tag}</span>' for tag in row['ModalityTags']])
            
            st.markdown(f"""
            <div class="deal-card">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div style="flex: 2;">
                        <span class="score-badge">PT {row['Score']}</span>
                        <p style="color: #94a3b8; font-size: 0.75rem; font-weight: bold; margin-top: 0.8rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">
                            {row['Date']} • {row['Category']}
                        </p>
                        <h2 class="insight-text">{row['Insight']}</h2>
                        <div class="deal-title">{row['Title']}</div>
                        <p class="summary-text">{row['Summary']}</p>
                        <div style="margin-top: 1rem;">
                            {tags_html}
                        </div>
                    </div>
                    <div style="flex: 1; margin-left: 2.5rem; border-left: 2px solid #f1f5f9; padding-left: 2.5rem;">
                        <div style="margin-bottom: 1.5rem;">
                            <p style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">Lead Partner</p>
                            <p style="font-weight: 800; color: #0f172a; font-size: 1.1rem; line-height: 1.2;">{row['PartnerA']}</p>
                            <p style="font-weight: 500; color: #64748b; font-size: 0.85rem;">{row['PartnerB']}</p>
                        </div>
                        <div>
                            <p style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.2rem;">Value</p>
                            <p style="font-size: 1.6rem; font-weight: 900; color: #059669; letter-spacing: -0.02em;">{row['DealValue']}</p>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Failed to initialize database: {e}")