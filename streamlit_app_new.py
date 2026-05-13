import streamlit as st
import pandas as pd
import numpy as np
import html
import re
import os
from datetime import datetime
from google import genai 

# --- 1. AI INITIALIZATION ---
try:
    # This pulls the NEW key you just saved in the Streamlit Dashboard
    GENAI_KEY = st.secrets["GEMINI_API_KEY"]
except:
    # This keeps your code safe if you run it locally
    GENAI_KEY = None 

if GENAI_KEY:
    ai_client = genai.Client(api_key=GENAI_KEY)
    AI_MODEL = "gemini-3.1-flash-lite-preview"
else:
    # This will only show up if you forget to add the key to Streamlit Secrets
    st.error("AI Configuration Error: Gemini API Key not found.")

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="SerGene Strategic Intelligence",
    page_icon="🧬",
    layout="wide"
)

# --- BI HIERARCHY DEFINITION ---
MODALITY_GROUPS = {
    "Cell Therapy": ["CAR-T", "TCR", "TILs", "NK Cells", "Tregs", "MSCs", "iPSCs", "gamma delta T cells", "γδ T cells", "Cell Therapy"],
    "Gene Therapy/Editing": ["CRISPR", "Base Editing", "Prime Editing", "Gene Editing", "Gene Therapy"],
    "RNA Therapeutics": ["mRNA", "siRNA", "RNAi", "miRNA", "ASO", "Antisense", "Aptamer", "RNA"],
    "Biologics": ["Antibody", "Bispecific", "ADC", "Multi-specific", "Peptide", "Biologics"],
    "Small Molecule": ["Small Molecule", "Protein Degrader", "Oral"]
}

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================
def parse_currency(val_str):
    """Parses strings like '$1.5B' or '$50M' into float Millions (USD)."""
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
    """Capitalizes lowercase names but preserves acronyms (BMS, ADC)."""
    if not name or pd.isna(name) or str(name).lower() == 'nan': return "N/A"
    text = str(name).strip()
    words = text.split()
    formatted_words = [word.capitalize() if word.islower() else word for word in words]
    return " ".join(formatted_words)

# ==========================================
# 3. ADVANCED UI STYLING (Restored Full CSS)
# ==========================================
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stApp { background-color: #f8fafc; }
    
    .deal-card {
        background-color: white; padding: 2.5rem; border-radius: 1.5rem;
        border: 1px solid #e2e8f0; margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
    .blurred-card {
        filter: blur(8px); opacity: 0.5; pointer-events: none; user-select: none;
    }
    .date-badge {
        color: #64748b; font-size: 0.75rem; font-weight: 800;
        text-transform: uppercase; letter-spacing: 0.075em; margin-bottom: 0.75rem;
    }
    .parent-tag {
        background-color: #eff6ff; color: #1e40af; padding: 0.35rem 0.85rem;
        border-radius: 0.75rem; font-size: 0.75rem; font-weight: 800;
        text-transform: uppercase; border: 1px solid #bfdbfe;
        display: inline-block; margin-bottom: 1rem;
    }
    .source-link { color: #2563eb; text-decoration: none; font-weight: 800; font-size: 1.5rem; }
    .source-link:hover { text-decoration: underline; color: #1d4ed8; }
    
    .summary-text { color: #475569; font-size: 0.95rem; line-height: 1.6; margin: 1.25rem 0; }
    
    .tag {
        display: inline-block; background-color: #f1f5f9; color: #475569;
        padding: 0.3rem 0.75rem; border-radius: 0.6rem; font-size: 0.7rem;
        font-weight: 700; margin-right: 0.5rem; margin-bottom: 0.5rem;
        border: 1px solid #e2e8f0; text-transform: uppercase;
    }
    .ratio-bar-container {
        height: 12px; background-color: #f1f5f9; border-radius: 6px;
        margin-top: 10px; overflow: hidden; border: 1px solid #e2e8f0;
    }
    .cta-banner {
        background-color: #fef2f2; border: 2px dashed #ef4444; 
        padding: 2.5rem; border-radius: 1.5rem; text-align: center; 
        margin-top: 2rem; margin-bottom: 5rem;
    }
    .ai-strategy-box {
        background-color: #f0f9ff; border-left: 6px solid #0ea5e9;
        padding: 1.75rem; border-radius: 0.75rem; margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 4. DATA REFINERY
# ==========================================
@st.cache_data
def load_and_refine_data():
    if not os.path.exists("sg_intel_assets.arrow"):
        return pd.DataFrame()
    df = pd.read_feather("sg_intel_assets.arrow") 
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(by='Date', ascending=False)
    
    refined_rows = []
    for _, row in df.iterrows():
        raw_tags = row.get('ModalityTags')
        tags = []
        if isinstance(raw_tags, (list, np.ndarray)):
            tags = [re.sub(r'([a-z])([A-Z])', r'\1 \2', str(t)).strip() for t in raw_tags]
        
        for col_name in row.index:
            val = row[col_name]
            col_l = str(col_name).lower().strip()
            is_hit = False
            try:
                if float(val) > 0: is_hit = True
            except:
                if str(val).lower().strip() in ['yes', 'y', 'true', '1']: is_hit = True
            
            if is_hit:
                if "msc" in col_l: tags.append("MSCs")
                elif "ipsc" in col_l: tags.append("iPSCs")
                elif any(x in col_l for x in ["gamma", "delta", "γ", "δ"]): tags.append("γδ T cells")

        tags = list(set([t for t in tags if t and str(t).lower() != 'nan']))
        parent = "Other"
        norm_tags = [t.lower() for t in tags]
        for group_name, keywords in MODALITY_GROUPS.items():
            lower_kws = [k.lower() for k in keywords]
            if any(t in lower_kws for t in norm_tags):
                parent = group_name
                break
        
        val_m = parse_currency(row.get('DealValue', ''))
        up_m = parse_currency(row.get('Upfront', ''))
        ratio = (up_m / val_m) if val_m > 0 else 0.0

        refined_rows.append({
            'ID': row.get('ID', 'N/A'),
            'Date': row.get('Date'),
            'DisplayDate': row.get('Date').strftime('%b %d, %Y') if pd.notnull(row.get('Date')) else "N/A",
            'ParentModality': parent,
            'SubModalities': tags,
            'TA': str(row.get('TA', 'Other/General')).strip(),
            'Stage': str(row.get('Stage', 'Pre-clinical')).strip(),
            'Category': row.get('Category', 'N/A'),
            'TotalValueM': val_m,
            'UpfrontRatio': ratio,
            'DisplayValue': str(row.get('DealValue', 'N/A')),
            'PartnerA': smart_format_company(row.get('PartnerA')),
            'PartnerB': smart_format_company(row.get('PartnerB')),
            'Insight': str(row.get('Insight', '')),
            'Title': str(row.get('Title', '')),
            'Summary': str(row.get('Summary', '')),
            'Link': str(row.get('Link', '#'))
        })
    return pd.DataFrame(refined_rows)

# ==========================================
# 5. UI & AUTHENTICATION
# ==========================================
try:
    # 1. Load the data
    df_master = load_and_refine_data()
    
    # 2. Advanced Date Parsing (The "May 13th" Fix)
    df_master['Date'] = pd.to_datetime(df_master['Date'], dayfirst=True, errors='coerce')
    df_master = df_master.dropna(subset=['Date']).sort_values('Date', ascending=False)

    # --- SIDEBAR UI ---
    st.sidebar.title("🧬 SerGene Intel")
    st.sidebar.markdown("---")

    # A. DATE FILTER (Priority #1 for Iframe Visibility)
    st.sidebar.subheader("📅 Select Timeframe")
    
    # We use .date() to ensure we are comparing dates to dates, not times
    min_date = df_master['Date'].min().date()
    max_date = df_master['Date'].max().date()
    
    date_sel = st.sidebar.date_input(
        "Date Range", 
        value=(min_date, max_date),
        min_value=min_date,
        max_value=datetime.now().date()
    )

    st.sidebar.divider()

    # B. CLIENT ACCESS
    with st.sidebar.expander("🔑 Client Access", expanded=False):
        try:
            MASTER_PASSWORD = st.secrets["access_password"]
        except:
            MASTER_PASSWORD = "SerGenePilot2026"
            
        password_input = st.text_input("Enter Access Code", type="password")
        is_authenticated = (password_input == MASTER_PASSWORD)
        
        if is_authenticated:
            st.success("Full Access Granted")
        elif password_input != "":
            st.error("Invalid Code")

    st.sidebar.divider()

    # C. ATTRIBUTE FILTERS
    sel_tas = st.sidebar.multiselect("Therapeutic Area", sorted(df_master['TA'].unique().tolist()))
    sel_stages = st.sidebar.multiselect("Development Stage", sorted(df_master['Stage'].unique().tolist()))
    sel_parents = st.sidebar.multiselect("Broad Modality", sorted(df_master['ParentModality'].unique().tolist()))
    
    # Re-adding the Sub-Modality filter from your original code
    all_subs = sorted(list(set([t for sub in df_master['SubModalities'] for t in sub])))
    sel_subs = st.sidebar.multiselect("Specific Platforms / Cell Types", all_subs)
    
    search_term = st.sidebar.text_input("🔍 Search Companies or Insights")

    # --- THE FILTERING ENGINE ---
    stats_df = df_master.copy()

    # 1. Filter by Date (Handling the tuple vs single-click)
    if isinstance(date_sel, (list, tuple)) and len(date_sel) == 2:
        stats_df = stats_df[
            (stats_df['Date'].dt.date >= date_sel[0]) & 
            (stats_df['Date'].dt.date <= date_sel[1])
        ]

    # 2. Filter by Categories
    if sel_tas:
        stats_df = stats_df[stats_df['TA'].isin(sel_tas)]
    if sel_stages:
        stats_df = stats_df[stats_df['Stage'].isin(sel_stages)]
    if sel_parents:
        stats_df = stats_df[stats_df['ParentModality'].isin(sel_parents)]
    
    # 3. Filter by Sub-Modalities (Platform/Cell Types)
    if sel_subs:
        stats_df = stats_df[stats_df['SubModalities'].apply(lambda x: any(s in x for s in sel_subs))]

    # 4. Filter by Search Keywords
    if search_term:
        stats_df = stats_df[
            stats_df['PartnerA'].str.contains(search_term, case=False, na=False) |
            stats_df['PartnerB'].str.contains(search_term, case=False, na=False) |
            stats_df['Insight'].str.contains(search_term, case=False, na=False)
        ]

    # --- THE "MOAT" LOGIC ---
    GLOBAL_PREVIEW_LIMIT = 5
    BLUR_LIMIT = 3
    
    if is_authenticated:
        visible_df = stats_df 
    else:
        visible_df = stats_df.head(GLOBAL_PREVIEW_LIMIT)

    # --- DEBUG COUNTER (Helpful for you to see in the sidebar) ---
    st.sidebar.write(f"📊 Showing {len(stats_df)} matching deals")

except Exception as e:
    st.sidebar.error(f"Error: {e}")
    # Fallback to empty dataframes so the rest of the app doesn't crash
    stats_df = pd.DataFrame()
    visible_df = pd.DataFrame()
    is_authenticated = False

    # ==========================================
    # 6. DASHBOARD & ANALYTICS (The Hidden Gem)
    # ==========================================
    st.title("Strategic Deal Intelligence Stream")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Database Depth", len(stats_df))
    m2.metric("Market Volume Analysed", f"${stats_df['TotalValueM'].sum()/1000:.1f}B")
    valid_r = stats_df[stats_df['UpfrontRatio'] > 0]['UpfrontRatio']
    avg_r = valid_r.mean() if not valid_r.empty else 0
    m3.metric("Avg. Upfront Ratio", f"{avg_r:.1%}")

    st.divider()
    with st.expander("📈 Market Trends & Competitive Landscape", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**Modality Mix**")
            st.bar_chart(stats_df['ParentModality'].value_counts(), color="#3b82f6")
        with c2:
            st.write("**Therapeutic Focus**")
            st.bar_chart(stats_df['TA'].value_counts(), color="#10b981")
        with c3:
            st.write("**Development Stage**")
            st.bar_chart(stats_df['Stage'].value_counts(), color="#6366f1")
            

    # --- THE AI STRATEGIC BRIEF BUTTON (Paste this here) ---
    st.write("") # Adds a tiny bit of spacing
    if st.button("🪄 Generate AI Strategic Brief"):
        if is_authenticated:
            # Paying users get the real deal
            with st.status("🤖 SerGene AI is analyzing current deal flow...", expanded=True):
                # Prepare a summary of the deals currently visible in the filters
                deal_list = ""
                # We take the top 20 deals based on your current filters
                for _, r in stats_df.head(20).iterrows():
                    deal_list += f"- {r['PartnerA']} & {r['PartnerB']}: {r['Insight']}\n"

                prompt = f"""
                You are a Senior Biotech Strategic Analyst. Analyze these recent deals:
                {deal_list}
                
                Provide a professional 3-point summary:
                1. What is the biggest trend in this specific segment?
                2. What does this suggest about the current market risk appetite?
                3. A 1-sentence 'Strategic Outlook' for an investor.
                
                Keep the tone executive, objective, and data-driven.
                """
                
                try:
                    response = ai_client.models.generate_content(model=AI_MODEL, contents=prompt)
                    st.markdown(f"""
                        <div class="ai-strategy-box">
                            <h3 style="margin-top:0;">🤖 Strategic Market Brief</h3>
                            <p style="white-space: pre-wrap;">{response.text}</p>
                        </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"AI Analysis currently unavailable: {str(e)}")
        else:
            # Guests see this instead
            st.warning("🔒 The AI Strategic Brief is a Premium Feature.")
            st.info("Please enter your Client Access Code in the sidebar to unlock real-time strategic analysis.")

    # ==========================================
    # 7. DEAL CARDS ENGINE (Restored Full HTML)
    # ==========================================
    if not is_authenticated and visible_df.empty and not stats_df.empty:
        st.warning(f"⚠️ {len(stats_df)} deals match these criteria in the full database, but they are outside the Top {GLOBAL_PREVIEW_LIMIT} preview window.")

    CARD_HTML = """
    <div class="deal-card {extra_class}">
        <div style="display: flex; justify-content: space-between; align-items: start; gap: 2.5rem;">
            <div style="flex: 2;">
                <div class="date-badge">{d_date} | {ta} • {stage}</div>
                <span class="parent-tag">{p_mod}</span>
                <h2 style="margin-top: 1rem;">
                    <a href="{link}" target="_blank" class="source-link">{insight}</a>
                </h2>
                <div style="font-weight: 700; color: #0f172a; font-size: 1.1rem; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.5rem;">{title}</div>
                <p class="summary-text">{summary}</p>
                <div style="margin-top: 1.5rem;">{tags}</div>
            </div>
            <div style="flex: 1; border-left: 2px solid #f1f5f9; padding-left: 2.5rem; min-width: 280px;">
                <div style="margin-bottom: 2rem;">
                    <p style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase;">Total Deal Value</p>
                    <p style="font-size: 1.85rem; font-weight: 900; color: #059669; margin: 0;">{value}</p>
                </div>
                <div style="margin-bottom: 2rem;">
                    <p style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase;">Upfront Ratio ({r_pct}%)</p>
                    <div class="ratio-bar-container">
                        <div style="height:100%; width:{r_pct}%; background-color:{r_color}; border-radius:6px;"></div>
                    </div>
                </div>
                <div style="margin-bottom: 1rem;">
                    <p style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase;">Partners</p>
                    <p style="font-weight: 800; color: #0f172a; font-size: 1.15rem; margin: 0;">{pA}</p>
                    <p style="color: #64748b; font-size: 0.85rem; margin-top: 0.25rem;">{pB}</p>
                </div>
            </div>
        </div>
    </div>
    """

    # 7.1 RENDER VISIBLE CARDS
    for _, row in visible_df.iterrows():
        rpct = round(row['UpfrontRatio'] * 100, 1)
        rcol = "#10b981" if rpct > 25 else "#f59e0b"
        tags_h = "".join([f'<span class="tag">{html.escape(str(t))}</span>' for t in row['SubModalities']])
        st.markdown(CARD_HTML.format(
            extra_class="", d_date=row['DisplayDate'], ta=row['TA'], stage=row['Stage'],
            p_mod=row['ParentModality'], link=row['Link'], insight=html.escape(row['Insight']),
            title=html.escape(row['Title']), summary=html.escape(row['Summary']), tags=tags_h,
            value=html.escape(row['DisplayValue']), r_pct=rpct, r_color=rcol, 
            pA=html.escape(row['PartnerA']), pB=html.escape(row['PartnerB'])
        ), unsafe_allow_html=True)

    # 7.2 RENDER BLURRED TEASERS
    if not is_authenticated:
        for _, row in df_master.iloc[GLOBAL_PREVIEW_LIMIT : GLOBAL_PREVIEW_LIMIT + BLUR_LIMIT].iterrows():
            st.markdown(CARD_HTML.format(
                extra_class="blurred-card", d_date=row['DisplayDate'], ta=row['TA'], stage=row['Stage'],
                p_mod=row['ParentModality'], link="#", insight="[LOCKED INSIGHT]",
                title=html.escape(row['Title']), summary=html.escape(row['Summary']), tags="",
                value="$$$,$$$", r_pct=50, r_color="#cbd5e1", pA="[LOCKED]", pB="[LOCKED]"
            ), unsafe_allow_html=True)

        # 7.3 CTA BANNER
        mailto_link = "mailto:spiros@sergenebio.co.uk?subject=Portal Access Inquiry"
        st.markdown(f"""
            <div class="cta-banner">
                <h2 style="color: #991b1b; margin-top: 0;">🔒 Unlock Strategic Access</h2>
                <p style="font-size: 1.1rem; color: #b91c1c; margin-bottom: 1.5rem;">
                    Analyze the full historical database and generate custom AI Strategic Briefs.
                </p>
                <a href="{mailto_link}" 
                   style="text-decoration: none; color: white; background-color: #ef4444; 
                   padding: 1rem 2rem; border-radius: 0.75rem; font-weight: 800; font-size: 1.1rem; display: inline-block;">
                   Request Access Code
                </a>
                <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px dashed #fca5a5;">
                    <p style="font-size: 0.9rem; color: #7f1d1d; margin: 0;">Fallback Email: <b>spiros@sergenebio.co.uk</b></p>
                </div>
            </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"BI Module Error: {e}")
