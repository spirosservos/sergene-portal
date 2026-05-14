import streamlit as st
import pandas as pd
import numpy as np
import html
import re
import os
from datetime import datetime
from google import genai 

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
# 4. DATA REFINERY
# ==========================================
@st.cache_data
def load_and_refine_data():
    if not os.path.exists("sg_intel_assets.arrow"):
        return pd.DataFrame()
    df = pd.read_feather("sg_intel_assets.arrow") 
    
    if 'Date' in df.columns:
        # Using dayfirst=True to handle UK/International Excel dates correctly
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.sort_values(by='Date', ascending=False)
    
    refined_rows = []
    for _, row in df.iterrows():
        raw_tags = row.get('ModalityTags', [])
        tags = [re.sub(r'([a-z])([A-Z])', r'\1 \2', str(t)).strip() for t in raw_tags] if isinstance(raw_tags, (list, np.ndarray)) else []
        
        # Specific Cell Type Logic
        for col_name in row.index:
            val = row[col_name]
            col_l = str(col_name).lower()
            is_hit = False
            try:
                if float(val) > 0: is_hit = True
            except:
                if str(val).lower() in ['yes', 'y', 'true', '1']: is_hit = True
            
            if is_hit:
                if "msc" in col_l: tags.append("MSCs")
                elif "ipsc" in col_l: tags.append("iPSCs")
                elif any(x in col_l for x in ["gamma", "delta", "γ", "δ"]): tags.append("γδ T cells")

        tags = list(set([t for t in tags if t and str(t).lower() != 'nan']))
        
        # Parent Modality Assignment
        parent = "Other"
        norm_tags = [t.lower() for t in tags]
        for group_name, keywords in MODALITY_GROUPS.items():
            if any(k.lower() in norm_tags for k in keywords):
                parent = group_name
                break
        
        val_m = parse_currency(row.get('DealValue', ''))
        up_m = parse_currency(row.get('Upfront', ''))
        ratio = (up_m / val_m) if val_m > 0 else 0.0

        refined_rows.append({
            'Date': row.get('Date'),
            'Date_Obj': row.get('Date').date() if pd.notnull(row.get('Date')) else None,
            'DisplayDate': row.get('Date').strftime('%b %d, %Y') if pd.notnull(row.get('Date')) else "N/A",
            'ParentModality': parent,
            'SubModalities': tags,
            'TA': str(row.get('TA', 'Other/General')).strip(),
            'Stage': str(row.get('Stage', 'Pre-clinical')).strip(),
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
# 5. UI, AUTHENTICATION & FILTERING
# ==========================================
try:
    df_master = load_and_refine_data()
    
    # 1. CRITICAL: Force Date_Obj to be actual date objects and remove any "float" nulls
    df_master['Date_Obj'] = pd.to_datetime(df_master['Date'], dayfirst=True, errors='coerce').dt.date
    df_master = df_master[df_master['Date_Obj'].notnull()]
    df_master = df_master.sort_values(by='Date_Obj', ascending=False)

    st.sidebar.title("🧬 SerGene Intelligence")
    
    # A. DATE FILTER (Top position for Iframe visibility)
    st.sidebar.subheader("📅 Select Timeframe")
    min_db_date = df_master['Date_Obj'].min()
    max_db_date = df_master['Date_Obj'].max()
    
    date_sel = st.sidebar.date_input(
        "Date Range", 
        value=(min_db_date, max_db_date),
        min_value=min_db_date,
        max_value=datetime.now().date()
    )

    st.sidebar.divider()

    # B. CLIENT ACCESS
    is_authenticated = False
    with st.sidebar.expander("🔑 Client Access", expanded=False):
        secret_pass = st.secrets.get("access_password")
        password_input = st.text_input("Enter Access Code", type="password")
        if password_input:
            if secret_pass and password_input == secret_pass:
                is_authenticated = True
                st.success("Full Access Granted")
            else:
                st.error("Invalid Code")
        if not is_authenticated:
            st.markdown("---")
            st.caption("Request Access:")
            st.code("spiros@sergenebio.co.uk")

    st.sidebar.divider()
    
    # C. ATTRIBUTE FILTERS
    sel_tas = st.sidebar.multiselect("Therapeutic Area", sorted(df_master['TA'].unique().tolist()))
    sel_stages = st.sidebar.multiselect("Development Stage", sorted(df_master['Stage'].unique().tolist()))
    sel_parents = st.sidebar.multiselect("Broad Modality", sorted(df_master['ParentModality'].unique().tolist()))
    
    all_subs = sorted(list(set([t for sub in df_master['SubModalities'] for t in sub])))
    sel_subs = st.sidebar.multiselect("Specific Platforms / Cell Types", all_subs)
    
    search_term = st.sidebar.text_input("🔍 Search Database")

    # --- 5.5 FILTERING ENGINE (The "Bulletproof" Fix) ---
    stats_df = df_master.copy()

    # We ensure date_sel is a valid tuple and stats_df has no invalid dates
    if isinstance(date_sel, (list, tuple)) and len(date_sel) == 2:
        start_date, end_date = date_sel
        # We force the comparison to only happen on rows that aren't null
        stats_df = stats_df[
            (stats_df['Date_Obj'] >= start_date) & 
            (stats_df['Date_Obj'] <= end_date)
        ]
    
    if sel_tas: stats_df = stats_df[stats_df['TA'].isin(sel_tas)]
    if sel_stages: stats_df = stats_df[stats_df['Stage'].isin(sel_stages)]
    if sel_parents: stats_df = stats_df[stats_df['ParentModality'].isin(sel_parents)]
    if sel_subs:
        stats_df = stats_df[stats_df['SubModalities'].apply(lambda x: any(s in x for s in sel_subs))]
    if search_term:
        stats_df = stats_df[
            stats_df['Insight'].str.contains(search_term, case=False, na=False) | 
            stats_df['Title'].str.contains(search_term, case=False, na=False)
        ]

    # Handle the "Moat" (Preview limit)
    GLOBAL_PREVIEW_LIMIT = 5
    BLUR_LIMIT = 3
    if is_authenticated:
        visible_df = stats_df
    else:
        visible_df = stats_df.head(GLOBAL_PREVIEW_LIMIT)

except Exception as e:
    st.error(f"BI Module Error: {e}")
    # Fallback to prevent the rest of the app from crashing
    stats_df = pd.DataFrame()
    visible_df = pd.DataFrame()
    is_authenticated = False

    # ==========================================
    # 6. DASHBOARD
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
            st.markdown("### **Modality Mix**")
            st.bar_chart(stats_df['ParentModality'].value_counts(), color="#3b82f6")
        with c2:
            st.markdown("### **Therapeutic Focus**")
            st.bar_chart(stats_df['TA'].value_counts(), color="#10b981")
        with c3:
            st.markdown("### **Development Stage**")
            st.bar_chart(stats_df['Stage'].value_counts(), color="#6366f1")

    # AI Strategic Brief Section
    if st.button("🪄 Generate AI Strategic Brief"):
        if is_authenticated:
            with st.status("🤖 SerGene AI is analyzing current deal flow...", expanded=True):
                deal_list = ""
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
                except Exception as ai_e:
                    st.error(f"AI Analysis currently unavailable: {str(ai_e)}")
        else:
            st.warning("🔒 Client Access Required to unlock AI Strategic Analysis.")

    # ==========================================
    # 7. DEAL CARDS (With Upfront Ratio Fix)
    # ==========================================
    CARD_HTML = """
    <div class="deal-card {extra_class}">
        <div style="display: flex; justify-content: space-between; align-items: start; gap: 2rem;">
            <div style="flex: 2;">
                <div class="date-badge">{d_date} | {ta} • {stage}</div>
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

    for _, row in visible_df.iterrows():
        tags_h = "".join([f'<span class="tag">{html.escape(t)}</span>' for t in row['SubModalities']])
        # Calculating Upfront % as an integer for the card
        r_val = int(round(row['UpfrontRatio'] * 100))
        
        st.markdown(CARD_HTML.format(
            extra_class="", d_date=row['DisplayDate'], ta=row['TA'], stage=row['Stage'],
            p_mod=row['ParentModality'], link=row['Link'], insight=html.escape(row['Insight']),
            title=html.escape(row['Title']), summary=html.escape(row['Summary']), tags=tags_h,
            value=row['DisplayValue'], r_pct=r_val, 
            pA=row['PartnerA'], pB=row['PartnerB']
        ), unsafe_allow_html=True)

    if not is_authenticated:
        # Show Blurred cards for guests
        for _, row in stats_df.iloc[GLOBAL_PREVIEW_LIMIT : GLOBAL_PREVIEW_LIMIT + BLUR_LIMIT].iterrows():
            st.markdown(CARD_HTML.format(
                extra_class="blurred-card", d_date=row['DisplayDate'], ta=row['TA'], stage=row['Stage'],
                p_mod=row['ParentModality'], link="#", insight="LOCKED", title="LOCKED",
                summary="Unlock full access to view details.", tags="",
                value="$$$", r_pct=0, pA="LOCKED", pB="LOCKED"
            ), unsafe_allow_html=True)
            
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
            </div>
        """, unsafe_allow_html=True)
            
except Exception as e:
    st.error(f"BI Module Error: {e}")
