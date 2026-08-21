import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime
import streamlit as st

# =====================================================================
# 1. PAGE SETUP & HIGH-CONTRAST THEMING
# =====================================================================
st.set_page_config(
    page_title="QUANT BETTING SUITE", 
    layout="wide", 
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast CSS Styling
st.markdown("""
    <style>
    /* Global Page Background & Font Hierarchy */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* High Contrast Headers */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 800 !important;
        color: #00FFCC !important;
    }
    
    /* Custom Value Badges */
    .value-box-green {
        background-color: #003B2B;
        border: 2px solid #00FF88;
        color: #00FF88;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
    }
    .value-box-red {
        background-color: #3B0000;
        border: 2px solid #FF3366;
        color: #FF3366;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
    }
    
    /* Input Container Styling */
    .stNumberInput, .stTextInput, .stSelectbox {
        background-color: #1A1D24;
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)

DB_PATH = "quant_live_ledger.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS live_bets (
                ticket_id TEXT PRIMARY KEY,
                placed_at TEXT,
                match_date TEXT,
                competition TEXT,
                match_name TEXT,
                selection TEXT,
                executed_odds REAL,
                cash_staked REAL,
                model_prob REAL,
                status TEXT DEFAULT 'OPEN',
                net_pnl REAL DEFAULT 0.0
            )
        """)
        conn.commit()

init_db()

# =====================================================================
# 2. DIXON-COLES POISSON ENGINE
# =====================================================================
def dixon_coles_tau(x, y, lambda_h, mu_a, rho=-0.06):
    if x == 0 and y == 0: return 1.0 - (lambda_h * mu_a * rho)
    elif x == 1 and y == 0: return 1.0 + (mu_a * rho)
    elif x == 0 and y == 1: return 1.0 + (lambda_h * rho)
    elif x == 1 and y == 1: return 1.0 - rho
    return 1.0

def calculate_match_probabilities(lambda_h, mu_a, max_goals=8):
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p_h = poisson.pmf(h, lambda_h)
            p_a = poisson.pmf(a, mu_a)
            tau = dixon_coles_tau(h, a, lambda_h, mu_a)
            matrix[h, a] = p_h * p_a * tau
    matrix /= np.sum(matrix)
    
    prob_home = float(np.sum(np.tril(matrix, -1)))
    prob_draw = float(np.sum(np.diag(matrix)))
    prob_away = float(np.sum(np.triu(matrix, 1)))
    
    # Over / Under 2.5 Goals Matrix Sum
    prob_under_25 = 0.0
    for h in range(3):
        for a in range(3 - h):
            prob_under_25 += matrix[h, a]
    prob_over_25 = 1.0 - prob_under_25
    
    return {
        "Home": prob_home, 
        "Draw": prob_draw, 
        "Away": prob_away,
        "Over 2.5": prob_over_25,
        "Under 2.5": prob_under_25
    }

# =====================================================================
# 3. APPLICATION HEADER
# =====================================================================
st.title("⚡ QUANTITATIVE FOOTBALL ANALYTICS")
st.markdown("<p style='color: #00FFCC; font-size: 16px; font-weight: 600;'>PROBABILISTIC MATCH EVALUATION & BANKROLL EXECUTION</p>", unsafe_allow_html=True)
st.write("---")

tab1, tab2, tab3 = st.tabs([
    "🎯 VALUE SCANNER", 
    "📝 EXECUTE BET", 
    "📈 CAPITAL LEDGER"
])

# ---------------------------------------------------------------------
# TAB 1: VALUE SCANNER
# ---------------------------------------------------------------------
with tab1:
    st.markdown("### 1. Match Inputs & Goal Expectations")
    
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        match_title = st.text_input("Fixture Name", value="Arsenal vs Chelsea")
    with c2:
        lambda_h = st.number_input("Home Expected Goals (λ)", value=1.85, step=0.05, help="Average goals expected for Home team")
    with c3:
        mu_a = st.number_input("Away Expected Goals (μ)", value=0.95, step=0.05, help="Average goals expected for Away team")
        
    st.markdown("### 2. Live Bookmaker Odds")
    o1, o2, o3, o4, o5 = st.columns(5)
    with o1: odds_h = st.number_input("Home Win Odds", value=1.85, step=0.01)
    with o2: odds_d = st.number_input("Draw Odds", value=3.80, step=0.01)
    with o3: odds_a = st.number_input("Away Win Odds", value=4.50, step=0.01)
    with o4: odds_o = st.number_input("Over 2.5 Odds", value=1.95, step=0.01)
    with o5: odds_u = st.number_input("Under 2.5 Odds", value=1.90, step=0.01)
    
    probs = calculate_match_probabilities(lambda_h, mu_a)
    
    markets = [
        ("Home Win", "Home", odds_h),
        ("Draw", "Draw", odds_d),
        ("Away Win", "Away", odds_a),
        ("Over 2.5 Goals", "Over 2.5", odds_o),
        ("Under 2.5 Goals", "Under 2.5", odds_u)
    ]
    
    st.markdown("---")
    st.markdown(f"### 📊 Valuation Breakdown for **{match_title}**")
    
    res_cols = st.columns(len(markets))
    
    for idx, (label, key, b_odds) in enumerate(markets):
        m_prob = probs[key]
        ev = (m_prob * b_odds) - 1.0
        
        # Quarter Kelly calculation
        b = b_odds - 1.0
        q = 1.0 - m_prob
        full_k = (m_prob * b - q) / b if b > 0 else 0
        q_kelly = max(0.0, full_k * 0.25) * 100
        
        with res_cols[idx]:
            st.markdown(f"**{label}**")
            st.markdown(f"Bookie: `{b_odds:.2f}`")
            st.markdown(f"Model: `{m_prob*100:.1f}%`")
            
            if ev > 0:
                st.markdown(f"""
                    <div class="value-box-green">
                        EDGE: +{ev*100:.1f}%<br>
                        STAKE: {q_kelly:.1f}%
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="value-box-red">
                        EDGE: {ev*100:.1f}%<br>
                        NO VALUE
                    </div>
                """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# TAB 2: EXECUTE BET
# ---------------------------------------------------------------------
with tab2:
    st.markdown("### Record Ticket Placed with Bookmaker")
    
    with st.form("bet_entry_form"):
        col_x, col_y = st.columns(2)
        with col_x:
            match_name = st.text_input("Match Name", value=match_title)
            competition = st.selectbox("League / Competition", ["Premier League", "La Liga", "Serie A", "Bundesliga", "UEFA Champions League", "Other"])
            selection = st.selectbox("Selection Placed", ["Home", "Draw", "Away", "Over 2.5 Goals", "Under 2.5 Goals"])
            executed_odds = st.number_input("Odds Secured", value=1.85, step=0.01)
        
        with col_y:
            match_date = st.date_input("Date Placed", datetime.now())
            cash_staked = st.number_input("Cash Staked ($ / KES)", value=100.0, step=10.0)
            model_prob = st.number_input("Model Prob (%)", value=62.0, step=1.0)
            
        submit_btn = st.form_submit_button("💾 Save Ticket to Master Ledger")
        
        if submit_btn:
            ticket_id = f"TICK_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO live_bets 
                    (ticket_id, placed_at, match_date, competition, match_name, selection, executed_odds, cash_staked, model_prob)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (ticket_id, datetime.now().isoformat(), str(match_date), competition, match_name, selection, executed_odds, cash_staked, model_prob))
                conn.commit()
            st.success(f"✅ Ticket {ticket_id} successfully logged!")

# ---------------------------------------------------------------------
# TAB 3: CAPITAL LEDGER
# ---------------------------------------------------------------------
with tab3:
    st.markdown("### Portfolio Performance & Ticket Settle")
    
    with sqlite3.connect(DB_PATH) as conn:
        df_bets = pd.read_sql_query("SELECT * FROM live_bets", conn)
        
    if df_bets.empty:
        st.info("No bets recorded in ledger yet.")
    else:
        open_bets = df_bets[df_bets['status'] == 'OPEN']
        settled_bets = df_bets[df_bets['status'] != 'OPEN']
        
        # Performance Top Metrics
        total_staked = settled_bets['cash_staked'].sum() if not settled_bets.empty else 0
        total_pnl = settled_bets['net_pnl'].sum() if not settled_bets.empty else 0
        roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric("TOTAL STAKED", f"${total_staked:,.2f}")
        m2.metric("NET PnL", f"${total_pnl:+,.2f}")
        m3.metric("ACTUAL YIELD (ROI)", f"{roi:+.2f}%")
        
        st.markdown("---")
        st.markdown("### Settle Pending Tickets")
        
        if not open_bets.empty:
            for idx, row in open_bets.iterrows():
                b1, b2, b3 = st.columns([3, 2, 1])
                with b1:
                    st.markdown(f"**{row['match_name']}** — `{row['selection']}` @ `{row['executed_odds']:.2f}` (Stake: `${row['cash_staked']:,.2f}`)")
                with b2:
                    outcome = st.selectbox("Outcome", ["WIN", "LOSS", "VOID"], key=f"sel_{row['ticket_id']}")
                with b3:
                    if st.button("Settle", key=f"btn_{row['ticket_id']}"):
                        pnl = (row['cash_staked'] * (row['executed_odds'] - 1)) if outcome == "WIN" else (-row['cash_staked'] if outcome == "LOSS" else 0.0)
                        with sqlite3.connect(DB_PATH) as conn:
                            cursor = conn.cursor()
                            cursor.execute("UPDATE live_bets SET status = ?, net_pnl = ? WHERE ticket_id = ?", (outcome, pnl, row['ticket_id']))
                            conn.commit()
                        st.rerun()
        else:
            st.write("No open tickets pending settlement.")
            
        if not settled_bets.empty:
            st.markdown("---")
            st.markdown("### Cumulative PnL Curve")
            settled_bets['Cumulative_PnL'] = settled_bets['net_pnl'].cumsum()
            st.line_chart(settled_bets['Cumulative_PnL'])
