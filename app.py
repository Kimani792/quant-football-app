import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime, date
import streamlit as st

# =====================================================================
# 1. PAGE SETUP & STYLING
# =====================================================================
st.set_page_config(
    page_title="QUANT FOOTBALL SUITE", 
    layout="wide", 
    page_icon="⚽",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0a0f1d 0%, #071828 50%, #0d1b2a 100%);
        color: #FFFFFF;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0b1320 !important;
        border-right: 2px solid #00E5FF;
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
        color: #00E5FF !important;
        font-weight: 700 !important;
    }
    
    .bankroll-card {
        background: #111a2e;
        border: 1px solid #00E5FF;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .bankroll-title { color: #94A3B8; font-size: 12px; font-weight: 700; text-transform: uppercase; }
    .bankroll-value { color: #00FFCC; font-size: 22px; font-weight: 800; }
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
# 2. DIXON-COLES ENGINE
# =====================================================================
def dixon_coles_tau(x, y, lambda_h, mu_a, rho=-0.06):
    if x == 0 and y == 0: return 1.0 - (lambda_h * mu_a * rho)
    elif x == 1 and y == 0: return 1.0 + (mu_a * rho)
    elif x == 0 and y == 1: return 1.0 + (lambda_h * rho)
    elif x == 1 and y == 1: return 1.0 - rho
    return 1.0

def calculate_match_metrics(lambda_h, mu_a, max_goals=8):
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p_h = poisson.pmf(h, lambda_h)
            p_a = poisson.pmf(a, mu_a)
            tau = dixon_coles_tau(h, a, lambda_h, mu_a)
            matrix[h, a] = p_h * p_a * tau
    matrix /= np.sum(matrix)
    
    p_home = float(np.sum(np.tril(matrix, -1)))
    p_draw = float(np.sum(np.diag(matrix)))
    p_away = float(np.sum(np.triu(matrix, 1)))
    p_under = sum(matrix[h, a] for h in range(3) for a in range(3 - h))
    p_over = 1.0 - p_under
    
    return {"Home": p_home, "Draw": p_draw, "Away": p_away, "Over 2.5": p_over, "Under 2.5": p_under}

# =====================================================================
# 3. SAMPLE MULTI-MATCH FIXTURES DATASET
# =====================================================================
def get_sample_fixtures():
    today_str = date.today().strftime("%Y-%m-%d")
    return [
        {"date": today_str, "league": "Premier League", "home": "Arsenal", "away": "Chelsea", "lambda_h": 1.85, "mu_a": 0.95, "odds": {"Home": 1.85, "Draw": 3.80, "Away": 4.50, "Over 2.5": 1.95, "Under 2.5": 1.90}},
        {"date": today_str, "league": "Premier League", "home": "Liverpool", "away": "Man City", "lambda_h": 1.60, "mu_a": 1.55, "odds": {"Home": 2.40, "Draw": 3.60, "Away": 2.80, "Over 2.5": 1.70, "Under 2.5": 2.15}},
        {"date": today_str, "league": "La Liga", "home": "Real Madrid", "away": "Barcelona", "lambda_h": 1.90, "mu_a": 1.30, "odds": {"Home": 2.05, "Draw": 3.70, "Away": 3.40, "Over 2.5": 1.65, "Under 2.5": 2.25}},
        {"date": today_str, "league": "Serie A", "home": "Inter", "away": "Juventus", "lambda_h": 1.45, "mu_a": 0.85, "odds": {"Home": 1.95, "Draw": 3.30, "Away": 4.20, "Over 2.5": 2.10, "Under 2.5": 1.75}},
    ]

# =====================================================================
# 4. SIDEBAR CONTROLS & BANKROLL
# =====================================================================
with st.sidebar:
    st.title("⚙️ CONTROL PANEL")
    
    st.markdown("### 📅 Date Range Filter")
    start_date = st.date_input("Start Date", value=date.today(), format="DD/MM/YYYY")
    end_date = st.date_input("End Date", value=date.today(), format="DD/MM/YYYY")
    
    st.markdown("---")
    st.markdown("### 🏆 Competition Filter")
    league_options = ["All Leagues", "Premier League", "La Liga", "Serie A", "Bundesliga", "UEFA Champions League", "Other"]
    selected_league = st.selectbox("Select League", league_options)
    
    st.markdown("---")
    st.markdown("### 💰 Live Capital Tracker")
    initial_bankroll = st.number_input("Starting Capital ($ / KES)", value=1000.0, step=100.0)
    
    with sqlite3.connect(DB_PATH) as conn:
        df_ledger = pd.read_sql_query("SELECT * FROM live_bets", conn)
        
    if not df_ledger.empty:
        open_bets = df_ledger[df_ledger['status'] == 'OPEN']
        total_open_stake = open_bets['cash_staked'].sum()
        settled_bets = df_ledger[df_ledger['status'] != 'OPEN']
        total_pnl = settled_bets['net_pnl'].sum()
    else:
        total_open_stake = 0.0
        total_pnl = 0.0
        
    current_available = initial_bankroll + total_pnl - total_open_stake
    
    st.markdown(f"""
        <div class="bankroll-card">
            <div class="bankroll-title">Committed Stakes (Open)</div>
            <div class="bankroll-value" style="color: #FFCC00;">${total_open_stake:,.2f}</div>
        </div>
        <div class="bankroll-card">
            <div class="bankroll-title">Available Capital</div>
            <div class="bankroll-value" style="color: #00FFCC;">${current_available:,.2f}</div>
        </div>
    """, unsafe_allow_html=True)

# =====================================================================
# 5. MAIN INTERACTION DECK
# =====================================================================
st.title("⚡ QUANTITATIVE FOOTBALL SUITE")
st.markdown(f"<p style='color: #00FFCC; font-size: 15px; font-weight: 700;'>ACTIVE FILTER: {start_date.strftime('%d/%m/%Y')} TO {end_date.strftime('%d/%m/%Y')} | LEAGUE: {selected_league.upper()}</p>", unsafe_allow_html=True)
st.write("---")

tab1, tab2, tab3 = st.tabs([
    "🎯 MATCH FIXTURES SCANNER", 
    "📝 EXECUTE TICKET", 
    "📈 CAPITAL LEDGER"
])

# ---------------------------------------------------------------------
# TAB 1: MULTI-MATCH FIXTURE TABLE & VALUE INDICATORS
# ---------------------------------------------------------------------
with tab1:
    st.markdown("### Upcoming Match Predictions & Value Analysis")
    
    fixtures = get_sample_fixtures()
    
    # Filter fixtures by league
    if selected_league != "All Leagues":
        fixtures = [f for f in fixtures if f["league"] == selected_league]
        
    if not fixtures:
        st.info("No games scheduled for the selected league.")
    else:
        for idx, match in enumerate(fixtures):
            match_title = f"⚽ {match['home']} vs {match['away']} ({match['league']})"
            probs = calculate_match_probabilities = calculate_match_metrics(match['lambda_h'], match['mu_a'])
            
            with st.expander(match_title, expanded=True):
                col_info, col_table = st.columns([1, 2])
                
                with col_info:
                    st.markdown(f"**League:** {match['league']}")
                    st.markdown(f"**Home xG (λ):** `{match['lambda_h']}` | **Away xG (μ):** `{match['mu_a']}`")
                
                # Table breakdown for each game
                table_data = []
                for selection in ["Home", "Draw", "Away", "Over 2.5", "Under 2.5"]:
                    m_prob = probs[selection]
                    b_odds = match["odds"][selection]
                    ev = (m_prob * b_odds) - 1.0
                    
                    b = b_odds - 1.0
                    q = 1.0 - m_prob
                    full_k = (m_prob * b - q) / b if b > 0 else 0
                    q_kelly = max(0.0, full_k * 0.25) * 100
                    
                    table_data.append({
                        "Selection": selection,
                        "Bookie Odds": f"{b_odds:.2f}",
                        "Model Prob": f"{m_prob*100:.1f}%",
                        "Expected Value (EV)": f"{ev*100:+.1f}%",
                        "Rec. Stake (% Capital)": f"{q_kelly:.1f}%" if ev > 0 else "0.0%",
                        "Value Status": "🟢 VALUE" if ev > 0 else "🔴 NO VALUE"
                    })
                
                df_match = pd.DataFrame(table_data)
                
                with col_table:
                    st.dataframe(df_match, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# TAB 2: EXECUTE TICKET
# ---------------------------------------------------------------------
with tab2:
    st.markdown("### Record Ticket Placed with Bookmaker")
    
    with st.form("bet_entry_form"):
        col_x, col_y = st.columns(2)
        with col_x:
            match_name = st.text_input("Match Name", value="Arsenal vs Chelsea")
            competition = st.selectbox("League / Competition", ["Premier League", "La Liga", "Serie A", "Bundesliga", "UEFA Champions League", "Other"], index=0)
            selection = st.selectbox("Selection Placed", ["Home", "Draw", "Away", "Over 2.5 Goals", "Under 2.5 Goals"])
            executed_odds = st.number_input("Odds Secured", value=1.85, step=0.01)
        
        with col_y:
            match_date = st.date_input("Date Placed", value=date.today(), format="DD/MM/YYYY")
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
            st.rerun()

# ---------------------------------------------------------------------
# TAB 3: CAPITAL LEDGER
# ---------------------------------------------------------------------
with tab3:
    st.markdown("### Portfolio Performance & Ticket Settlement")
    
    with sqlite3.connect(DB_PATH) as conn:
        df_bets = pd.read_sql_query("SELECT * FROM live_bets", conn)
        
    if df_bets.empty:
        st.info("No bets recorded in ledger yet.")
    else:
        df_bets['match_date_dt'] = pd.to_datetime(df_bets['match_date']).dt.date
        mask = (df_bets['match_date_dt'] >= start_date) & (df_bets['match_date_dt'] <= end_date)
        if selected_league != "All Leagues":
            mask = mask & (df_bets['competition'] == selected_league)
            
        filtered_df = df_bets[mask]
        
        open_bets = filtered_df[filtered_df['status'] == 'OPEN']
        settled_bets = filtered_df[filtered_df['status'] != 'OPEN']
        
        total_staked = settled_bets['cash_staked'].sum() if not settled_bets.empty else 0
        total_pnl = settled_bets['net_pnl'].sum() if not settled_bets.empty else 0
        roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric("TOTAL STAKED (FILTERED)", f"${total_staked:,.2f}")
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
            st.write("No open tickets matching current filters.")
            
        if not settled_bets.empty:
            st.markdown("---")
            st.markdown("### Cumulative PnL Curve")
            settled_bets['Cumulative_PnL'] = settled_bets['net_pnl'].cumsum()
            st.line_chart(settled_bets['Cumulative_PnL'])
