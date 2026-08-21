import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime, date
import streamlit as st

# =====================================================================
# 1. PAGE SETUP & HIGH-CONTRAST SPORTPESA THEMING
# =====================================================================
st.set_page_config(
    page_title="QUANT FOOTBALL | SPORTPESA EDITION", 
    layout="wide", 
    page_icon="⚽",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #060913 0%, #0a1128 50%, #001220 100%);
        color: #FFFFFF;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #050b14 !important;
        border-right: 2px solid #00E5FF;
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
        color: #00E5FF !important;
        font-weight: 700 !important;
    }
    
    .match-header {
        background: #0d1b2a;
        border: 1px solid #1b263b;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .badge-value {
        background-color: #00FF88;
        color: #000000;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 900;
        font-size: 13px;
    }

    .badge-novalue {
        background-color: #2a111e;
        color: #FF3366;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 800;
        font-size: 13px;
    }
    
    .bankroll-card {
        background: #0d1b2a;
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
# 2. DIXON-COLES PROBABILITY ENGINE
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
    
    # Both Teams To Score (BTTS)
    p_btts_yes = float(np.sum(matrix[1:, 1:]))
    p_btts_no = 1.0 - p_btts_yes
    
    return {
        "1": p_home, "X": p_draw, "2": p_away, 
        "Over 2.5": p_over, "Under 2.5": p_under,
        "BTTS Yes": p_btts_yes, "BTTS No": p_btts_no
    }

# =====================================================================
# 3. FIXTURES WITH CREST URLS & SPORTPESA MARKETS
# =====================================================================
def get_sportpesa_fixtures():
    today_str = date.today().strftime("%Y-%m-%d")
    return [
        {
            "id": "SP_001",
            "date": today_str, 
            "league": "Premier League", 
            "home": "Arsenal", 
            "away": "Chelsea", 
            "home_logo": "https://crests.football-data.org/57.png",
            "away_logo": "https://crests.football-data.org/61.png",
            "lambda_h": 1.85, "mu_a": 0.95, 
            "sportpesa_odds": {"1": 1.85, "X": 3.80, "2": 4.50, "Over 2.5": 1.95, "Under 2.5": 1.90, "BTTS Yes": 1.80, "BTTS No": 2.00}
        },
        {
            "id": "SP_002",
            "date": today_str, 
            "league": "Premier League", 
            "home": "Liverpool", 
            "away": "Manchester City", 
            "home_logo": "https://crests.football-data.org/64.png",
            "away_logo": "https://crests.football-data.org/65.png",
            "lambda_h": 1.60, "mu_a": 1.55, 
            "sportpesa_odds": {"1": 2.40, "X": 3.60, "2": 2.80, "Over 2.5": 1.70, "Under 2.5": 2.15, "BTTS Yes": 1.55, "BTTS No": 2.30}
        },
        {
            "id": "SP_003",
            "date": today_str, 
            "league": "La Liga", 
            "home": "Real Madrid", 
            "away": "FC Barcelona", 
            "home_logo": "https://crests.football-data.org/86.png",
            "away_logo": "https://crests.football-data.org/81.png",
            "lambda_h": 1.90, "mu_a": 1.30, 
            "sportpesa_odds": {"1": 2.05, "X": 3.70, "2": 3.40, "Over 2.5": 1.65, "Under 2.5": 2.25, "BTTS Yes": 1.60, "BTTS No": 2.20}
        }
    ]

# =====================================================================
# 4. SIDEBAR & BANKROLL TRACKER
# =====================================================================
with st.sidebar:
    st.title("⚙️ CONTROL PANEL")
    
    st.markdown("### 📅 Date Filter")
    start_date = st.date_input("Start Date", value=date.today(), format="DD/MM/YYYY")
    end_date = st.date_input("End Date", value=date.today(), format="DD/MM/YYYY")
    
    st.markdown("---")
    st.markdown("### 🏆 League Filter")
    league_options = ["All Leagues", "Premier League", "La Liga", "Serie A", "Bundesliga"]
    selected_league = st.selectbox("Select League", league_options)
    
    st.markdown("---")
    st.markdown("### 🇰🇪 SportPesa Bankroll")
    initial_bankroll = st.number_input("Capital (KES)", value=10000.0, step=1000.0)
    
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
            <div class="bankroll-title">Open Bets Stake</div>
            <div class="bankroll-value" style="color: #FFCC00;">KES {total_open_stake:,.2f}</div>
        </div>
        <div class="bankroll-card">
            <div class="bankroll-title">Available Capital</div>
            <div class="bankroll-value" style="color: #00FFCC;">KES {current_available:,.2f}</div>
        </div>
    """, unsafe_allow_html=True)

# =====================================================================
# 5. MAIN NAVIGATION DECK
# =====================================================================
st.title("⚽ SPORTPESA VALUE SCANNER")
st.write("---")

tab1, tab2, tab3 = st.tabs([
    "🎯 LIVE FIXTURES SCANNER", 
    "📝 EXECUTE TICKET", 
    "📈 BANKROLL LEDGER"
])

# ---------------------------------------------------------------------
# TAB 1: VISUAL SPORTPESA FIXTURES & VALUE INDICATORS
# ---------------------------------------------------------------------
with tab1:
    fixtures = get_sportpesa_fixtures()
    if selected_league != "All Leagues":
        fixtures = [f for f in fixtures if f["league"] == selected_league]
        
    for match in fixtures:
        probs = calculate_match_metrics(match['lambda_h'], match['mu_a'])
        
        # Match Visual Header with Team Logos
        st.markdown(f"""
            <div class="match-header">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <img src="{match['home_logo']}" width="36" height="36"/>
                        <span style="font-size: 18px; font-weight: 800;">{match['home']}</span>
                        <span style="color: #94A3B8; font-weight: 700;">VS</span>
                        <span style="font-size: 18px; font-weight: 800;">{match['away']}</span>
                        <img src="{match['away_logo']}" width="36" height="36"/>
                    </div>
                    <div style="color: #00E5FF; font-weight: 700; font-size: 13px; background: #050b14; padding: 4px 10px; border-radius: 6px;">
                        {match['league']}
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        table_rows = []
        for market in ["1", "X", "2", "Over 2.5", "Under 2.5", "BTTS Yes", "BTTS No"]:
            m_prob = probs[market]
            sp_odds = match["sportpesa_odds"][market]
            ev = (m_prob * sp_odds) - 1.0
            
            b = sp_odds - 1.0
            q = 1.0 - m_prob
            full_k = (m_prob * b - q) / b if b > 0 else 0
            q_kelly = max(0.0, full_k * 0.25) * 100
            rec_kes = (q_kelly / 100) * current_available
            
            table_rows.append({
                "Market": market,
                "SportPesa Odds": f"{sp_odds:.2f}",
                "Model Prob": f"{m_prob*100:.1f}%",
                "Expected Value (EV)": f"{ev*100:+.1f}%",
                "Kelly Stake (%)": f"{q_kelly:.1f}%",
                "Rec. Stake (KES)": f"KES {rec_kes:,.0f}" if ev > 0 else "KES 0",
                "Edge Status": "🟢 VALUE" if ev > 0 else "🔴 NO VALUE"
            })
            
        df_sp = pd.DataFrame(table_rows)
        
        c_table, c_space = st.columns([4, 1])
        with c_table:
            st.dataframe(df_sp, use_container_width=True, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# TAB 2: EXECUTE TICKET
# ---------------------------------------------------------------------
with tab2:
    st.markdown("### Record SportPesa Bet Ticket")
    
    with st.form("sp_bet_entry"):
        col_x, col_y = st.columns(2)
        with col_x:
            match_name = st.text_input("Match Name", value="Arsenal vs Chelsea")
            competition = st.selectbox("League", ["Premier League", "La Liga", "Serie A", "Bundesliga", "Other"])
            selection = st.selectbox("Selection Placed", ["1", "X", "2", "Over 2.5", "Under 2.5", "BTTS Yes", "BTTS No"])
            executed_odds = st.number_input("SportPesa Odds Secured", value=1.85, step=0.01)
        
        with col_y:
            match_date = st.date_input("Date Placed", value=date.today(), format="DD/MM/YYYY")
            cash_staked = st.number_input("Cash Staked (KES)", value=500.0, step=100.0)
            model_prob = st.number_input("Model Prob (%)", value=62.0, step=1.0)
            
        submit_btn = st.form_submit_button("💾 Save Ticket to Ledger")
        
        if submit_btn:
            ticket_id = f"SP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO live_bets 
                    (ticket_id, placed_at, match_date, competition, match_name, selection, executed_odds, cash_staked, model_prob)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (ticket_id, datetime.now().isoformat(), str(match_date), competition, match_name, selection, executed_odds, cash_staked, model_prob))
                conn.commit()
            st.success(f"✅ Ticket {ticket_id} successfully recorded!")
            st.rerun()

# ---------------------------------------------------------------------
# TAB 3: CAPITAL LEDGER
# ---------------------------------------------------------------------
with tab3:
    st.markdown("### Bankroll Ledger & Ticket Settlement")
    
    with sqlite3.connect(DB_PATH) as conn:
        df_bets = pd.read_sql_query("SELECT * FROM live_bets", conn)
        
    if df_bets.empty:
        st.info("No tickets recorded yet.")
    else:
        open_bets = df_bets[df_bets['status'] == 'OPEN']
        settled_bets = df_bets[df_bets['status'] != 'OPEN']
        
        total_staked = settled_bets['cash_staked'].sum() if not settled_bets.empty else 0
        total_pnl = settled_bets['net_pnl'].sum() if not settled_bets.empty else 0
        roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric("TOTAL STAKED", f"KES {total_staked:,.2f}")
        m2.metric("NET PnL", f"KES {total_pnl:+,.2f}")
        m3.metric("ACTUAL YIELD (ROI)", f"{roi:+.2f}%")
        
        st.markdown("---")
        st.markdown("### Settle Open SportPesa Tickets")
        
        if not open_bets.empty:
            for idx, row in open_bets.iterrows():
                b1, b2, b3 = st.columns([3, 2, 1])
                with b1:
                    st.markdown(f"**{row['match_name']}** — `{row['selection']}` @ `{row['executed_odds']:.2f}` (Stake: `KES {row['cash_staked']:,.2f}`)")
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
            st.write("No open tickets in ledger.")
            
        if not settled_bets.empty:
            st.markdown("---")
            st.markdown("### Cumulative Capital Growth Curve")
            settled_bets['Cumulative_PnL'] = settled_bets['net_pnl'].cumsum()
            st.line_chart(settled_bets['Cumulative_PnL'])
