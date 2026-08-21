import sqlite3
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
from datetime import datetime, date
import streamlit as st

# =====================================================================
# 1. PAGE SETUP & SPORTPESA HIGH-CONTRAST THEMING
# =====================================================================
st.set_page_config(
    page_title="QUANT FOOTBALL | SPORTPESA SIMPLIFIED", 
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
    
    .match-card {
        background: #0d1b2a;
        border: 1px solid #1b263b;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
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
# 2. POISSON & DIXON-COLES ENGINE
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
    
    p_btts_yes = float(np.sum(matrix[1:, 1:]))
    p_btts_no = 1.0 - p_btts_yes
    
    return {
        "1": p_home, "X": p_draw, "2": p_away, 
        "Over 2.5": p_over, "Under 2.5": p_under,
        "BTTS Yes": p_btts_yes, "BTTS No": p_btts_no
    }

def get_best_bet_selection(probs, odds_dict):
    """Selects the single best bet (Growth Anchor if >75% prob, else best +EV Alpha Generator)."""
    best_market = "1"
    best_score = -999.0
    strategy_type = "Growth Anchor"
    
    for mkt, prob in probs.items():
        sp_odds = odds_dict.get(mkt, 1.80)
        ev = (prob * sp_odds) - 1.0
        
        # Scoring logic: prioritize high probability safety or high EV
        if prob >= 0.75:
            score = prob * 2.0 # Heavy weight on high safety
            current_strat = "🛡️ Growth Anchor"
        else:
            score = ev * 3.0   # Heavy weight on value edge
            current_strat = "⚡ Alpha Generator"
            
        if score > best_score:
            best_score = score
            best_market = mkt
            strategy_type = current_strat
            
    return best_market, strategy_type

# =====================================================================
# 3. DYNAMIC FIXTURE LOADER
# =====================================================================
LEAGUE_CODES = {
    "Premier League": "PL",
    "La Liga": "PD",
    "Serie A": "SA",
    "Bundesliga": "BL1",
    "UEFA Champions League": "CL"
}

@st.cache_data(ttl=3600)
def fetch_league_fixtures(league_name, date_from, date_to, api_key=""):
    if not api_key:
        today_str = date.today().strftime("%Y-%m-%d")
        all_fixtures = [
            {"date": today_str, "league": "Premier League", "home": "Arsenal", "away": "Chelsea", "home_logo": "https://crests.football-data.org/57.png", "away_logo": "https://crests.football-data.org/61.png", "lambda_h": 1.85, "mu_a": 0.95, "odds": {"1": 1.85, "X": 3.80, "2": 4.50, "Over 2.5": 1.95, "Under 2.5": 1.90, "BTTS Yes": 1.80, "BTTS No": 2.00}},
            {"date": today_str, "league": "Premier League", "home": "Liverpool", "away": "Manchester City", "home_logo": "https://crests.football-data.org/64.png", "away_logo": "https://crests.football-data.org/65.png", "lambda_h": 1.60, "mu_a": 1.55, "odds": {"1": 2.40, "X": 3.60, "2": 2.80, "Over 2.5": 1.70, "Under 2.5": 2.15, "BTTS Yes": 1.55, "BTTS No": 2.30}},
            {"date": today_str, "league": "Premier League", "home": "Manchester United", "away": "Tottenham", "home_logo": "https://crests.football-data.org/66.png", "away_logo": "https://crests.football-data.org/73.png", "lambda_h": 1.50, "mu_a": 1.40, "odds": {"1": 2.10, "X": 3.50, "2": 3.20, "Over 2.5": 1.75, "Under 2.5": 2.05, "BTTS Yes": 1.65, "BTTS No": 2.10}},
            {"date": today_str, "league": "La Liga", "home": "Real Madrid", "away": "FC Barcelona", "home_logo": "https://crests.football-data.org/86.png", "away_logo": "https://crests.football-data.org/81.png", "lambda_h": 1.90, "mu_a": 1.30, "odds": {"1": 2.05, "X": 3.70, "2": 3.40, "Over 2.5": 1.65, "Under 2.5": 2.25, "BTTS Yes": 1.60, "BTTS No": 2.20}},
            {"date": today_str, "league": "Serie A", "home": "Inter Milan", "away": "AC Milan", "home_logo": "https://crests.football-data.org/108.png", "away_logo": "https://crests.football-data.org/98.png", "lambda_h": 1.65, "mu_a": 1.10, "odds": {"1": 2.00, "X": 3.40, "2": 3.80, "Over 2.5": 1.85, "Under 2.5": 1.95, "BTTS Yes": 1.75, "BTTS No": 2.00}},
        ]
        if league_name != "All Leagues":
            all_fixtures = [f for f in all_fixtures if f["league"] == league_name]
        return all_fixtures

    code = LEAGUE_CODES.get(league_name, "")
    url = f"https://api.football-data.org/v4/competitions/{code}/matches?dateFrom={date_from}&dateTo={date_to}"
    headers = {"X-Auth-Token": api_key}
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        parsed = []
        for m in data.get("matches", []):
            parsed.append({
                "date": m["utcDate"][:10],
                "league": league_name,
                "home": m["homeTeam"]["name"],
                "away": m["awayTeam"]["name"],
                "home_logo": m["homeTeam"].get("crest", ""),
                "away_logo": m["awayTeam"].get("crest", ""),
                "lambda_h": 1.60,
                "mu_a": 1.10,
                "odds": {"1": 2.00, "X": 3.40, "2": 3.60, "Over 2.5": 1.90, "Under 2.5": 1.90, "BTTS Yes": 1.75, "BTTS No": 2.00}
            })
        return parsed
    except Exception:
        return []

# =====================================================================
# 4. SIDEBAR CONTROLS
# =====================================================================
with st.sidebar:
    st.title("⚙️ CONTROL PANEL")
    
    st.markdown("### 📅 Date Range Filter")
    start_date = st.date_input("Start Date", value=date.today(), format="DD/MM/YYYY")
    end_date = st.date_input("End Date", value=date.today(), format="DD/MM/YYYY")
    
    st.markdown("---")
    st.markdown("### 🏆 League Filter")
    league_options = ["All Leagues", "Premier League", "La Liga", "Serie A", "Bundesliga", "UEFA Champions League"]
    selected_league = st.selectbox("Select League", league_options)
    
    st.markdown("---")
    st.markdown("### 🔑 Football API Token (Optional)")
    api_token = st.text_input("football-data.org Key", type="password", help="Leave blank to use internal fixture database")
    
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
st.title("⚽ SPORTPESA SIMPLIFIED SCANNER")
st.markdown(f"<p style='color: #00FFCC; font-size: 14px; font-weight: 700;'>FILTER: {start_date.strftime('%d/%m/%Y')} TO {end_date.strftime('%d/%m/%Y')} | LEAGUE: {selected_league.upper()}</p>", unsafe_allow_html=True)
st.write("---")

tab1, tab2, tab3 = st.tabs([
    "🎯 TOP PICKS & PARLAY", 
    "📝 EXECUTE TICKET", 
    "📈 CAPITAL LEDGER"
])

# ---------------------------------------------------------------------
# TAB 1: SIMPLIFIED SINGLE BEST BET PER MATCH & PARLAY
# ---------------------------------------------------------------------
with tab1:
    matches = fetch_league_fixtures(
        selected_league, 
        start_date.strftime("%Y-%m-%d"), 
        end_date.strftime("%Y-%m-%d"), 
        api_token
    )
    
    if not matches:
        st.warning(f"No matches scheduled for {selected_league} between {start_date.strftime('%d/%m/%Y')} and {end_date.strftime('%d/%m/%Y')}.")
    else:
        st.markdown(f"### High-Confidence Picks (**{len(matches)}** Match)")
        
        summary_rows = []
        for match in matches:
            probs = calculate_match_metrics(match['lambda_h'], match['mu_a'])
            best_market, strategy_type = get_best_bet_selection(probs, match['odds'])
            
            m_prob = probs[best_market]
            sp_odds = match['odds'][best_market]
            ev = (m_prob * sp_odds) - 1.0
            
            # Kelly stake suggestion
            b = sp_odds - 1.0
            q = 1.0 - m_prob
            full_k = (m_prob * b - q) / b if b > 0 else 0
            rec_stake = max(0.0, full_k * 0.25) * current_available
            
            summary_rows.append({
                "Match": f"{match['home']} vs {match['away']}",
                "League": match['league'],
                "Best Selection": best_market,
                "Category": strategy_type,
                "Odds": f"{sp_odds:.2f}",
                "Model Prob": f"{m_prob*100:.1f}%",
                "Expected Value": f"{ev*100:+.1f}%",
                "Rec. Stake": f"KES {rec_stake:,.0f}" if ev > -0.05 else "KES 0"
            })
            
            # Match Card UI with Logos
            st.markdown(f"""
                <div class="match-card">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <img src="{match['home_logo']}" width="28" height="28" />
                            <span style="font-size: 16px; font-weight: 800;">{match['home']}</span>
                            <span style="color: #00E5FF; font-weight: 900; margin: 0 4px;">VS</span>
                            <span style="font-size: 16px; font-weight: 800;">{match['away']}</span>
                            <img src="{match['away_logo']}" width="28" height="28" />
                        </div>
                        <div style="color: #94A3B8; font-size: 12px; font-weight: 700;">
                            {match['date']} | <span style="color: #00FFCC;">{match['league']}</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📋 Recommended Single Bets Matrix")
        df_summary = pd.DataFrame(summary_rows)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### 🏆 High-Safety Cup Parlay Multi-Bet")
        parlay_odds = 3.12
        parlay_stake = 500.0
        st.markdown(f"""
            * **Leg 1:** Arsenal vs Chelsea — **1X (Double Chance)** @ `1.22`
            * **Leg 2:** Real Madrid vs FC Barcelona — **1 (Home Win)** @ `2.05`
            * **Leg 3:** Inter Milan vs AC Milan — **Over 1.5 Goals** @ `1.25`
            
            > **Combined Parlay Odds:** `{parlay_odds}` | **Recommended Stake:** `KES {parlay_stake:,.0f}` | **Potential Payout:** `KES {parlay_stake * parlay_odds:,.2f}`
        """)

# ---------------------------------------------------------------------
# TAB 2: EXECUTE TICKET
# ---------------------------------------------------------------------
with tab2:
    st.markdown("### Record SportPesa Bet Ticket")
    
    with st.form("bet_entry_form"):
        col_x, col_y = st.columns(2)
        with col_x:
            match_name = st.text_input("Match Name", value="Arsenal vs Chelsea")
            competition = st.selectbox("League", ["Premier League", "La Liga", "Serie A", "Bundesliga", "UEFA Champions League", "Other"])
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
            st.success(f"✅ Ticket {ticket_id} successfully logged!")
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
        st.markdown("### Settle Open Tickets")
        
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
            st.markdown("### Cumulative Capital Curve")
            settled_bets['Cumulative_PnL'] = settled_bets['net_pnl'].cumsum()
            st.line_chart(settled_bets['Cumulative_PnL'])
