import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime
import streamlit as st

# =====================================================================
# 1. PAGE SETUP & STYLING
# =====================================================================
st.set_page_config(
    page_title="Quant Football Analytics", 
    layout="wide", 
    page_icon="⚽"
)

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
    
    return {"Home": prob_home, "Draw": prob_draw, "Away": prob_away}

# =====================================================================
# 3. INTERACTIVE NAVIGATION & TABS
# =====================================================================
st.title("⚽ Quantitative Football Analytics & Ledger")

tab1, tab2, tab3 = st.tabs([
    "🔥 Match Valuation & +EV Scanner", 
    "🎟️ Log Executed Wager", 
    "📊 Capital Ledger & Analytics"
])

# ---------------------------------------------------------------------
# TAB 1: MATCH VALUATION & +EV SCANNER
# ---------------------------------------------------------------------
with tab1:
    st.subheader("Interactive Match Odds Evaluator")
    
    col_a, col_b = st.columns(2)
    with col_a:
        match_title = st.text_input("Fixture Name", value="Arsenal vs Chelsea")
        lambda_h = st.number_input("Home Expected Goals (λ)", value=1.85, step=0.05)
        mu_a = st.number_input("Away Expected Goals (μ)", value=0.95, step=0.05)
    
    with col_b:
        odds_h = st.number_input("Bookmaker Home Odds", value=1.85, step=0.01)
        odds_d = st.number_input("Bookmaker Draw Odds", value=3.80, step=0.01)
        odds_a = st.number_input("Bookmaker Away Odds", value=4.50, step=0.01)
        
    probs = calculate_match_probabilities(lambda_h, mu_a)
    
    results = []
    for sel, bookie_odds in [("Home", odds_h), ("Draw", odds_d), ("Away", odds_a)]:
        model_p = probs[sel]
        ev = (model_p * bookie_odds) - 1.0
        
        # Quarter Kelly calculation
        b = bookie_odds - 1.0
        q = 1.0 - model_p
        full_k = (model_p * b - q) / b if b > 0 else 0
        q_kelly_pct = max(0.0, full_k * 0.25) * 100

        results.append({
            "Selection": sel,
            "Bookmaker Odds": bookie_odds,
            "Model Probability": f"{model_p * 100:.1f}%",
            "+EV Edge": f"{ev * 100:+.1f}%",
            "Suggested Stake (%)": f"{q_kelly_pct:.2f}%",
            "Status": "✅ VALUE" if ev > 0 else "❌ NO VALUE"
        })

    st.write(f"### Valuation for {match_title}")
    st.dataframe(pd.DataFrame(results), use_container_width=True)

# ---------------------------------------------------------------------
# TAB 2: LOG EXECUTED WAGER
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Record Live Wager Slipped with Bookmaker")
    
    with st.form("bet_entry_form"):
        c1, c2 = st.columns(2)
        with c1:
            match_name = st.text_input("Match Name", value="Arsenal vs Chelsea")
            competition = st.selectbox("Competition", ["Premier League", "La Liga", "Bundesliga", "Serie A", "Other"])
            selection = st.selectbox("Selection", ["Home", "Draw", "Away", "Over 2.5", "Under 2.5"])
            executed_odds = st.number_input("Odds Secured", value=1.85, step=0.01)
        
        with c2:
            match_date = st.date_input("Match Date", datetime.now())
            cash_staked = st.number_input("Cash Staked ($ / KES)", value=100.0, step=10.0)
            model_prob = st.number_input("Model Probability (%)", value=62.0, step=1.0)
            
        submitted = st.form_submit_button("Log Ticket into Ledger")
        
        if submitted:
            ticket_id = f"TICK_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO live_bets 
                    (ticket_id, placed_at, match_date, competition, match_name, selection, executed_odds, cash_staked, model_prob)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (ticket_id, datetime.now().isoformat(), str(match_date), competition, match_name, selection, executed_odds, cash_staked, model_prob))
                conn.commit()
            st.success(f"Ticket {ticket_id} logged into master ledger!")

# ---------------------------------------------------------------------
# TAB 3: CAPITAL LEDGER & ANALYTICS
# ---------------------------------------------------------------------
with tab3:
    st.subheader("Settle Open Bets & Track Live Equity")
    
    with sqlite3.connect(DB_PATH) as conn:
        df_bets = pd.read_sql_query("SELECT * FROM live_bets", conn)
        
    if df_bets.empty:
        st.info("No wagers recorded in ledger yet.")
    else:
        st.write("### Pending Open Tickets")
        open_bets = df_bets[df_bets['status'] == 'OPEN']
        
        if not open_bets.empty:
            for idx, row in open_bets.iterrows():
                col_x, col_y, col_z = st.columns([3, 2, 2])
                with col_x:
                    st.write(f"**{row['match_name']}** — {row['selection']} @ {row['executed_odds']} (Stake: {row['cash_staked']})")
                with col_y:
                    outcome = st.selectbox("Outcome", ["WIN", "LOSS", "VOID"], key=f"sel_{row['ticket_id']}")
                with col_z:
                    if st.button("Settle Ticket", key=f"btn_{row['ticket_id']}"):
                        pnl = (row['cash_staked'] * (row['executed_odds'] - 1)) if outcome == "WIN" else (-row['cash_staked'] if outcome == "LOSS" else 0.0)
                        with sqlite3.connect(DB_PATH) as conn:
                            cursor = conn.cursor()
                            cursor.execute("UPDATE live_bets SET status = ?, net_pnl = ? WHERE ticket_id = ?", (outcome, pnl, row['ticket_id']))
                            conn.commit()
                        st.rerun()
        else:
            st.write("No pending tickets.")
            
        settled_bets = df_bets[df_bets['status'] != 'OPEN']
        if not settled_bets.empty:
            st.write("---")
            st.write("### Portfolio Performance")
            
            total_staked = settled_bets['cash_staked'].sum()
            total_pnl = settled_bets['net_pnl'].sum()
            roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Capital Staked", f"${total_staked:,.2f}")
            m2.metric("Net Profit / Loss", f"${total_pnl:+,.2f}")
            m3.metric("Actual Yield (ROI)", f"{roi:+.2f}%")
            
            settled_bets['Cumulative_PnL'] = settled_bets['net_pnl'].cumsum()
            st.line_chart(settled_bets['Cumulative_PnL'])
