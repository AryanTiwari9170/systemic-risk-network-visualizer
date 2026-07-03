import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import sqlite3
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. PAGE CONFIG & THEME
# ============================================================
st.set_page_config(page_title="Systemic Risk Network Visualizer", layout="wide", page_icon="🕸️", initial_sidebar_state="expanded")

DARK_BG, PANEL_BG, GRID_C, TEXT_C = "#0F172A", "#1E293B", "#334155", "#CBD5E1"
COLORS = {"safe": "#10B981", "warn": "#F59E0B", "danger": "#EF4444", "neutral": "#3B82F6"}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.2rem; max-width: 1400px; }
.deal-banner {
    background: linear-gradient(120deg, #0F172A 0%, #1E293B 60%, #0F172A 100%);
    border: 1px solid #334155; border-radius: 14px; padding: 20px 26px; margin-bottom: 18px;
}
.deal-title { font-size: 21px; font-weight: 800; color: #F8FAFC; }
.deal-sub { font-size: 12px; color: #94A3B8; margin-top: 4px; }
.kpi-card {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); border: 1px solid #334155;
    border-radius: 12px; padding: 16px 20px; flex: 1; min-width: 160px;
}
.kpi-label { font-size: 10.5px; color: #94A3B8; text-transform: uppercase; letter-spacing: .8px; margin-bottom: 6px; font-weight: 600; }
.kpi-value { font-size: 23px; font-weight: 800; color: #F1F5F9; }
.kpi-sub { font-size: 11px; color: #64748B; margin-top: 4px; }
.section-header {
    background: #0F172A; border-left: 4px solid #3B82F6; padding: 8px 14px; border-radius: 6px;
    color: #F1F5F9; font-size: 14px; font-weight: 700; margin: 18px 0 10px 0;
}
.quiz-box { background: #111C33; border: 1px solid #334155; border-radius: 10px; padding: 16px 20px; margin-bottom: 14px; }
.sql-note { background: #111C33; border: 1px solid #334155; border-left: 4px solid #10B981; border-radius: 8px; padding: 12px 16px; font-size: 12.5px; color: #94A3B8; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

def kpi(label, value, sub):
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG, font=dict(color=TEXT_C, family="Inter, sans-serif", size=12),
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(visible=False), yaxis=dict(visible=False),
    hoverlabel=dict(bgcolor=PANEL_BG, font_color=TEXT_C, bordercolor=GRID_C),
)
# Bar/line charts need visible axes, unlike the network graphs above — so this variant
# omits xaxis/yaxis to avoid a "multiple values for keyword argument" clash when charts
# pass their own xaxis=/yaxis= into update_layout().
PLOTLY_LAYOUT_AXES = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")}

# ============================================================
# 2. SQLITE DATABASE LAYER
# ============================================================
DB_PATH = "systemic_risk_results.db"

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS simulation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_seed INTEGER,
            num_banks INTEGER,
            recovery_rate REAL,
            bank TEXT,
            assets REAL,
            capital REAL,
            eigenvector REAL,
            defaults_caused INTEGER,
            capital_destroyed REAL,
            systemic_score REAL,
            tbtf_index REAL,
            systemic_rank INTEGER,
            run_timestamp TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ============================================================
# 3. SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.markdown("## 🕸️ Systemic Risk Visualizer")
    st.caption("Interbank Contagion Simulation — self-directed project")
    st.markdown("---")
    with st.expander("🏦 Network Generation", expanded=True):
        num_banks = st.slider("Number of Banks", 10, 40, 20)
        connection_probability = st.slider("Connection Probability", 0.05, 0.5, 0.20, 0.05)
        max_exposure_fraction = st.slider("Max Exposure (% of lender assets)", 0.05, 0.5, 0.20, 0.05)
        seed = st.number_input("Random Seed", value=42, step=1)
    with st.expander("💥 Contagion Parameters", expanded=True):
        recovery_rate = st.slider("Recovery Rate on Default", 0.0, 0.9, 0.4, 0.05)
    st.markdown("---")
    with st.expander("🗄️ Database", expanded=False):
        row_count = pd.read_sql_query("SELECT COUNT(*) as c FROM simulation_runs", get_connection())["c"].iloc[0]
        st.caption(f"{row_count} rows stored in `{DB_PATH}`")
        if st.button("🗑️ Clear all stored results", use_container_width=True):
            conn = get_connection()
            conn.execute("DELETE FROM simulation_runs")
            conn.commit()
            conn.close()
            st.rerun()
    st.markdown("---")
    st.caption("Self-initiated project · Python, NetworkX, Plotly, SQLite")

np.random.seed(int(seed))

# ============================================================
# 4. NETWORK GENERATION & METRICS
# ============================================================
def safe_eigenvector(G):
    """Eigenvector centrality requires a connected graph; falls back to PageRank otherwise."""
    try:
        return nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        return nx.pagerank(G, weight="weight")

@st.cache_data(show_spinner=False)
def build_network(num_banks, connection_probability, max_exposure_fraction, seed):
    np.random.seed(int(seed))
    bank_ids = [f"Bank_{i+1}" for i in range(num_banks)]
    assets = np.random.uniform(1000, 10000, num_banks)
    capital_ratios = np.random.uniform(0.08, 0.15, num_banks)
    capital = assets * capital_ratios
    banks_df = pd.DataFrame({"Bank": bank_ids, "Assets": assets, "Capital": capital}).round(2)

    exposures = []
    for lender in banks_df["Bank"]:
        lender_assets = banks_df.loc[banks_df["Bank"] == lender, "Assets"].values[0]
        for borrower in banks_df["Bank"]:
            if lender != borrower and np.random.rand() < connection_probability:
                amt = np.random.uniform(0, max_exposure_fraction * lender_assets)
                exposures.append({"Lender": lender, "Borrower": borrower, "Exposure": round(amt, 2)})
    exposure_df = pd.DataFrame(exposures)

    G = nx.DiGraph()
    for _, row in banks_df.iterrows():
        G.add_node(row["Bank"], assets=row["Assets"], capital=row["Capital"])
    for _, row in exposure_df.iterrows():
        G.add_edge(row["Lender"], row["Borrower"], weight=row["Exposure"])
    return banks_df, exposure_df, G

banks_df, exposure_df, G = build_network(num_banks, connection_probability, max_exposure_fraction, seed)

@st.cache_data(show_spinner=False)
def compute_metrics(_G, banks_tuple):
    G = _G
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    pagerank = nx.pagerank(G, weight="weight")
    eigenvector = safe_eigenvector(G)
    return pd.DataFrame({
        "Bank": list(G.nodes()),
        "Assets": [G.nodes[n]["assets"] for n in G.nodes()],
        "Capital": [G.nodes[n]["capital"] for n in G.nodes()],
        "Betweenness": [betweenness.get(n, 0) for n in G.nodes()],
        "PageRank": [pagerank.get(n, 0) for n in G.nodes()],
        "Eigenvector": [eigenvector.get(n, 0) for n in G.nodes()],
    })

network_metrics_df = compute_metrics(G, tuple(banks_df["Bank"]))

# ============================================================
# 5. CONTAGION SIMULATION
# ============================================================
def simulate_contagion(G, initial_default, recovery_rate):
    capital = {node: G.nodes[node]["capital"] for node in G.nodes()}
    original_total = sum(capital.values())
    defaulted = {initial_default}
    newly_defaulted = {initial_default}
    layers = {0: [initial_default]}
    layer_num = 1
    while newly_defaulted:
        next_defaults = set()
        for def_bank in newly_defaulted:
            for creditor in G.predecessors(def_bank):
                if creditor not in defaulted:
                    loss = G[creditor][def_bank]["weight"] * (1 - recovery_rate)
                    capital[creditor] -= loss
                    if capital[creditor] < 0:
                        next_defaults.add(creditor)
        if next_defaults:
            layers[layer_num] = list(next_defaults)
        newly_defaulted = next_defaults
        defaulted.update(next_defaults)
        layer_num += 1
    remaining = sum(capital[n] for n in capital if n not in defaulted)
    capital_destroyed = original_total - remaining
    return defaulted, capital_destroyed, layers

@st.cache_data(show_spinner=False)
def run_all_scenarios(_G, banks_tuple, recovery_rate):
    G = _G
    records = []
    for bank in banks_tuple:
        defaulted, capital_destroyed, _ = simulate_contagion(G, bank, recovery_rate)
        records.append({"Bank": bank, "Defaults_Caused": len(defaulted) - 1, "Capital_Destroyed": round(capital_destroyed, 2)})
    return pd.DataFrame(records)

systemic_df = run_all_scenarios(G, tuple(banks_df["Bank"]), recovery_rate)

comparison_df = network_metrics_df.merge(systemic_df, on="Bank")
total_banks_n = len(comparison_df)
total_system_capital = comparison_df["Capital"].sum()
comparison_df["Systemic_Score"] = (
    0.4 * comparison_df["Eigenvector"].rank(pct=True) +
    0.3 * comparison_df["Assets"].rank(pct=True) +
    0.3 * comparison_df["Capital_Destroyed"].rank(pct=True)
)
comparison_df["TBTF_Index"] = (comparison_df["Defaults_Caused"] / total_banks_n) * (comparison_df["Capital_Destroyed"] / total_system_capital)
comparison_df = comparison_df.sort_values("Systemic_Score", ascending=False).reset_index(drop=True)
comparison_df["Systemic_Rank"] = comparison_df.index + 1
top_bank = comparison_df.iloc[0]["Bank"]

# ============================================================
# 6. PERSIST CURRENT RUN TO SQLITE
# ============================================================
def save_run_to_db(df, seed, num_banks, recovery_rate):
    conn = get_connection()
    for _, r in df.iterrows():
        conn.execute("""
            INSERT INTO simulation_runs
            (run_seed, num_banks, recovery_rate, bank, assets, capital, eigenvector,
             defaults_caused, capital_destroyed, systemic_score, tbtf_index, systemic_rank)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (int(seed), int(num_banks), float(recovery_rate), r["Bank"], float(r["Assets"]), float(r["Capital"]),
              float(r["Eigenvector"]), int(r["Defaults_Caused"]), float(r["Capital_Destroyed"]),
              float(r["Systemic_Score"]), float(r["TBTF_Index"]), int(r["Systemic_Rank"])))
    conn.commit()
    conn.close()

def multi_seed_analysis(seeds, num_banks, connection_probability, max_exposure_fraction, recovery_rate):
    """Runs the full pipeline across multiple random seeds and stores each into SQLite."""
    for s in seeds:
        b_df, e_df, g = build_network(num_banks, connection_probability, max_exposure_fraction, s)
        m_df = compute_metrics(g, tuple(b_df["Bank"]))
        sys_df = run_all_scenarios(g, tuple(b_df["Bank"]), recovery_rate)
        comp = m_df.merge(sys_df, on="Bank")
        tot_cap = comp["Capital"].sum()
        comp["Systemic_Score"] = (0.4 * comp["Eigenvector"].rank(pct=True) + 0.3 * comp["Assets"].rank(pct=True) + 0.3 * comp["Capital_Destroyed"].rank(pct=True))
        comp["TBTF_Index"] = (comp["Defaults_Caused"] / num_banks) * (comp["Capital_Destroyed"] / tot_cap)
        comp = comp.sort_values("Systemic_Score", ascending=False).reset_index(drop=True)
        comp["Systemic_Rank"] = comp.index + 1
        save_run_to_db(comp, s, num_banks, recovery_rate)

# ============================================================
# HELPER: network plot
# ============================================================
@st.cache_data(show_spinner=False)
def get_layout(_G, seed):
    return nx.spring_layout(_G, seed=int(seed))

pos = get_layout(G, seed)

def plot_network(highlight_layers=None, size_by="Eigenvector"):
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color=GRID_C, width=0.6), hoverinfo="none", opacity=0.4)

    size_map = comparison_df.set_index("Bank")[size_by]
    node_x, node_y, node_size, node_color, node_text = [], [], [], [], []
    bank_to_layer = {}
    if highlight_layers:
        for layer, banks in highlight_layers.items():
            for b in banks:
                bank_to_layer[b] = layer
    layer_palette = ["#EF4444", "#F59E0B", "#FBBF24", "#10B981", "#3B82F6", "#8B5CF6", "#EC4899"]

    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x); node_y.append(y)
        node_size.append(18 + 60 * (size_map.get(n, 0) / (size_map.max() or 1)))
        if highlight_layers:
            node_color.append(layer_palette[bank_to_layer[n] % len(layer_palette)] if n in bank_to_layer else "#334155")
        else:
            node_color.append(COLORS["neutral"])
        assets = G.nodes[n]["assets"]; cap = G.nodes[n]["capital"]
        node_text.append(f"{n}<br>Assets: {assets:,.0f}<br>Capital: {cap:,.0f}<br>{size_by}: {size_map.get(n,0):.4f}")

    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text", text=list(G.nodes()), textposition="top center",
                             textfont=dict(size=8, color=TEXT_C), hovertext=node_text, hoverinfo="text",
                             marker=dict(size=node_size, color=node_color, line=dict(width=1, color=DARK_BG)))
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(**PLOTLY_LAYOUT, height=500, showlegend=False)
    return fig

# ============================================================
# TOP BANNER
# ============================================================
st.markdown(f"""
<div class="deal-banner">
  <div class="deal-title">🕸️ Systemic Risk Network Visualizer</div>
  <div class="deal-sub">Interbank Contagion &amp; Too-Big-To-Fail Analysis · {num_banks} synthetic banks · {len(exposure_df)} exposure links</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🕸️ Network Overview", "💥 Contagion Simulator", "🛡️ Stress Testing", "🏆 Systemic Risk Ranking", "🗄️ SQL Database Explorer", "🎮 Quiz Challenge"])

# ---------- TAB 1: NETWORK OVERVIEW ----------
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Total Banks", f"{num_banks}", "Nodes in network")
    with c2: kpi("Exposure Links", f"{len(exposure_df)}", "Directed edges")
    with c3: kpi("Total System Capital", f"{banks_df['Capital'].sum():,.0f}", "Sum across all banks")
    with c4: kpi("Network Density", f"{nx.density(G):.3f}", "Edges / possible edges")

    st.markdown('<div class="section-header">Interactive Interbank Network (node size = eigenvector centrality)</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_network(size_by="Eigenvector"), use_container_width=True)
    st.caption("Hover over a node to see assets, capital, and centrality. Larger nodes = more systemically central.")

# ---------- TAB 2: CONTAGION SIMULATOR ----------
with tabs[1]:
    st.markdown('<div class="section-header">💥 Run a Contagion Scenario</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        shock_bank = st.selectbox("Select bank to default first", sorted(banks_df["Bank"], key=lambda x: int(x.split("_")[1])))
        run_sim = st.button("▶ Run Simulation", use_container_width=True)
    if run_sim or "last_sim" in st.session_state:
        if run_sim:
            defaulted, capital_destroyed, layers = simulate_contagion(G, shock_bank, recovery_rate)
            st.session_state["last_sim"] = (shock_bank, defaulted, capital_destroyed, layers)
        shock_bank, defaulted, capital_destroyed, layers = st.session_state["last_sim"]

        c1, c2, c3 = st.columns(3)
        with c1: kpi("Initial Default", shock_bank, "Shock origin")
        with c2: kpi("Total Defaults Triggered", f"{len(defaulted)-1}", f"out of {num_banks-1} other banks")
        with c3: kpi("Capital Destroyed", f"{capital_destroyed:,.0f}", f"{capital_destroyed/total_system_capital:.1%} of system")

        st.markdown('<div class="section-header">🌊 Stress Propagation by Layer</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_network(highlight_layers=layers, size_by="Eigenvector"), use_container_width=True)
        for layer, banks in layers.items():
            st.markdown(f"**Layer {layer}:** {', '.join(banks)}")

# ---------- TAB 3: STRESS TESTING ----------
with tabs[2]:
    st.markdown('<div class="section-header">🛡️ Stress Test — Every Bank as Initial Shock</div>', unsafe_allow_html=True)
    st.caption("Capital destroyed and defaults triggered if EACH bank were to fail first, holding recovery rate constant.")
    disp = systemic_df.copy().sort_values("Capital_Destroyed", ascending=False)

    fig = go.Figure(go.Bar(x=disp["Bank"], y=disp["Capital_Destroyed"], marker_color=COLORS["danger"]))
    fig.update_layout(**PLOTLY_LAYOUT_AXES, height=380, xaxis=dict(visible=True, tickangle=-45, gridcolor=GRID_C),
                       yaxis=dict(visible=True, gridcolor=GRID_C, title="Capital Destroyed"))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ---------- TAB 4: SYSTEMIC RISK RANKING ----------
with tabs[3]:
    st.markdown('<div class="section-header">🏆 Composite Systemic Risk Ranking (Current Run)</div>', unsafe_allow_html=True)
    st.caption("Systemic_Score = 40% eigenvector centrality + 30% asset size + 30% capital destroyed on default (percentile-ranked). TBTF_Index = (defaults caused / total banks) × (capital destroyed / system capital).")
    disp = comparison_df[["Systemic_Rank", "Bank", "Assets", "Capital", "Eigenvector", "Betweenness", "Defaults_Caused", "Capital_Destroyed", "Systemic_Score", "TBTF_Index"]].copy()
    for c in ["Assets", "Capital", "Capital_Destroyed"]: disp[c] = disp[c].map(lambda v: f"{v:,.0f}")
    for c in ["Eigenvector", "Betweenness", "Systemic_Score", "TBTF_Index"]: disp[c] = disp[c].map(lambda v: f"{v:.4f}")
    st.dataframe(disp, use_container_width=True, hide_index=True, height=450)

    st.markdown(f"""
    <div class="quiz-box">
    <b>📌 Key Finding</b><br>
    <b>{top_bank}</b> is the most systemically important bank in this network run — highest composite score
    combining centrality, size, and actual simulated contagion impact.
    </div>
    """, unsafe_allow_html=True)

    if st.button("💾 Save this run to database", use_container_width=True):
        save_run_to_db(comparison_df, seed, num_banks, recovery_rate)
        st.success(f"Saved {len(comparison_df)} rows (seed={seed}) to `{DB_PATH}`. Check the SQL Database Explorer tab.")

# ---------- TAB 5: SQL DATABASE EXPLORER ----------
with tabs[4]:
    st.markdown('<div class="section-header">🗄️ SQL Database Explorer</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sql-note">
    Every simulation run can be persisted to a real SQLite database (<code>systemic_risk_results.db</code>).
    This lets you ask questions a single run can't answer — e.g. "which bank is systemically important
    <b>across many different random network structures</b>, not just one lucky/unlucky seed."
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Run a Multi-Seed Analysis")
    c1, c2 = st.columns([1, 3])
    with c1:
        n_seeds = st.number_input("Number of random seeds to test", min_value=2, max_value=50, value=10)
        if st.button("▶ Run & Store", use_container_width=True):
            with st.spinner(f"Running {n_seeds} simulations and writing to SQLite..."):
                seeds_to_run = list(range(1, int(n_seeds) + 1))
                multi_seed_analysis(seeds_to_run, num_banks, connection_probability, max_exposure_fraction, recovery_rate)
            st.success(f"Stored results for {n_seeds} network seeds ({n_seeds * num_banks} rows).")
            st.rerun()

    st.markdown("#### Query the Database")
    example_queries = {
        "Banks that appear in Top-3 most often across all seeds": """SELECT bank, COUNT(*) AS times_in_top3, ROUND(AVG(systemic_rank), 1) AS avg_rank
FROM simulation_runs
WHERE systemic_rank <= 3
GROUP BY bank
ORDER BY times_in_top3 DESC;""",
        "Average capital destroyed per bank across all runs": """SELECT bank, ROUND(AVG(capital_destroyed), 1) AS avg_capital_destroyed, COUNT(*) AS num_runs
FROM simulation_runs
GROUP BY bank
ORDER BY avg_capital_destroyed DESC
LIMIT 10;""",
        "Correlation check: does higher assets always mean higher systemic score?": """SELECT bank, ROUND(AVG(assets),0) AS avg_assets, ROUND(AVG(systemic_score),4) AS avg_score
FROM simulation_runs
GROUP BY bank
ORDER BY avg_assets DESC
LIMIT 10;""",
        "All stored runs (raw)": "SELECT * FROM simulation_runs ORDER BY run_timestamp DESC LIMIT 100;",
    }
    choice = st.selectbox("Load an example query (or write your own below)", ["-- custom --"] + list(example_queries.keys()))
    default_sql = example_queries.get(choice, "SELECT * FROM simulation_runs LIMIT 20;")
    sql_input = st.text_area("SQL query", value=default_sql, height=140)

    if st.button("▶ Run Query", use_container_width=True):
        try:
            conn = get_connection()
            result = pd.read_sql_query(sql_input, conn)
            conn.close()
            st.dataframe(result, use_container_width=True, hide_index=True)
            st.caption(f"{len(result)} row(s) returned.")
            if len(result) > 0 and "bank" in [c.lower() for c in result.columns]:
                numeric_cols = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
                if numeric_cols:
                    fig = go.Figure(go.Bar(x=result.iloc[:, 0], y=result[numeric_cols[0]], marker_color=COLORS["neutral"]))
                    fig.update_layout(**PLOTLY_LAYOUT_AXES, height=320, xaxis=dict(visible=True, tickangle=-45, gridcolor=GRID_C), yaxis=dict(visible=True, gridcolor=GRID_C))
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"SQL error: {e}")

# ---------- TAB 6: QUIZ / GAMIFICATION ----------
with tabs[5]:
    st.markdown('<div class="section-header">🎮 Can You Spot the Systemically Important Bank?</div>', unsafe_allow_html=True)
    st.caption("Look at the network below, then guess which bank you think would cause the most damage if it defaulted.")
    st.plotly_chart(plot_network(size_by="Assets"), use_container_width=True)

    guess = st.selectbox("Your guess:", sorted(banks_df["Bank"], key=lambda x: int(x.split("_")[1])), key="quiz_guess")
    if st.button("Check my answer", use_container_width=True):
        actual_rank = comparison_df.reset_index(drop=True)
        guess_rank = actual_rank[actual_rank["Bank"] == guess].index[0] + 1
        if guess == top_bank:
            st.success(f"🎯 Correct! {guess} is indeed the #1 most systemically important bank.")
        elif guess_rank <= 3:
            st.warning(f"Close! {guess} ranks #{guess_rank} out of {num_banks} — top 3, but {top_bank} is actually #1.")
        else:
            st.error(f"{guess} ranks #{guess_rank} out of {num_banks}. The actual most systemic bank is {top_bank}. Hint: look for high asset size AND high connectivity, not just size alone.")

st.write("---")
st.caption("Self-initiated project · Synthetic data · Built with Python, NetworkX, Plotly, Streamlit, SQLite")
