# Systemic Risk Network Visualizer

Self-initiated project simulating interbank lending networks and analyzing
systemic risk using network science.

## Features
- Interactive interbank network graph (NetworkX + Plotly)
- Contagion cascade simulation (bank default → creditor capital loss → further defaults)
- Composite Systemic Risk Score (eigenvector centrality + asset size + simulated impact)
- Too-Big-To-Fail (TBTF) Index
- Multi-seed stress testing persisted to SQLite, queryable via SQL
- Built with: Python, NetworkX, Plotly, Streamlit, SQLite

## Run locally
```
pip install -r requirements.txt
streamlit run systemic_risk_dashboard.py
```
