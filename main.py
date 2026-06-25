from steps.step1_load import load_data
from steps.step2_eda import run_eda

# ── Run the full pipeline ──────────────────────────────────────

df = load_data('data/creditcard.csv')

run_eda(df)