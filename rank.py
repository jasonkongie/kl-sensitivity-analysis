import json
import pandas as pd
from scipy.stats import kendalltau

PATH = "sensitivity_results_mamba2-130m_4bits.json"

with open(PATH) as f:
    metrics = json.load(f)

df = pd.DataFrame.from_dict(metrics, orient='index')

df = df.drop(index="lm_head", errors="ignore")

orders = {
    'sqnr_db': True,
    'kl_teacher_to_student': False,
    'kl_student_to_teacher': False,
    'delta_cross_entropy': False
}

ppl_rank = df['perplexity'].rank(ascending=False)

for metric, asc in orders.items():
    m_rank = df[metric].rank(ascending=asc)
    tau, p = kendalltau(m_rank, ppl_rank)
    print(f"{metric}: Kendall τ={tau:.4f}, p-value={p:.4g}")
