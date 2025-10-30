import json
import pandas as pd
from scipy.stats import kendalltau

PATH = "" #rank
PPL_PATH = "" #perplexity_path


with open(PATH) as f:
    metrics = json.load(f)
with open(PPL_PATH) as f:
    ppl = json.load(f)

df = pd.DataFrame.from_dict(metrics, orient='index')
df['perplexity'] = pd.Series({k: v['perplexity'] for k, v in ppl.items()})

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
    print(f"{metric}: Kendall τ={tau:.4f}, p‐value={p:.4g}")% 
