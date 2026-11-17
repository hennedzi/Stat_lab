# =====================================================================
#  ЛАБ. 7. ОДНОФАКТОРНЫЙ ДИСПЕРСИОННЫЙ АНАЛИЗ — Positron (аналог Statistica)
#  Нужно: pip install pandas numpy scipy matplotlib
#  Графики сохраняются в папку graphs/
# =====================================================================

# %% Пример (Statistica): привесы животных трёх пород
# ЗАДАНИЕ: Выяснить, значимо ли различаются привесы у животных пород П1, П2, П3
#   (группирующая переменная PORODA, зависимая PRIVES), и какие именно породы различаются.
# Statistica: ANOVA - One-way ANOVA (Categorical factor: PORODA; Dependent: PRIVES);
#   All effects / Graphs; Cell statistics; Post-hoc - Fisher LSD
import os
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st
os.makedirs("graphs", exist_ok=True); plt.close("all")

d = pd.DataFrame({
    "PORODA": ["П1"]*5 + ["П2"]*6 + ["П3"]*5,
    "PRIVES": [640, 637, 648, 624, 639,  638, 625, 662, 645, 640, 637,  625, 617, 631, 629, 619]})

# а) значимость различия между группами (All effects)
g = {k: v.PRIVES.values for k, v in d.groupby("PORODA")}
F, p = st.f_oneway(*g.values())
n, m = len(d), len(g); df_err = n - m
mse = sum(((v - v.mean())**2).sum() for v in g.values()) / df_err       # MS Error (внутригрупповой разброс)
print(f"F({m-1}; {df_err}) = {F:.3f}, p = {p:.4f} ->", "различие ЗНАЧИМО" if p < .05 else "различие незначимо")

# б) статистики по группам (Cell statistics) и график средних с 95% ДИ
tbl = d.groupby("PORODA").PRIVES.agg(N="count", Mean="mean", SD="std")
tbl["ДИ ±"] = st.t.ppf(.975, df_err) * np.sqrt(mse / tbl.N)
print("\n", tbl.round(2))
fig, ax = plt.subplots()
ax.errorbar(range(m), tbl.Mean, tbl["ДИ ±"], fmt="o", capsize=8)
ax.set_xticks(range(m), tbl.index); ax.set_xlim(-.5, m-.5)
ax.set_ylabel("Привес, г"); ax.set_title("Средние по породам и 95% доверительные интервалы")
fig.savefig("graphs/lab07_example_means.png", dpi=150, bbox_inches="tight"); plt.show()

# в) Post-hoc: наименьшая существенная разность (Fisher LSD)
print("\nFisher LSD (попарные сравнения):")
for a, b in combinations(g, 2):
    diff = g[a].mean() - g[b].mean()
    se = np.sqrt(mse * (1/len(g[a]) + 1/len(g[b])))
    pv = 2 * st.t.sf(abs(diff) / se, df_err)
    print(f"  {a} - {b}: разность = {diff:6.2f}, p = {pv:.4f} ->", "ЗНАЧИМО" if pv < .05 else "незначимо")

# %% Задание (Statistica): прочность четырёх материалов
# ЗАДАНИЕ: Определить, имеется ли значимое различие материалов по прочности. Если да — определить
#   пары материалов, различающиеся значимо, а также наиболее и наименее прочный материал.
import os
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st
os.makedirs("graphs", exist_ok=True); plt.close("all")

g = {"M1": np.array([38, 35, 41, 32, 28, 36, 35]),
     "M2": np.array([29, 31, 34, 25, 28, 25]),
     "M3": np.array([37, 42, 46, 38, 35, 32, 39]),
     "M4": np.array([32, 27, 35, 29, 26])}

F, p = st.f_oneway(*g.values())
n, m = sum(map(len, g.values())), len(g); df_err = n - m
mse = sum(((v - v.mean())**2).sum() for v in g.values()) / df_err
print(f"F({m-1}; {df_err}) = {F:.3f}, p = {p:.5f} ->", "различие ЗНАЧИМО" if p < .05 else "различие незначимо")

means = {k: v.mean() for k, v in g.items()}
print("Средние:", {k: round(v, 2) for k, v in means.items()})
print("Наиболее прочный:", max(means, key=means.get), "| наименее прочный:", min(means, key=means.get))

fig, ax = plt.subplots()
ci = [st.t.ppf(.975, df_err) * np.sqrt(mse / len(v)) for v in g.values()]
ax.errorbar(range(m), list(means.values()), ci, fmt="o", capsize=8)
ax.set_xticks(range(m), list(g)); ax.set_xlim(-.5, m-.5)
ax.set_ylabel("Прочность"); ax.set_title("Средние по материалам и 95% ДИ")
fig.savefig("graphs/lab07_task_means.png", dpi=150, bbox_inches="tight"); plt.show()

print("\nFisher LSD (попарные сравнения):")
for a, b in combinations(g, 2):
    diff = g[a].mean() - g[b].mean()
    se = np.sqrt(mse * (1/len(g[a]) + 1/len(g[b])))
    pv = 2 * st.t.sf(abs(diff) / se, df_err)
    print(f"  {a} - {b}: разность = {diff:6.2f}, p = {pv:.4f} ->", "ЗНАЧИМО" if pv < .05 else "незначимо")
