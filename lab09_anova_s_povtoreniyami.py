# =====================================================================
#  ЛАБ. 9. МНОГОФАКТОРНЫЙ ДИСПЕРСИОННЫЙ АНАЛИЗ С ПОВТОРЕНИЯМИ — Positron (аналог Statistica)
#  Нужно: pip install pandas numpy scipy statsmodels matplotlib
#  Графики сохраняются в папку graphs/
# =====================================================================

# %% Пример (Statistica): зарплата — форма собственности (2) и размер предприятия (3)
# ЗАДАНИЕ: Выяснить, какие факторы (форма собственности, размер) или их комбинации (взаимодействие)
#   значимо влияют на зарплату; выявить значимо различающиеся группы (Fisher LSD).
# Statistica: ANOVA - Factorial ANOVA (Categorical factors: Forma, Razmer; Dependent: Zarplata);
#   All effects; All effects/Graphs (в т.ч. Razmer*Forma); Post-hoc - Fisher LSD для Size*Ownership
import os
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as st
os.makedirs("graphs", exist_ok=True); plt.close("all")

cells = {("Госуд.", "Крупные"): [180, 200, 215, 260], ("Госуд.", "Средние"): [215, 180, 130, 200],
         ("Госуд.", "Мелкие"): [200, 160, 130, 170],  ("Негосуд.", "Крупные"): [180, 190, 180, 250],
         ("Негосуд.", "Средние"): [240, 230, 190, 220], ("Негосуд.", "Мелкие"): [240, 260, 200, 250]}
d = pd.DataFrame([(f, r, z) for (f, r), v in cells.items() for z in v], columns=["Forma", "Razmer", "Zarplata"])
assert len(d) == 24

m = smf.ols("Zarplata ~ C(Forma, Sum) * C(Razmer, Sum)", d).fit()     # полная модель с взаимодействием
an = sm.stats.anova_lm(m, typ=3).drop("Intercept"); an["SS %"] = 100 * an.sum_sq / an.sum_sq.sum()
print(an.round(4))
for e in an.index[:-1]:
    print(f"{e}: p = {an.loc[e, 'PR(>F)']:.4f} ->", "ЗНАЧИМ" if an.loc[e, "PR(>F)"] < .05 else "незначим")

# графики средних (Razmer * Forma) — для анализа взаимодействия
mt = d.groupby(["Razmer", "Forma"]).Zarplata.mean().unstack()[["Госуд.", "Негосуд."]].loc[["Крупные", "Средние", "Мелкие"]]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
mt.plot(marker="o", ax=ax[0], title="Зарплата по размеру (линии — форма собственности)")
mt.T.plot(marker="o", ax=ax[1], title="Зарплата по форме (линии — размер)")
fig.savefig("graphs/lab09_example_interaction.png", dpi=150, bbox_inches="tight"); plt.show()

# Fisher LSD по всем 6 группам (Forma x Razmer)
mse, df_err = m.mse_resid, m.df_resid
print(f"\nFisher LSD (MSE = {mse:.2f}, df = {df_err:.0f}), значимые пары:")
for (a, va), (b, vb) in combinations(cells.items(), 2):
    diff = np.mean(va) - np.mean(vb); pv = 2 * st.t.sf(abs(diff) / np.sqrt(mse * (1/len(va) + 1/len(vb))), df_err)
    if pv < .05: print(f"  {a} - {b}: разность = {diff:7.2f}, p = {pv:.4f}")

# %% Задание 1 (Statistica): зарплата — отрасль (2) и регион (4), по 3 предприятия в группе
# ЗАДАНИЕ: Определить, зависит ли зарплата от отрасли и региона (и их взаимодействия). Если различие значимо —
#   определить отрасли и регионы, для которых зарплата различается значимо.
import os
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as st
os.makedirs("graphs", exist_ok=True); plt.close("all")

cells = {("Металлургическая", "Север"): [420, 370, 520], ("Металлургическая", "Юг"): [370, 480, 390],
         ("Металлургическая", "Восток"): [480, 550, 520], ("Металлургическая", "Запад"): [480, 390, 510],
         ("Химическая", "Север"): [620, 540, 580],        ("Химическая", "Юг"): [620, 540, 580],
         ("Химическая", "Восток"): [390, 480, 490],       ("Химическая", "Запад"): [490, 570, 520]}
d = pd.DataFrame([(o, r, z) for (o, r), v in cells.items() for z in v], columns=["Otrasl", "Region", "Zarplata"])
assert len(d) == 24

m = smf.ols("Zarplata ~ C(Otrasl, Sum) * C(Region, Sum)", d).fit()
an = sm.stats.anova_lm(m, typ=3).drop("Intercept"); an["SS %"] = 100 * an.sum_sq / an.sum_sq.sum()
print(an.round(4))
for e in an.index[:-1]:
    print(f"{e}: p = {an.loc[e, 'PR(>F)']:.4f} ->", "ЗНАЧИМ" if an.loc[e, "PR(>F)"] < .05 else "незначим")

mt = d.groupby(["Region", "Otrasl"]).Zarplata.mean().unstack().loc[["Север", "Юг", "Восток", "Запад"]]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
mt.plot(marker="o", ax=ax[0], title="Зарплата по регионам (линии — отрасль)")
mt.T.plot(marker="o", ax=ax[1], title="Зарплата по отраслям (линии — регион)")
fig.savefig("graphs/lab09_task1_interaction.png", dpi=150, bbox_inches="tight"); plt.show()

mse, df_err = m.mse_resid, m.df_resid
print(f"\nFisher LSD по 8 группам (MSE = {mse:.1f}, df = {df_err:.0f}), значимые пары:")
for (a, va), (b, vb) in combinations(cells.items(), 2):
    diff = np.mean(va) - np.mean(vb); pv = 2 * st.t.sf(abs(diff) / np.sqrt(mse * (1/len(va) + 1/len(vb))), df_err)
    if pv < .05: print(f"  {a} - {b}: разность = {diff:7.2f}, p = {pv:.4f}")

# %% Задание 2 (Statistica): зарплата — отрасль (2), регион (4), размер (2); 40 предприятий (группы неравны)
# ЗАДАНИЕ: Выполнить анализ данных методами дисперсионного анализа (трёхфакторный анализ с повторениями):
#   определить значимые факторы и их взаимодействия.
# Statistica: ANOVA - Factorial ANOVA (три Categorical factors). При неравных группах — суммы квадратов типа III.
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
os.makedirs("graphs", exist_ok=True); plt.close("all")

reg = ["Север", "Север", "Юг", "Юг", "Восток", "Восток", "Запад", "Запад"]          # столбцы таблицы
siz = ["Крупн.", "Мелк."] * 4
met = [[420, 370], [320, 400, 310], [370, 480, 390], [300, 360], [480, 550], [310, 400, 320], [480, 390, 510], [410, 370, 320]]
him = [[620, 540], [480, 570], [620, 540, 580], [560, 520], [590, 480], [580, 510, 480], [490, 570, 520], [530, 550]]
d = pd.DataFrame([(o, r, s, z) for o, tab in [("Металлургическая", met), ("Химическая", him)]
                  for r, s, v in zip(reg, siz, tab) for z in v], columns=["Otrasl", "Region", "Razmer", "Zarplata"])
print("N =", len(d)); assert len(d) == 40

m = smf.ols("Zarplata ~ C(Otrasl, Sum) * C(Region, Sum) * C(Razmer, Sum)", d).fit()
an = sm.stats.anova_lm(m, typ=3).drop("Intercept"); print(an.round(4))
for e in an.index[:-1]:
    if an.loc[e, "PR(>F)"] < .05: print(f"ЗНАЧИМО: {e} (p = {an.loc[e, 'PR(>F)']:.4f})")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, f in zip(ax, ["Otrasl", "Region", "Razmer"]):
    mean = d.groupby(f).Zarplata.mean(); a.plot(mean.index, mean.values, "o-"); a.set_title(f"Зарплата: {f}")
fig.savefig("graphs/lab09_task2_means.png", dpi=150, bbox_inches="tight"); plt.show()
