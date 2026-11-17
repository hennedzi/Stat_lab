# =====================================================================
#  ЛАБ. 8. МНОГОФАКТОРНЫЙ ДИСПЕРСИОННЫЙ АНАЛИЗ БЕЗ ПОВТОРЕНИЙ — Positron (аналог Statistica)
#  Нужно: pip install pandas numpy scipy statsmodels matplotlib
#  Графики сохраняются в папку graphs/
# =====================================================================

# %% Пример (Statistica): прочность изделий — материал (5) и способ обработки (4)
# ЗАДАНИЕ: а) Определить, значимо ли различается прочность для разных материалов и способов обработки;
#   б) для значимых факторов определить уровни, существенно различающиеся по прочности.
# Statistica: ANOVA - Main effects ANOVA (Categorical factors: Material, Treatment; Dependent: Strength);
#   All effects; All effects/Graphs; Post-hoc - Fisher LSD для каждого фактора
import os
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as st
os.makedirs("graphs", exist_ok=True); plt.close("all")

tab = pd.DataFrame([[35.4, 37.1, 36.6, 35.8], [39.6, 40.4, 38.3, 39.5], [35.6, 34.8, 35.8, 36.5],
                    [41.6, 39.9, 39.3, 45.6], [46.2, 45.4, 43.8, 49.9]],
                   index=["M1", "M2", "M3", "M4", "M5"], columns=["T1", "T2", "T3", "T4"])
d = tab.stack().rename_axis(["Material", "Treatment"]).rename("Strength").reset_index()   # 20 строк

m = smf.ols("Strength ~ C(Material) + C(Treatment)", d).fit()          # модель главных эффектов
an = sm.stats.anova_lm(m, typ=2); print(an.round(4))
mse, df_err = m.mse_resid, m.df_resid
for f in ["Material", "Treatment"]:
    p = an.loc[f"C({f})", "PR(>F)"]
    print(f"Фактор {f}: p = {p:.4g} ->", "ЗНАЧИМ" if p < .05 else "незначим")

for f in ["Material", "Treatment"]:                                      # средние и Fisher LSD по каждому фактору
    mean = d.groupby(f).Strength.mean(); n = d.groupby(f).size()
    fig, ax = plt.subplots(); ci = st.t.ppf(.975, df_err) * np.sqrt(mse / n)
    ax.errorbar(range(len(mean)), mean, ci, fmt="o", capsize=8); ax.set_xticks(range(len(mean)), mean.index)
    ax.set_title(f"Средняя прочность по фактору {f} (95% ДИ)"); ax.set_ylabel("Прочность")
    fig.savefig(f"graphs/lab08_example_{f}.png", dpi=150, bbox_inches="tight"); plt.show()
    print(f"\nFisher LSD для {f} (только значимые пары):")
    for a, b in combinations(mean.index, 2):
        pv = 2 * st.t.sf(abs(mean[a] - mean[b]) / np.sqrt(mse * (1/n[a] + 1/n[b])), df_err)
        if pv < .05: print(f"  {a} - {b}: разность = {mean[a]-mean[b]:6.2f}, p = {pv:.4f}")

# %% Задание 1 (Statistica): зарплата — отрасль (3) и регион (4), по одному предприятию
# ЗАДАНИЕ: Определить, зависит ли зарплата от отрасли и региона. Если различие значимо — определить
#   отрасли и регионы, для которых зарплата различается значимо.
import os
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as st
os.makedirs("graphs", exist_ok=True); plt.close("all")

tab = pd.DataFrame([[420, 370, 400, 480], [620, 530, 510, 490], [680, 510, 630, 520]],
                   index=["Металлургическая", "Химическая", "Нефтяная"],
                   columns=["Север", "Юг", "Восток", "Запад"])
d = tab.stack().rename_axis(["Otrasl", "Region"]).rename("Zarplata").reset_index()      # 12 строк

m = smf.ols("Zarplata ~ C(Otrasl) + C(Region)", d).fit()
an = sm.stats.anova_lm(m, typ=2); print(an.round(4))
mse, df_err = m.mse_resid, m.df_resid
for f in ["Otrasl", "Region"]:
    p = an.loc[f"C({f})", "PR(>F)"]
    print(f"Фактор {f}: p = {p:.4g} ->", "ЗНАЧИМ" if p < .05 else "незначим")

for f in ["Otrasl", "Region"]:
    mean = d.groupby(f).Zarplata.mean(); n = d.groupby(f).size()
    fig, ax = plt.subplots(); ci = st.t.ppf(.975, df_err) * np.sqrt(mse / n)
    ax.errorbar(range(len(mean)), mean, ci, fmt="o", capsize=8); ax.set_xticks(range(len(mean)), mean.index)
    ax.set_title(f"Средняя зарплата по фактору {f} (95% ДИ)"); ax.set_ylabel("Зарплата")
    fig.savefig(f"graphs/lab08_task1_{f}.png", dpi=150, bbox_inches="tight"); plt.show()
    print(f"\nFisher LSD для {f}:")
    for a, b in combinations(mean.index, 2):
        pv = 2 * st.t.sf(abs(mean[a] - mean[b]) / np.sqrt(mse * (1/n[a] + 1/n[b])), df_err)
        print(f"  {a} - {b}: разность = {mean[a]-mean[b]:7.2f}, p = {pv:.4f} ->", "ЗНАЧИМО" if pv < .05 else "незначимо")

# %% Задание 2 (Statistica): трёхфакторный анализ без повторений — отрасль, регион, форма собственности
# ЗАДАНИЕ: Решить задачу для 24 предприятий (12 государственных из задания 1 и 12 негосударственных):
#   определить, значимо ли влияют на зарплату отрасль, регион и форма собственности.
# Statistica: ANOVA - Main effects ANOVA (три Categorical factors)
import os
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as st
os.makedirs("graphs", exist_ok=True); plt.close("all")

cols, idx = ["Север", "Юг", "Восток", "Запад"], ["Металлургическая", "Химическая", "Нефтяная"]
gos = pd.DataFrame([[420, 370, 400, 480], [620, 530, 510, 490], [680, 510, 630, 520]], idx, cols)
neg = pd.DataFrame([[515, 420, 430, 460], [610, 570, 520, 480], [690, 480, 670, 540]], idx, cols)
d = pd.concat([gos.stack().rename("Zarplata").reset_index().assign(Forma="Госуд."),
               neg.stack().rename("Zarplata").reset_index().assign(Forma="Негосуд.")])
d.columns = ["Otrasl", "Region", "Zarplata", "Forma"]; assert len(d) == 24

m = smf.ols("Zarplata ~ C(Otrasl) + C(Region) + C(Forma)", d).fit()     # только главные эффекты
an = sm.stats.anova_lm(m, typ=2); print(an.round(4))
mse, df_err = m.mse_resid, m.df_resid
for f in ["Otrasl", "Region", "Forma"]:
    p = an.loc[f"C({f})", "PR(>F)"]
    print(f"Фактор {f}: p = {p:.4g} ->", "ЗНАЧИМ" if p < .05 else "незначим")

for f in ["Otrasl", "Region", "Forma"]:
    mean = d.groupby(f).Zarplata.mean(); n = d.groupby(f).size()
    fig, ax = plt.subplots(); ci = st.t.ppf(.975, df_err) * np.sqrt(mse / n)
    ax.errorbar(range(len(mean)), mean, ci, fmt="o", capsize=8); ax.set_xticks(range(len(mean)), mean.index)
    ax.set_title(f"Средняя зарплата по фактору {f} (95% ДИ)"); ax.set_ylabel("Зарплата")
    fig.savefig(f"graphs/lab08_task2_{f}.png", dpi=150, bbox_inches="tight"); plt.show()
    print(f"\nFisher LSD для {f}:")
    for a, b in combinations(mean.index, 2):
        pv = 2 * st.t.sf(abs(mean[a] - mean[b]) / np.sqrt(mse * (1/n[a] + 1/n[b])), df_err)
        print(f"  {a} - {b}: разность = {mean[a]-mean[b]:7.2f}, p = {pv:.4f} ->", "ЗНАЧИМО" if pv < .05 else "незначимо")
