# =====================================================================
#  ЛАБ. 15. СРАВНЕНИЕ РАСПРЕДЕЛЕНИЙ: КРИТЕРИЙ КОЛМОГОРОВА-СМИРНОВА — Positron (аналог Statistica)
#  Файлы с данными лежат в рабочей папке:
#    MSA_15_..._Kolmogorov_Smirnov_primer.xlsx, MSA_15_..._Kolmogorov_Smirnov_zadaniia.xlsx
#  Нужно: pip install pandas numpy scipy matplotlib openpyxl
#  Графики сохраняются в папку graphs/
# =====================================================================

# %% Пример (Statistica): две выборки прочности — материал A (40 образцов) и материал B (34 образца)
# ЗАДАНИЕ: Определить, одинаково ли распределены прочности материалов A и B: сравнить диапазоны и гистограммы,
#   затем применить двухвыборочный критерий Колмогорова-Смирнова.
# Statistica: Nonparametrics - Comparing two independent samples (groups) - Kolmogorov-Smirnov two-sample test
#   (Dependent: Strength; Indep. (grouping): Material); Categorized histograms by group
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st
os.makedirs("graphs", exist_ok=True); plt.close("all")

d = pd.read_excel("MSA_15_Sravnenie_raspredelenii_Kolmogorov_Smirnov_primer.xlsx", usecols=[0, 1])
a, b = d.iloc[:, 0].dropna().values, d.iloc[:, 1].dropna().values
print(f"A: n = {len(a)}, диапазон {a.min():.2f} ... {a.max():.2f}")
print(f"B: n = {len(b)}, диапазон {b.min():.2f} ... {b.max():.2f}")

D, p = st.ks_2samp(a, b)
crit = 1.36 * np.sqrt((len(a) + len(b)) / (len(a) * len(b)))               # критическое значение для α = 0,05
print(f"D (λ) = {D:.4f}, критическое = {crit:.4f}, p = {p:.4f} ->",
      "распределения РАЗЛИЧАЮТСЯ" if p < .05 else "различие незначимо: распределения одинаковы")

edges = np.arange(4, 18, 2)
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for a_, v, t in zip(ax, [a, b], ["Материал A", "Материал B"]):
    a_.hist(v, edges, edgecolor="k"); a_.set_title(t); a_.set_xlabel("Прочность")
fig.savefig("graphs/lab15_example_hist.png", dpi=150, bbox_inches="tight"); plt.show()

fig, ax = plt.subplots()                                                   # эмпирические функции распределения
for v, t in [(a, "A"), (b, "B")]: ax.step(np.sort(v), np.arange(1, len(v) + 1) / len(v), where="post", label=t)
ax.legend(); ax.set_title("Эмпирические функции распределения")
fig.savefig("graphs/lab15_example_ecdf.png", dpi=150, bbox_inches="tight"); plt.show()

# %% Задание (Statistica): две выборки своего варианта
# ЗАДАНИЕ: Проверить две выборки (согласно номеру в списке группы) на равенство закона распределения по критерию
#   Колмогорова-Смирнова. Если различие значимо — проверить, значимо ли различаются средние
#   (критерий Стьюдента; F-критерий дисперсий определяет вариант критерия).
#   Файл «zadaniia»: листы 1-6, в каждом две выборки (столбцы A и B).
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st
os.makedirs("graphs", exist_ok=True); plt.close("all")

VARIANT = 1                                                                # <-- ВАШ НОМЕР ВАРИАНТА (лист 1-6)
d = pd.read_excel("MSA_15_Sravnenie_raspredelenii_Kolmogorov_Smirnov_zadaniia.xlsx", sheet_name=str(VARIANT), header=None)
a, b = d.iloc[:, 0].dropna().values, d.iloc[:, 1].dropna().values
print(f"Вариант {VARIANT}: A: n = {len(a)}, {a.min():.2f}...{a.max():.2f}; B: n = {len(b)}, {b.min():.2f}...{b.max():.2f}")

D, p = st.ks_2samp(a, b)
print(f"D = {D:.4f}, p = {p:.4f} ->", "распределения РАЗЛИЧАЮТСЯ" if p < .05 else "различие незначимо")

edges = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 9)
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for a_, v, t in zip(ax, [a, b], ["Выборка A", "Выборка B"]):
    a_.hist(v, edges, edgecolor="k"); a_.set_title(t)
fig.savefig(f"graphs/lab15_task_variant{VARIANT}_hist.png", dpi=150, bbox_inches="tight"); plt.show()

if p < .05:                                                                # сравнение средних (Лаб. 2)
    F = a.var(ddof=1) / b.var(ddof=1); d1, d2 = len(a) - 1, len(b) - 1
    p_var = 2 * min(st.f.cdf(F, d1, d2), st.f.sf(F, d1, d2)); equal = p_var > .05
    t, pt = st.ttest_ind(a, b, equal_var=equal)
    print(f"F-критерий дисперсий: p = {p_var:.4f} -> дисперсии {'равны' if equal else 'различаются'}")
    print(f"t-критерий ({'Стьюдента' if equal else 'Уэлча'}): t = {t:.3f}, p = {pt:.4f} ->",
          "средние различаются ЗНАЧИМО" if pt < .05 else "средние различаются незначимо")
