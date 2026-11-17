# =====================================================================
#  ЛАБ. 13. КРИТЕРИЙ «ХИ-КВАДРАТ»: НЕПРЕРЫВНЫЕ ВЕЛИЧИНЫ — Positron (аналог Statistica)
#  Файлы с данными (из задания) должны лежать в рабочей папке проекта:
#    MSA_13_..._normalnoe_primer.xlsx, ..._normalnoe_zadaniia.xlsx, ..._raznye_zadaniia.xlsx
#  Нужно: pip install pandas numpy scipy matplotlib openpyxl
#  Графики сохраняются в папку graphs/
# =====================================================================

# %% Пример 1 (Statistica): проверка выборки из 100 значений на нормальное распределение
# ЗАДАНИЕ: Определить, можно ли считать, что выборка соответствует нормальному закону распределения
#   (визуальная проверка по гистограмме + критерий «хи-квадрат»).
# Statistica: Statistics - Distribution Fittings - Normal; Variables; Plot of observed and expected
#   distribution (гистограмма + кривая, Chi-Square Test, df, p); Summary
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st
os.makedirs("graphs", exist_ok=True); plt.close("all")

x = pd.read_excel("MSA_13_khi_kvadrat_proverka_raspredelenii_normalnoe_primer.xlsx", usecols=[0]).iloc[:, 0].dropna().values
n, mu, s = len(x), x.mean(), x.std(ddof=1)
print(f"n = {n}, min = {x.min():.2f}, max = {x.max():.2f}, среднее = {mu:.3f}, SD = {s:.3f}")

edges = np.arange(15, 19.6, 0.5)                                          # границы интервалов (шаг 0,5)
full = np.r_[-np.inf, edges, np.inf]                                      # + (-inf; 15) и (19,5; +inf)
obs = pd.Series(pd.cut(x, full)).value_counts(sort=False).values          # фактические частоты
exp = n * np.diff(st.norm.cdf(full, mu, s))                               # теоретические частоты

o2, e2, co, ce = [], [], 0, 0                                             # объединяем интервалы, пока теор. частота < 5
for o, e in zip(obs, exp):
    co, ce = co + o, ce + e
    if ce >= 5: o2.append(co); e2.append(ce); co = ce = 0
if ce > 0: o2[-1] += co; e2[-1] += ce
o2, e2 = np.array(o2), np.array(e2)

chi2 = ((o2 - e2)**2 / e2).sum(); k, r = len(o2), 2                       # r = 2 параметра (m, sigma)
p = st.chi2.sf(chi2, k - r - 1)
print(f"k = {k} интервалов, χ² = {chi2:.3f}, df = {k-r-1}, p = {p:.4f} ->",
      "гипотеза о нормальности НЕ отвергается" if p > .05 else "выборка НЕ нормальна")

fig, ax = plt.subplots(); ax.hist(x, edges, density=True, edgecolor="k", alpha=.6)
xs = np.linspace(x.min() - 1, x.max() + 1, 200); ax.plot(xs, st.norm.pdf(xs, mu, s), "r")
ax.set_title(f"Пример 1: гистограмма и нормальная кривая (p = {p:.3f})")
fig.savefig("graphs/lab13_example_normal.png", dpi=150, bbox_inches="tight"); plt.show()

# %% Задание 1 (Statistica): проверка выборки своего варианта на нормальное распределение
# ЗАДАНИЕ: Проверить выборку согласно номеру варианта на соответствие нормальному закону по критерию
#   «хи-квадрат» (включая визуальную проверку). Файл «normalnoe_zadaniia»: варианты 1-9 = столбцы листа «Лист2».
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st
os.makedirs("graphs", exist_ok=True); plt.close("all")

VARIANT = 1                                                                # <-- ВАШ НОМЕР ВАРИАНТА (1-9)
x = pd.read_excel("MSA_13_khi_kvadrat_proverka_raspredelenii_normalnoe_zadaniia.xlsx",
                  sheet_name="Лист2", header=None).iloc[:, VARIANT - 1].dropna().values
n, mu, s = len(x), x.mean(), x.std(ddof=1)
print(f"Вариант {VARIANT}: n = {n}, min = {x.min():.2f}, max = {x.max():.2f}, среднее = {mu:.3f}, SD = {s:.3f}")

edges = np.linspace(np.floor(x.min()), np.ceil(x.max()), 11)               # 10 интервалов (гистограмма)
full = np.r_[-np.inf, edges, np.inf]
obs = pd.Series(pd.cut(x, full)).value_counts(sort=False).values
exp = n * np.diff(st.norm.cdf(full, mu, s))

o2, e2, co, ce = [], [], 0, 0
for o, e in zip(obs, exp):
    co, ce = co + o, ce + e
    if ce >= 5: o2.append(co); e2.append(ce); co = ce = 0
if ce > 0: o2[-1] += co; e2[-1] += ce
o2, e2 = np.array(o2), np.array(e2)

chi2 = ((o2 - e2)**2 / e2).sum(); k, r = len(o2), 2
p = st.chi2.sf(chi2, k - r - 1)
print(f"k = {k}, χ² = {chi2:.3f}, df = {k-r-1}, p = {p:.4f} ->",
      "гипотеза о нормальности НЕ отвергается" if p > .05 else "выборка НЕ нормальна")

fig, ax = plt.subplots(); ax.hist(x, edges, density=True, edgecolor="k", alpha=.6)
xs = np.linspace(x.min(), x.max(), 200); ax.plot(xs, st.norm.pdf(xs, mu, s), "r")
ax.set_title(f"Вариант {VARIANT}: гистограмма и нормальная кривая (p = {p:.3f})")
fig.savefig(f"graphs/lab13_task1_variant{VARIANT}.png", dpi=150, bbox_inches="tight"); plt.show()

# %% Задание 2 (Statistica): выбор закона распределения по гистограмме и проверка
# ЗАДАНИЕ: Проверить выборку своего варианта на соответствие закону распределения, выбранному по гистограмме
#   (нормальное / экспоненциальное / равномерное). Файл «raznye_zadaniia»: листы 1-6, в каждом два столбца
#   (варианты 1-12: лист = (вариант+1)//2, столбец = 1 или 2).
# Statistica: Distribution Fittings - Normal / Exponential / Rectangular (Parameters: Number of categories, Lower/Upper limit)
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st
os.makedirs("graphs", exist_ok=True); plt.close("all")

VARIANT = 1                                                                # <-- ВАШ НОМЕР ВАРИАНТА (1-12)
x = pd.read_excel("MSA_13_khi_kvadrat_proverka_raspredelenii_raznye_zadaniia.xlsx",
                  sheet_name=str((VARIANT + 1) // 2), header=None).iloc[:, (VARIANT - 1) % 2].dropna().values
n = len(x); print(f"Вариант {VARIANT}: n = {n}, min = {x.min():.2f}, max = {x.max():.2f}, среднее = {x.mean():.2f}")

edges = np.linspace(x.min(), x.max(), 11)                                  # 10 интервалов
full = np.r_[-np.inf, edges[1:-1], np.inf]                                 # (-inf; e1], ..., (e9; +inf)
obs = pd.Series(pd.cut(x, full)).value_counts(sort=False).values

dists = {"Нормальное":      (st.norm(x.mean(), x.std(ddof=1)), 2),
         "Экспоненциальное": (st.expon(0, x.mean()), 1),
         "Равномерное":      (st.uniform(x.min(), x.max() - x.min()), 2)}
fig, ax = plt.subplots(); ax.hist(x, edges, density=True, edgecolor="k", alpha=.5); xs = np.linspace(x.min(), x.max(), 200)
for name, (dist, r) in dists.items():
    exp = n * np.diff(dist.cdf(full))
    o2, e2, co, ce = [], [], 0, 0
    for o, e in zip(obs, exp):
        co, ce = co + o, ce + e
        if ce >= 5: o2.append(co); e2.append(ce); co = ce = 0
    if ce > 0: o2[-1] += co; e2[-1] += ce
    o2, e2 = np.array(o2), np.array(e2); k = len(o2); chi2 = ((o2 - e2)**2 / e2).sum()
    p = st.chi2.sf(chi2, k - r - 1) if k - r - 1 > 0 else float("nan")
    print(f"{name:<18} k = {k:2d}, χ² = {chi2:7.2f}, df = {k-r-1}, p = {p:.4f} ->",
          "СООТВЕТСТВУЕТ" if p > .05 else "не соответствует")
    ax.plot(xs, dist.pdf(xs), label=name)
ax.legend(); ax.set_title(f"Вариант {VARIANT}: гистограмма и теоретические плотности")
fig.savefig(f"graphs/lab13_task2_variant{VARIANT}.png", dpi=150, bbox_inches="tight"); plt.show()
