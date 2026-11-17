# =====================================================================
#  ЛАБ. 14. КРИТЕРИЙ «ХИ-КВАДРАТ»: ДИСКРЕТНЫЕ ВЕЛИЧИНЫ (ПУАССОН) — Positron (аналог Statistica)
#  Файлы с данными лежат в рабочей папке: MSA_14_..._puasson_primer.xlsx, MSA_14_..._puasson_zadaniia.xlsx
#  Нужно: pip install pandas numpy scipy matplotlib openpyxl
#  Графики сохраняются в папку graphs/
# =====================================================================

# %% Пример 1 (Statistica): поток запросов к базе данных — пуассоновский ли он?
# ЗАДАНИЕ: Определить, соответствует ли количество запросов в единицу времени (80 периодов) закону Пуассона.
# Statistica: Statistics - Distribution Fittings - Discrete - Poisson; Parameters: Number of categories = 16,
#   Lower limit = 1, Upper limit = 17; Summary
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st
os.makedirs("graphs", exist_ok=True); plt.close("all")

x = pd.read_excel("MSA_14_khi_kvadrat_proverka_raspredelenii_puasson_primer.xlsx", sheet_name="Poisson",
                  header=None, usecols=[0]).iloc[:, 0].dropna().astype(int).values
n, lam = len(x), x.mean(); print(f"n = {n}, λ = среднее = {lam:.3f}, min = {x.min()}, max = {x.max()}")

vals = np.arange(0, x.max() + 1)                                          # значения 0..max, плюс хвост «> max»
obs = np.r_[np.bincount(x, minlength=len(vals)), 0]
exp = n * np.r_[st.poisson.pmf(vals, lam), st.poisson.sf(x.max(), lam)]
o2, e2, co, ce = [], [], 0, 0                                             # объединение значений, пока теор. частота < 5
for o, e in zip(obs, exp):
    co, ce = co + o, ce + e
    if ce >= 5: o2.append(co); e2.append(ce); co = ce = 0
if ce > 0: o2[-1] += co; e2[-1] += ce
o2, e2 = np.array(o2), np.array(e2)

chi2 = ((o2 - e2)**2 / e2).sum(); k, r = len(o2), 1                       # r = 1 параметр (λ)
p = st.chi2.sf(chi2, k - r - 1)
print(f"k = {k}, χ² = {chi2:.3f}, df = {k-r-1}, p = {p:.4f} ->",
      "гипотеза о пуассоновском потоке НЕ отвергается" if p > .05 else "поток НЕ пуассоновский")

fig, ax = plt.subplots(); w = .4
ax.bar(vals - w/2, obs[:-1], w, label="Фактические"); ax.bar(vals + w/2, exp[:-1], w, label="Теоретические (Пуассон)")
ax.set_xlabel("Число запросов"); ax.legend(); ax.set_title(f"Пример 1: χ² = {chi2:.2f}, p = {p:.3f}")
fig.savefig("graphs/lab14_example_poisson.png", dpi=150, bbox_inches="tight"); plt.show()

# %% Задание (Statistica): проверка выборки своего варианта на закон Пуассона
# ЗАДАНИЕ: Проверить выборку согласно номеру варианта на соответствие пуассоновскому закону по критерию
#   «хи-квадрат». Файл «puasson_zadaniia»: варианты 1-6 = столбцы листа «Лист1» (по 70 значений).
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st
os.makedirs("graphs", exist_ok=True); plt.close("all")

VARIANT = 1                                                                # <-- ВАШ НОМЕР ВАРИАНТА (1-6)
x = pd.read_excel("MSA_14_khi_kvadrat_proverka_raspredelenii_puasson_zadaniia.xlsx", sheet_name="Лист1",
                  header=None).iloc[:, VARIANT - 1].dropna().astype(int).values
n, lam = len(x), x.mean(); print(f"Вариант {VARIANT}: n = {n}, λ = {lam:.3f}, min = {x.min()}, max = {x.max()}")

vals = np.arange(0, x.max() + 1)
obs = np.r_[np.bincount(x, minlength=len(vals)), 0]
exp = n * np.r_[st.poisson.pmf(vals, lam), st.poisson.sf(x.max(), lam)]
o2, e2, co, ce = [], [], 0, 0
for o, e in zip(obs, exp):
    co, ce = co + o, ce + e
    if ce >= 5: o2.append(co); e2.append(ce); co = ce = 0
if ce > 0: o2[-1] += co; e2[-1] += ce
o2, e2 = np.array(o2), np.array(e2)

chi2 = ((o2 - e2)**2 / e2).sum(); k, r = len(o2), 1
p = st.chi2.sf(chi2, k - r - 1)
print(f"k = {k}, χ² = {chi2:.3f}, df = {k-r-1}, p = {p:.4f} ->",
      "соответствует закону Пуассона" if p > .05 else "НЕ соответствует закону Пуассона")

fig, ax = plt.subplots(); w = .4
ax.bar(vals - w/2, obs[:-1], w, label="Фактические"); ax.bar(vals + w/2, exp[:-1], w, label="Теоретические (Пуассон)")
ax.set_xlabel("Значение"); ax.legend(); ax.set_title(f"Вариант {VARIANT}: χ² = {chi2:.2f}, p = {p:.3f}")
fig.savefig(f"graphs/lab14_task_variant{VARIANT}.png", dpi=150, bbox_inches="tight"); plt.show()
