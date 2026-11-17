# =====================================================================
#  ЛАБ. 3. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ — Positron (аналог Statistica)
#  Ячейки "# %%" запускаются по очереди (Ctrl+Enter), каждая самостоятельна.
#  Нужно: pip install pandas numpy scipy matplotlib
#  Графики сохраняются в папку graphs/
# =====================================================================

# %% Пример 5 (Statistica): парные корреляции текучести кадров с зарплатой и численностью
# ЗАДАНИЕ: По данным примера 1 найти и проанализировать коэффициенты парной корреляции:
#   а) текучесть кадров <-> средняя заработная плата;
#   б) текучесть кадров <-> численность работающих.
# Statistica: Statistics - Basic Statistics - Correlation Matrices - Two lists (rect. matrix):
#   список 1: TEK, список 2: ZARP и CHISL; Display r, p-levels and N's; p-level for highlighting = .05
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats as st
os.makedirs("graphs", exist_ok=True); plt.close("all")

d = pd.DataFrame({
    "ZARP":  [135, 170, 115, 160, 230, 180, 90, 210, 150],        # средняя заработная плата, ден.ед.
    "CHISL": [200, 320, 290, 200, 170, 230, 280, 300, 260],       # численность работающих
    "TEK":   [6.2, 8.7, 11.2, 4.9, 3.1, 4.5, 9.7, 4.2, 5.1]})     # текучесть кадров, %

for x in ["ZARP", "CHISL"]:
    r, p = st.pearsonr(d[x], d["TEK"])
    print(f"TEK и {x}: r = {r:.3f}, p = {p:.4f}, N = {len(d)} -> связь",
          "ЗНАЧИМА" if p < .05 else "незначима", "(отрицательная)" if r < 0 else "(положительная)")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))                    # диаграммы рассеяния
for a, x in zip(ax, ["ZARP", "CHISL"]):
    a.scatter(d[x], d["TEK"]); k, b = st.linregress(d[x], d["TEK"])[:2]
    a.plot(sorted(d[x]), [k*v + b for v in sorted(d[x])], "r")
    a.set_xlabel(x); a.set_ylabel("TEK, %"); a.set_title(f"TEK и {x}")
fig.savefig("graphs/lab03_ex5_scatter.png", dpi=150, bbox_inches="tight"); plt.show()

# %% Пример 6 (Statistica): частные корреляции прочности меха
# ЗАДАНИЕ: По данным примера 3 найти и проанализировать коэффициенты частной корреляции:
#   а) прочность меха <-> длина шерсти (без влияния веса);
#   б) прочность меха <-> вес животного (без влияния длины шерсти).
# Statistica: Correlation Matrices - Two lists: First = DLINA и PROCHN, Second = VES; Partial Correlations
import numpy as np
import pandas as pd
from scipy import stats as st

d = pd.DataFrame({
    "DLINA":  [14, 25, 11, 17, 29, 10, 15, 27, 19, 11, 18, 28],                    # длина шерсти, мм
    "VES":    [6.7, 5.8, 3.9, 4.7, 5.2, 3.5, 7.1, 6.6, 7.1, 4.2, 7.4, 5.1],       # вес животного, кг
    "PROCHN": [48, 30, 38, 42, 37, 42, 49, 41, 53, 52, 43, 37]})                   # прочность меха, баллы
n, k = len(d), 2                                   # объём выборки и число входных величин

P = np.linalg.inv(d[["DLINA", "VES", "PROCHN"]].corr().values)   # обратная матрица корреляций
for i, x in enumerate(["DLINA", "VES"]):
    r = -P[i, 2] / np.sqrt(P[i, i] * P[2, 2])                    # частный коэффициент корреляции
    t = r * np.sqrt((n - k - 1) / (1 - r**2)); p = 2 * st.t.sf(abs(t), n - k - 1)
    print(f"PROCHN и {x} (остальные исключены): r = {r:.3f}, t = {t:.2f}, p = {p:.4f} ->",
          "ЗНАЧИМ" if p < .05 else "незначим")

# %% Задание 3 (Statistica): частные корреляции заработной платы
# ЗАДАНИЕ: По данным задания 1 найти частные коэффициенты корреляции заработной платы с каждым фактором
#   (производительность, глубина добычи, доля высшего сорта). Проверить значимость, сделать выводы.
import numpy as np
import pandas as pd
from scipy import stats as st

d = pd.DataFrame({
    "PROIZV": [115, 96, 124, 88, 117, 106, 92, 105, 120, 92, 125],      # производительность, т/день
    "GLUB":   [4.3, 5.6, 3.4, 9.7, 6.5, 3.1, 4.1, 7.1, 3.7, 2.8, 5.1],  # глубина добычи, м
    "DOLYA":  [46, 34, 48, 32, 46, 36, 34, 35, 47, 32, 47],             # доля высшего сорта, %
    "ZARP":   [1200, 1000, 1300, 900, 1250, 1080, 950, 1000, 1240, 950, 1300]})  # зарплата
n, k = len(d), 3

P = np.linalg.inv(d.corr().values)                                 # столбцы: PROIZV, GLUB, DOLYA, ZARP
for i, x in enumerate(["PROIZV", "GLUB", "DOLYA"]):
    r = -P[i, 3] / np.sqrt(P[i, i] * P[3, 3])
    t = r * np.sqrt((n - k - 1) / (1 - r**2)); p = 2 * st.t.sf(abs(t), n - k - 1)
    print(f"ZARP и {x}: r = {r:.3f}, t = {t:.2f}, p = {p:.4f} ->", "ЗНАЧИМ" if p < .05 else "незначим")
