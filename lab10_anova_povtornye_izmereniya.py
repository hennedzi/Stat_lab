# =====================================================================
#  ЛАБ. 10. ДИСПЕРСИОННЫЙ АНАЛИЗ С ПОВТОРНЫМИ ИЗМЕРЕНИЯМИ — Positron (аналог Statistica)
#  Нужно: pip install pandas numpy scipy statsmodels matplotlib
#  Графики сохраняются в папку graphs/
# =====================================================================

# %% Пример 1 (Statistica): прибыль 12 предприятий за три года
# ЗАДАНИЕ: Определить, значимо ли различаются прибыли в разные годы; если да — какие именно годы.
# Statistica: ANOVA - Repeated measures ANOVA (Dependent: prib2012, prib2013, prib2014; Within effects: 3 уровня,
#   фактор GOD); All effects; All effects/Graphs; Post-hoc - Fisher LSD
import os
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st
from statsmodels.stats.anova import AnovaRM
os.makedirs("graphs", exist_ok=True); plt.close("all")

w = pd.DataFrame({"2012": [300, 240, 380, 180, 320, 340, 280, 170, 300, 320, 300, 210],
                  "2013": [330, 210, 370, 190, 340, 360, 250, 210, 290, 300, 280, 220],
                  "2014": [270, 210, 300, 210, 320, 300, 270, 200, 270, 300, 260, 190]})
long = w.rename_axis("id").reset_index().melt("id", var_name="God", value_name="Prib")

r = AnovaRM(long, "Prib", "id", within=["God"]).fit(); print(r.anova_table.round(4))
p = r.anova_table["Pr > F"].iloc[0]; print("Фактор «время» (год):", "ЗНАЧИМ" if p < .05 else "незначим", f"(p = {p:.4f})")

# Fisher LSD: ошибка = MS (год x предприятие), как в Statistica для повторных измерений
n, k = len(w), w.shape[1]; df_err = (n - 1) * (k - 1)
ss_err = ((w.sub(w.mean(axis=1), axis=0) - w.mean() + w.values.mean())**2).values.sum(); mse = ss_err / df_err
means = w.mean(); print("\nСредние:", means.round(2).to_dict())
for a, b in combinations(w.columns, 2):
    pv = 2 * st.t.sf(abs(means[a] - means[b]) / np.sqrt(2 * mse / n), df_err)
    print(f"  {a} - {b}: разность = {means[a]-means[b]:6.2f}, p = {pv:.4f} ->", "ЗНАЧИМО" if pv < .05 else "незначимо")

fig, ax = plt.subplots(); ci = st.t.ppf(.975, df_err) * np.sqrt(mse / n)
ax.errorbar(range(k), means, ci, fmt="o-", capsize=8); ax.set_xticks(range(k), w.columns)
ax.set_title("Средняя прибыль по годам (95% ДИ)"); ax.set_ylabel("Прибыль")
fig.savefig("graphs/lab10_ex1_means.png", dpi=150, bbox_inches="tight"); plt.show()

# %% Пример 2 (Statistica): прибыль за три года + категориальная переменная «отрасль»
# ЗАДАНИЕ: Определить факторы, существенно влияющие на прибыль (время, отрасль, взаимодействие);
#   выявить значимо различающиеся группы (Fisher LSD).
# Statistica: Repeated measures ANOVA (Dependent: prib2012-2014; Categorical factors: Otrasl; Within effects: God)
import os
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st
os.makedirs("graphs", exist_ok=True); plt.close("all")

w = pd.DataFrame({"Otrasl": ["Н","М","Н","М","Н","Н","М","М","М","Н","М","Н"],      # Н - нефтяная, М - металлургическая
                  "2012": [300, 240, 380, 180, 320, 340, 280, 170, 300, 320, 300, 210],
                  "2013": [330, 210, 370, 190, 340, 360, 250, 210, 290, 300, 280, 220],
                  "2014": [270, 210, 300, 210, 320, 300, 270, 200, 270, 300, 260, 190]})
Y = w[["2012", "2013", "2014"]]; N, k = len(w), 3
G = w.Otrasl.nunique(); ng = w.groupby("Otrasl").size()            # группы одинакового размера (6 и 6)
grand = Y.values.mean(); smean = Y.mean(axis=1); gmean = w.groupby("Otrasl")[["2012", "2013", "2014"]].mean()

ss_otr = k * (ng * (Y.groupby(w.Otrasl).mean().mean(axis=1) - grand)**2).sum()                 # между отраслями
ss_subj = k * ((smean - smean.groupby(w.Otrasl).transform("mean"))**2).sum()                   # ошибка между предприятиями
ss_god = N * ((Y.mean() - grand)**2).sum()                                                     # время
inter = gmean.sub(gmean.mean(axis=1), axis=0).sub(Y.mean() , axis=1) + grand
ss_int = (ng.values[:, None] * inter**2).values.sum()                                          # время x отрасль
ss_err = ((Y.sub(smean, axis=0))**2).values.sum() - ss_god - ss_int                            # ошибка внутри предприятий
rows = [("Otrasl", ss_otr, G-1, ss_subj, N-G), ("God", ss_god, k-1, ss_err, (N-G)*(k-1)),
        ("God*Otrasl", ss_int, (G-1)*(k-1), ss_err, (N-G)*(k-1))]
print("Эффект        SS        df      F        p")
for name, ss, df, sse, dfe in rows:
    F = (ss / df) / (sse / dfe); p = st.f.sf(F, df, dfe)
    print(f"{name:<12}{ss:9.1f}{df:6d}{F:9.3f}{p:9.4f}  ->", "ЗНАЧИМ" if p < .05 else "незначим")

print("\nСредние по отраслям и годам:\n", gmean.round(1))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
gmean.T.plot(marker="o", ax=ax[0], title="Прибыль по годам (линии — отрасль)")
gmean.plot(marker="o", ax=ax[1], title="Прибыль по отраслям (линии — год)")
fig.savefig("graphs/lab10_ex2_interaction.png", dpi=150, bbox_inches="tight"); plt.show()

# Fisher LSD по 6 группам (отрасль x год): внутри одной отрасли ошибка = MS(внутри предприятий), между отраслями —
# объединённая ошибка (приближённо; результаты Statistica могут немного отличаться)
mse_w, df_w = ss_err / ((N - G) * (k - 1)), (N - G) * (k - 1)
mse_p, df_p = (ss_subj + ss_err) / (N - G + df_w), (N - G) + df_w
print(f"\nFisher LSD по группам (отрасль x год), значимые пары:")
cells = {(o, y): Y.loc[w.Otrasl == o, y].mean() for o in ng.index for y in Y.columns}
for (a, va), (b, vb) in combinations(cells.items(), 2):
    mse, dfe = (mse_w, df_w) if a[0] == b[0] else (mse_p, df_p)
    pv = 2 * st.t.sf(abs(va - vb) / np.sqrt(2 * mse / ng.iloc[0]), dfe)
    if pv < .05: print(f"  {a} - {b}: разность = {va-vb:6.1f}, p = {pv:.4f}")

# %% Задание 1 (Statistica): зарплаты 6 предприятий за 4 года
# ЗАДАНИЕ: Найти, значимо ли различались зарплаты в разные годы; если да — определить годы, когда зарплаты
#   различались значимо.
import os
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st
from statsmodels.stats.anova import AnovaRM
os.makedirs("graphs", exist_ok=True); plt.close("all")

w = pd.DataFrame({"2015": [420, 370, 480, 480, 620, 620], "2016": [370, 480, 430, 390, 540, 540],
                  "2017": [520, 460, 570, 510, 600, 580], "2018": [490, 410, 480, 500, 570, 520]})
long = w.rename_axis("id").reset_index().melt("id", var_name="God", value_name="Zarp")
r = AnovaRM(long, "Zarp", "id", within=["God"]).fit(); print(r.anova_table.round(4))
p = r.anova_table["Pr > F"].iloc[0]; print("Различие по годам:", "ЗНАЧИМО" if p < .05 else "незначимо", f"(p = {p:.4f})")

n, k = len(w), w.shape[1]; df_err = (n - 1) * (k - 1)
mse = ((w.sub(w.mean(axis=1), axis=0) - w.mean() + w.values.mean())**2).values.sum() / df_err
means = w.mean(); print("\nСредние:", means.round(1).to_dict(), "\nFisher LSD:")
for a, b in combinations(w.columns, 2):
    pv = 2 * st.t.sf(abs(means[a] - means[b]) / np.sqrt(2 * mse / n), df_err)
    print(f"  {a} - {b}: разность = {means[a]-means[b]:6.1f}, p = {pv:.4f} ->", "ЗНАЧИМО" if pv < .05 else "незначимо")
fig, ax = plt.subplots(); ci = st.t.ppf(.975, df_err) * np.sqrt(mse / n)
ax.errorbar(range(k), means, ci, fmt="o-", capsize=8); ax.set_xticks(range(k), w.columns)
ax.set_title("Средняя зарплата по годам (95% ДИ)")
fig.savefig("graphs/lab10_task1_means.png", dpi=150, bbox_inches="tight"); plt.show()

# %% Задание 2 (Statistica): зарплаты 10 предприятий трёх отраслей за два года
# ЗАДАНИЕ: Определить факторы, значимо влиявшие на зарплату (год, отрасль) и их взаимодействие;
#   найти комбинации факторов, при которых зарплата различалась значимо.
# !!! ВНИМАНИЕ: в методичке ячейки с названиями отраслей объединены, границы групп по документу не видны.
#   Принято: предприятия 1-3 — металлургия, 4-7 — химия, 8-10 — нефть (предприятие 6 — химическое, как в тексте).
#   Если у вас иначе — измените список OTRASL.
import os
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
os.makedirs("graphs", exist_ok=True); plt.close("all")

OTRASL = ["Мет"]*3 + ["Хим"]*4 + ["Неф"]*3
w = pd.DataFrame({"Otrasl": OTRASL,
                  "y15": [170, 300, 380, 180, 320, 340, 280, 420, 350, 210],
                  "y16": [240, 370, 430, 220, 340, 360, 260, 330, 270, 200]})
w["d"] = w.y16 - w.y15; w["s"] = (w.y15 + w.y16) / 2                         # разность лет и среднее по предприятию

time = smf.ols("d ~ C(Otrasl, Sum)", w).fit(); tt = sm.stats.anova_lm(time, typ=3)    # эффекты со временем
betw = smf.ols("s ~ C(Otrasl, Sum)", w).fit(); tb = sm.stats.anova_lm(betw, typ=3)    # межгрупповые эффекты
print(f"Отрасль:          F = {tb.F.iloc[1]:.3f}, p = {tb['PR(>F)'].iloc[1]:.4f}")
print(f"Год:              F = {tt.F.iloc[0]:.3f}, p = {tt['PR(>F)'].iloc[0]:.4f}")
print(f"Год x Отрасль:    F = {tt.F.iloc[1]:.3f}, p = {tt['PR(>F)'].iloc[1]:.4f}")

g = w.groupby("Otrasl")[["y15", "y16"]].mean(); print("\nСредние:\n", g.round(1))
ax = g.T.plot(marker="o", title="Зарплата по годам (линии — отрасль)"); ax.set_xticks([0, 1], ["2015", "2016"])
ax.figure.savefig("graphs/lab10_task2_interaction.png", dpi=150, bbox_inches="tight"); plt.show()

# %% Задание 3 (Statistica): зарплаты 22 предприятий, три отрасли и два региона, два года
# ЗАДАНИЕ: Определить факторы (год, отрасль, регион), значимо влиявшие на зарплату; найти комбинации
#   факторов, при которых зарплата различалась значимо.
# !!! ВНИМАНИЕ: границы отраслей в таблице методички не видны (ячейки объединены).
#   Принято: Север (10 предприятий) — 3 мет / 4 хим / 3 неф; Юг (12) — по 4 в каждой отрасли. Измените OTRASL при необходимости.
import os
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
os.makedirs("graphs", exist_ok=True); plt.close("all")

north = pd.DataFrame({"Region": "Север", "Otrasl": ["Мет"]*3 + ["Хим"]*4 + ["Неф"]*3,
                      "y15": [170, 300, 380, 180, 320, 340, 280, 420, 350, 210],
                      "y16": [240, 370, 430, 220, 340, 360, 260, 330, 270, 200]})
south = pd.DataFrame({"Region": "Юг", "Otrasl": ["Мет"]*4 + ["Хим"]*4 + ["Неф"]*4,
                      "y15": [270, 320, 350, 480, 420, 310, 480, 420, 380, 510, 400, 470],
                      "y16": [240, 410, 380, 500, 450, 330, 460, 470, 420, 380, 400, 430]})
w = pd.concat([north, south], ignore_index=True); print("N =", len(w))
w["d"] = w.y16 - w.y15; w["s"] = (w.y15 + w.y16) / 2

tt = sm.stats.anova_lm(smf.ols("d ~ C(Otrasl, Sum) * C(Region, Sum)", w).fit(), typ=3)   # год и его взаимодействия
tb = sm.stats.anova_lm(smf.ols("s ~ C(Otrasl, Sum) * C(Region, Sum)", w).fit(), typ=3)   # межгрупповые эффекты
names = {"Intercept": "Год", "C(Otrasl, Sum)": "Год x Отрасль", "C(Region, Sum)": "Год x Регион",
         "C(Otrasl, Sum):C(Region, Sum)": "Год x Отрасль x Регион"}
print("Межгрупповые эффекты:")
for e in tb.index[1:-1]: print(f"  {e}: F = {tb.loc[e, 'F']:.3f}, p = {tb.loc[e, 'PR(>F)']:.4f}")
print("Эффекты, включающие время:")
for e in tt.index[:-1]: print(f"  {names[e]}: F = {tt.loc[e, 'F']:.3f}, p = {tt.loc[e, 'PR(>F)']:.4f}")

g = w.groupby(["Region", "Otrasl"])[["y15", "y16"]].mean(); print("\nСредние:\n", g.round(1))
