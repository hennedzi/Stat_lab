# =====================================================================
#  ЛАБ. 5. РЕГРЕССИОННЫЙ АНАЛИЗ (две величины) — Positron (аналог Statistica)
#  Нужно: pip install pandas numpy scipy statsmodels matplotlib
#  Графики сохраняются в папку graphs/
# =====================================================================

# %% Пример 3 (Statistica): линейная модель Y = a0 + a1*X и прогноз при X = 1.8
# ЗАДАНИЕ: Построить и проанализировать линейную модель связи Y и X (данные примера 1):
#   - уравнение, адекватность модели (p по F), значимость коэффициентов (p-level), R^2;
#   - прогноз среднего значения Y и отдельных значений Y при X = 1,8 (95%);
#   - график модели с уравнением и R^2.
# Statistica: Statistics - Multiple Regression - Summary: Regression results; Predict dependent variable;
#   Graphs - Scatterplots (Fit: Linear, R square, Regression Equation)
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
os.makedirs("graphs", exist_ok=True); plt.close("all")

X = pd.Series([2.2, 2.8, 3.5, 2.6, 2, 2.9, 3.1, 3.4, 1.9, 3.5])
Y = pd.Series([21.4, 28.6, 35.7, 21.7, 18.5, 27.1, 34.1, 31.4, 18.1, 21.2])

m = sm.OLS(Y, sm.add_constant(X)).fit()
print("Коэффициенты (B):", m.params.round(4).to_dict())
print("p-level коэффициентов:", m.pvalues.round(4).to_dict())
print(f"R^2 = {m.rsquared:.4f}; F = {m.fvalue:.3f}; p (адекватность модели) = {m.f_pvalue:.4f}")
print(f"Модель: Y = {m.params.iloc[0]:.2f} + {m.params.iloc[1]:.2f}*X ->",
      "адекватна" if m.f_pvalue < .05 else "НЕадекватна")

pr = m.get_prediction(np.array([[1, 1.8]])).summary_frame(alpha=.05)
print(f"\nПрогноз при X=1.8: {pr['mean'].iloc[0]:.2f}")
print(f"  среднее значение Y (95%):  ({pr['mean_ci_lower'].iloc[0]:.2f}; {pr['mean_ci_upper'].iloc[0]:.2f})")
print(f"  отдельные значения Y (95%): ({pr['obs_ci_lower'].iloc[0]:.2f}; {pr['obs_ci_upper'].iloc[0]:.2f})")

fig, ax = plt.subplots()
ax.scatter(X, Y); xs = np.linspace(X.min(), X.max(), 50)
ax.plot(xs, m.params.iloc[0] + m.params.iloc[1]*xs, "r")
ax.set_xlabel("X"); ax.set_ylabel("Y")
ax.set_title(f"Y = {m.params.iloc[0]:.2f} + {m.params.iloc[1]:.2f}X,  R² = {m.rsquared:.3f}")
fig.savefig("graphs/lab05_ex3_linear.png", dpi=150, bbox_inches="tight"); plt.show()

# %% Задание 2 (Statistica): зарплата от производительности
# ЗАДАНИЕ: Выполнить задание 1 в Statistica: линейная модель зависимости зарплаты Y от
#   производительности X; анализ и проверка модели; диаграмма с трендом; прогноз зарплаты при X = 90.
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
os.makedirs("graphs", exist_ok=True); plt.close("all")

X = pd.Series([115, 96, 124, 88, 117, 106, 92, 105, 120, 92, 125])               # производительность
Y = pd.Series([1200, 1000, 1300, 900, 1250, 1080, 950, 1000, 1240, 950, 1300])   # зарплата

m = sm.OLS(Y, sm.add_constant(X)).fit()
print("Коэффициенты (B):", m.params.round(4).to_dict())
print("p-level коэффициентов:", m.pvalues.round(5).to_dict())
print(f"R^2 = {m.rsquared:.4f}; F = {m.fvalue:.2f}; p (адекватность) = {m.f_pvalue:.5f}")
print(f"Модель: Y = {m.params.iloc[0]:.2f} + {m.params.iloc[1]:.2f}*X ->",
      "адекватна" if m.f_pvalue < .05 else "НЕадекватна")

pr = m.get_prediction(np.array([[1, 90]])).summary_frame(alpha=.05)
print(f"\nПрогноз зарплаты при X=90: {pr['mean'].iloc[0]:.1f}")
print(f"  среднее (95%):  ({pr['mean_ci_lower'].iloc[0]:.1f}; {pr['mean_ci_upper'].iloc[0]:.1f})")
print(f"  отдельные (95%): ({pr['obs_ci_lower'].iloc[0]:.1f}; {pr['obs_ci_upper'].iloc[0]:.1f})")

fig, ax = plt.subplots()
ax.scatter(X, Y); xs = np.linspace(85, 128, 50)
ax.plot(xs, m.params.iloc[0] + m.params.iloc[1]*xs, "r")
ax.set_xlabel("Производительность, изделий/день"); ax.set_ylabel("Средняя зарплата")
ax.set_title(f"Y = {m.params.iloc[0]:.1f} + {m.params.iloc[1]:.2f}X,  R² = {m.rsquared:.3f}")
fig.savefig("graphs/lab05_task2_linear.png", dpi=150, bbox_inches="tight"); plt.show()

# %% Пример 4 (Statistica): нелинейные модели, выбор наиболее точной
# ЗАДАНИЕ: Построить модели зависимости Y от X: логарифмическую, степенную, экспоненциальную и линейную.
#   Определить наиболее точную модель (по коэффициенту детерминации).
# Statistica: Nonlinear estimation - User-specified regression, least squares:
#   Y=A0+A1*LOG(X);  Y=A0*EXP(A1*X);  Y=A0*X**A1;  (линейная) Y=A0+A1*X
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
os.makedirs("graphs", exist_ok=True); plt.close("all")

X = np.array([5, 10, 20, 30, 40, 50, 60.])
Y = np.array([48, 32, 17, 14, 9, 5, 2.])

models = {                                                     # уравнение, начальное приближение
    "Логарифмическая  Y=A0+A1*ln(X)": (lambda x, a0, a1: a0 + a1*np.log(x), [50, -10]),
    "Экспоненциальная Y=A0*exp(A1*X)": (lambda x, a0, a1: a0*np.exp(a1*x), [40, -.05]),
    "Степенная        Y=A0*X^A1":      (lambda x, a0, a1: a0*x**a1, [400, -1]),
    "Линейная         Y=A0+A1*X":      (lambda x, a0, a1: a0 + a1*x, [40, -1])}

fig, axs = plt.subplots(2, 2, figsize=(10, 7)); best = (None, -1)
for ax, (name, (f, p0)) in zip(axs.flat, models.items()):
    p, _ = curve_fit(f, X, Y, p0=p0, maxfev=10000)             # метод наименьших квадратов
    r2 = 1 - ((Y - f(X, *p))**2).sum() / ((Y - Y.mean())**2).sum()   # Proportion of variance accounted for
    print(f"{name}:  A0 = {p[0]:.4f}, A1 = {p[1]:.4f}, R² = {r2:.4f}")
    best = (name, r2) if r2 > best[1] else best
    xs = np.linspace(X.min(), X.max(), 100)
    ax.scatter(X, Y); ax.plot(xs, f(xs, *p), "r"); ax.set_title(f"{name.split()[0]}: R² = {r2:.3f}")
print("\nНаиболее точная модель:", best[0].split("  ")[0].strip(), f"(R² = {best[1]:.4f})")
fig.savefig("graphs/lab05_ex4_nonlinear.png", dpi=150, bbox_inches="tight"); plt.show()
