# =====================================================================
#  ЛАБ. 6. МНОЖЕСТВЕННЫЙ РЕГРЕССИОННЫЙ АНАЛИЗ — Positron (аналог Statistica)
#  Нужно: pip install pandas numpy scipy statsmodels
# =====================================================================

# %% Пример 2 (Statistica): текучесть кадров от зарплаты и численности + прогноз
# ЗАДАНИЕ: Построить и проанализировать линейную модель связи текучести кадров со средней зарплатой
#   и численностью работающих. Найти прогноз средней текучести и её отдельных значений
#   при зарплате 200 ден.ед. и численности 300 чел.
# Statistica: Statistics - Multiple Regression (Dependent: TEK; Independent: ZARP, CHISL);
#   Predict dependent variable (confidence limits / prediction limits)
import numpy as np
import pandas as pd
import statsmodels.api as sm

d = pd.DataFrame({
    "ZARP":  [135, 170, 115, 160, 230, 180, 90, 210, 150],
    "CHISL": [200, 320, 290, 200, 170, 230, 280, 300, 260],
    "TEK":   [6.2, 8.7, 11.2, 4.9, 3.1, 4.5, 9.7, 4.2, 5.1]})

m = sm.OLS(d["TEK"], sm.add_constant(d[["ZARP", "CHISL"]])).fit()
print("Коэффициенты (B):", m.params.round(4).to_dict())
print("p-level коэффициентов:", m.pvalues.round(4).to_dict())
print(f"R² = {m.rsquared:.3f}; F = {m.fvalue:.2f}; p (адекватность) = {m.f_pvalue:.4f} ->",
      "модель адекватна" if m.f_pvalue < .05 else "модель НЕадекватна")

pr = m.get_prediction(np.array([[1, 200, 300]])).summary_frame(alpha=.05)
print(f"\nПрогноз при ZARP=200, CHISL=300: {pr['mean'].iloc[0]:.2f}  (в методичке 5,72 — из-за округления коэффициентов)")
print(f"  среднее значение (95%):  ({pr['mean_ci_lower'].iloc[0]:.2f}; {pr['mean_ci_upper'].iloc[0]:.2f})")
print(f"  отдельные значения (95%): ({pr['obs_ci_lower'].iloc[0]:.2f}; {pr['obs_ci_upper'].iloc[0]:.2f})")

# %% Задание 2 (Statistica): зарплата от трёх факторов + прогноз
# ЗАДАНИЕ: Выполнить задание 1 в Statistica: линейная модель средней зарплаты от производительности,
#   глубины добычи и доли высшего сорта; анализ и проверка модели;
#   прогноз зарплаты при производительности 90 т/день, глубине 5 м, доле высшего сорта 30%.
import numpy as np
import pandas as pd
import statsmodels.api as sm

d = pd.DataFrame({
    "PROIZV": [115, 96, 124, 88, 117, 106, 92, 105, 120, 92, 125],
    "GLUB":   [4.3, 5.6, 3.4, 9.7, 6.5, 3.1, 4.1, 7.1, 3.7, 2.8, 5.1],
    "DOLYA":  [46, 34, 48, 32, 46, 36, 34, 35, 47, 32, 47],
    "ZARP":   [1200, 1000, 1300, 900, 1250, 1080, 950, 1000, 1240, 950, 1300]})

m = sm.OLS(d["ZARP"], sm.add_constant(d[["PROIZV", "GLUB", "DOLYA"]])).fit()
print("Коэффициенты (B):", m.params.round(3).to_dict())
print("p-level коэффициентов:", m.pvalues.round(4).to_dict())
print(f"R² = {m.rsquared:.3f}; F = {m.fvalue:.2f}; p (адекватность) = {m.f_pvalue:.2e} ->",
      "модель адекватна" if m.f_pvalue < .05 else "модель НЕадекватна")

pr = m.get_prediction(np.array([[1, 90, 5, 30]])).summary_frame(alpha=.05)
print(f"\nПрогноз: {pr['mean'].iloc[0]:.1f}")
print(f"  среднее значение (95%):  ({pr['mean_ci_lower'].iloc[0]:.1f}; {pr['mean_ci_upper'].iloc[0]:.1f})")
print(f"  отдельные значения (95%): ({pr['obs_ci_lower'].iloc[0]:.1f}; {pr['obs_ci_upper'].iloc[0]:.1f})")

# %% Пример 3 (Statistica): линейная и степенная модели Y от X1, X2 (18 наблюдений)
# ЗАДАНИЕ: Построить линейную Y = a0 + a1*X1 + a2*X2 и степенную Y = a0 * X1^a1 * X2^a2 модели
#   зависимости Y от X1 и X2. Выбрать более точную модель (по коэффициенту детерминации).
# Statistica: линейная - Multiple Regression; степенная - Nonlinear estimation, User-specified
#   regression, least squares: Y=A0*X1**A1*X2**A2
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit

d = pd.DataFrame({
    "X1": [50, 50, 50, 60, 60, 60, 70, 70, 70, 80, 80, 80, 90, 90, 90, 100, 100, 100],
    "X2": [10, 15, 28, 12, 18, 30, 10, 20, 27, 11, 17, 28, 13, 17, 29, 12, 16, 27],
    "Y":  [7.4, 9.3, 11.2, 12.6, 17.8, 21.2, 14.2, 24.5, 28.7, 15.3, 25.1, 29.6, 16.7, 25.4, 30.4, 16.9, 26.3, 30.7]})
assert len(d) == 18

lin = sm.OLS(d["Y"], sm.add_constant(d[["X1", "X2"]])).fit()
print(f"Линейная: Y = {lin.params.iloc[0]:.3f} + {lin.params.iloc[1]:.4f}*X1 + {lin.params.iloc[2]:.4f}*X2,",
      f"R² = {lin.rsquared:.4f}, p(F) = {lin.f_pvalue:.2e}")

f = lambda x, a0, a1, a2: a0 * x[0]**a1 * x[1]**a2
p, _ = curve_fit(f, d[["X1", "X2"]].values.T, d["Y"], p0=[.1, 1, 1], maxfev=20000)   # МНК на исходной шкале
r2 = 1 - ((d["Y"] - f(d[["X1", "X2"]].values.T, *p))**2).sum() / ((d["Y"] - d["Y"].mean())**2).sum()
print(f"Степенная: Y = {p[0]:.4f} * X1^{p[1]:.4f} * X2^{p[2]:.4f}, R² = {r2:.4f}")
print("Более точная модель:", "линейная" if lin.rsquared > r2 else "степенная")

# %% Пример 5 (Statistica): пошаговый обратный регрессионный анализ (Backward removal, p = 0.2)
# ЗАДАНИЕ: Построить линейную модель связи Y с X1..X5, оставив только значимые переменные
#   (пошаговый обратный анализ, порог значимости 0,2).
# Statistica: General regression - Multiple regression; Options: Backward removal, p1 (enter) = p2 (remove) = 0,2
import pandas as pd
import statsmodels.api as sm

d = pd.DataFrame({
    "X1": [1.1, 1.7, 0.5, 3.1, 1.1, 1.2, 0.2, 0.7, 0.6, 0.5],
    "X2": [0.2, 0.3, 0.3, 0.2, 9.2, 0.05, 0.02, 0.2, 0.3, 0.05],
    "X3": [24, 35, 10, 34, 30, 28, 18, 20, 30, 30],
    "X4": [40, 120, 100, 168, 100, 180, 25, 18, 40, 80],
    "X5": [0.6, 0.2, 0.9, 0.2, 0.6, 0.7, 1.1, 0.8, 0.6, 0.9],
    "Y":  [90, 186, 26, 200, 82, 50, 8, 20, 45, 15]})
ALPHA = 0.2                                                      # порог удаления

vars_ = ["X1", "X2", "X3", "X4", "X5"]
while True:
    m = sm.OLS(d["Y"], sm.add_constant(d[vars_])).fit()
    print(f"Шаг: переменные {vars_} | R² = {m.rsquared:.3f}, p(F) = {m.f_pvalue:.2e}")
    print(m.pvalues.round(4).to_dict())
    worst = m.pvalues.drop("const").idxmax()
    if m.pvalues[worst] <= ALPHA or len(vars_) == 1: break
    print(f"  -> удаляем {worst} (p = {m.pvalues[worst]:.3f} > {ALPHA})\n"); vars_.remove(worst)

print("\nИТОГ: Y = " + f"{m.params['const']:.2f}" + "".join(f" {v:+.2f}*{k}" for k, v in m.params.drop("const").items()))
print(f"R² = {m.rsquared:.3f}; p(F) = {m.f_pvalue:.2e}")

# %% Задание 3 (Statistica): пошаговый регрессионный анализ для данных задания 1
# ЗАДАНИЕ: Выполнить задание 1 методом пошагового регрессионного анализа
#   (зарплата от производительности, глубины добычи и доли высшего сорта).
import pandas as pd
import statsmodels.api as sm

d = pd.DataFrame({
    "PROIZV": [115, 96, 124, 88, 117, 106, 92, 105, 120, 92, 125],
    "GLUB":   [4.3, 5.6, 3.4, 9.7, 6.5, 3.1, 4.1, 7.1, 3.7, 2.8, 5.1],
    "DOLYA":  [46, 34, 48, 32, 46, 36, 34, 35, 47, 32, 47],
    "ZARP":   [1200, 1000, 1300, 900, 1250, 1080, 950, 1000, 1240, 950, 1300]})
ALPHA = 0.2                                                      # порог удаления (можно 0.05)

vars_ = ["PROIZV", "GLUB", "DOLYA"]
while True:
    m = sm.OLS(d["ZARP"], sm.add_constant(d[vars_])).fit()
    print(f"Шаг: переменные {vars_} | R² = {m.rsquared:.3f}, p(F) = {m.f_pvalue:.2e}")
    print(m.pvalues.round(4).to_dict())
    worst = m.pvalues.drop("const").idxmax()
    if m.pvalues[worst] <= ALPHA or len(vars_) == 1: break
    print(f"  -> удаляем {worst} (p = {m.pvalues[worst]:.3f} > {ALPHA})\n"); vars_.remove(worst)

print("\nИТОГ: ZARP = " + f"{m.params['const']:.2f}" + "".join(f" {v:+.3f}*{k}" for k, v in m.params.drop("const").items()))
print(f"R² = {m.rsquared:.3f}; p(F) = {m.f_pvalue:.2e}")
