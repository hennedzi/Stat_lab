# =====================================================================
#  ЛАБ. 11. НЕПАРАМЕТРИЧЕСКИЕ МЕТОДЫ — Positron (аналог Statistica)
#  Нужно: pip install pandas numpy scipy matplotlib
#  Графики сохраняются в папку graphs/
# =====================================================================

# %% Пример 3.1 (Statistica): U-критерий Манна-Уитни — привесы опытной и контрольной групп
# ЗАДАНИЕ: Выяснить, значимо ли различаются привесы в опытной (O) и контрольной (K) группе (α = 0,05).
# Statistica: Nonparametrics - Comparing two independent samples (groups) - Mann-Whitney U-test
#   (Dependent: PRIVES; Indep. (grouping): GRUPPA)
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st
os.makedirs("graphs", exist_ok=True); plt.close("all")

opyt = [692, 700, 621, 540, 560, 680, 630, 670]       # опытная группа (получала добавку)
kontr = [560, 635, 580, 420, 530, 490, 470]           # контрольная группа

u1, p = st.mannwhitneyu(opyt, kontr, method="asymptotic", use_continuity=False)   # p-level, как в Statistica
U = min(u1, len(opyt) * len(kontr) - u1)
print(f"U = {U}, p-level = {p:.4f} ->", "различие ЗНАЧИМО" if p < .05 else "различие незначимо")
fig, ax = plt.subplots(); ax.boxplot([opyt, kontr], tick_labels=["Опытная", "Контрольная"], showmeans=True)
ax.set_ylabel("Привес, г"); ax.set_title("Пример 3.1: привесы")
fig.savefig("graphs/lab11_ex31_box.png", dpi=150, bbox_inches="tight"); plt.show()

# %% Задание к 3.2 (Statistica): содержание вещества в водах двух рек
# ЗАДАНИЕ: Используя критерий Манна-Уитни, определить, имеется ли значимое различие между реками
#   по содержанию вещества (7 проб из реки A и 12 проб из реки B).
import os
import matplotlib.pyplot as plt
from scipy import stats as st
os.makedirs("graphs", exist_ok=True); plt.close("all")

a = [240, 235, 270, 280, 185, 287, 148]
b = [314, 270, 220, 226, 230, 305, 278, 210, 228, 335, 305, 335]

u1, p = st.mannwhitneyu(a, b, method="asymptotic", use_continuity=False)
print(f"U = {min(u1, len(a)*len(b) - u1)}, p-level = {p:.4f} ->", "различие ЗНАЧИМО" if p < .05 else "различие незначимо")
fig, ax = plt.subplots(); ax.boxplot([a, b], tick_labels=["Река A", "Река B"], showmeans=True)
ax.set_ylabel("Содержание вещества"); ax.set_title("Задание: две реки")
fig.savefig("graphs/lab11_task_rivers.png", dpi=150, bbox_inches="tight"); plt.show()

# %% Пример 3.2 (Statistica): критерий Вилкоксона — лечение (до / после)
# ЗАДАНИЕ: Выяснить, значимо ли различие содержания вещества в крови 12 пациентов до и после лечения (α = 0,05).
# Statistica: Nonparametrics - Comparing two dependent samples (groups) - Wilcoxon matched pairs test
import numpy as np
from scipy import stats as st

do    = [112, 98, 99, 120, 95, 94, 101, 115, 105, 93, 108, 107]
posle = [63, 106, 105, 68, 95, 44, 33, 63, 103, 82, 58, 43]

T, p = st.wilcoxon(do, posle)                  # нулевые разности (пациент 5) отбрасываются
print(f"T = {T}, p-level = {p:.4f} ->", "различие ЗНАЧИМО" if p < .05 else "различие незначимо")
print("Снижение у", int((np.array(do) > np.array(posle)).sum()), "из", len(do), "пациентов")

# %% Задание к 3.3 (Statistica): прочность до и после термообработки
# ЗАДАНИЕ: Используя критерий Вилкоксона, определить, имеется ли значимое различие прочности
#   материала до и после обработки (11 образцов).
import numpy as np
from scipy import stats as st

do    = [10, 9, 8, 8, 7, 7, 5, 5, 6, 6, 7]
posle = [10, 8, 9, 10, 6, 10, 6, 6, 7, 7, 9]

T, p = st.wilcoxon(do, posle)
print(f"T = {T}, p-level = {p:.4f} ->", "различие ЗНАЧИМО" if p < .05 else "различие незначимо")
print("Прочность выросла у", int((np.array(posle) > np.array(do)).sum()), "из", len(do), "образцов")

# %% Пример 4.1 (Statistica): ранговый коэффициент Спирмена — шум и заболеваемость
# ЗАДАНИЕ: Определить, значима ли связь заболевания с временем работы в условиях сильного шума
#   (SHUM — ранги 1..7, ZAB — заболеваемость, %).
# Statistica: Nonparametrics - Correlations (Spearman, Kendall tau, gamma) - Spearman rank R
from scipy import stats as st

shum = [1, 2, 3, 4, 5, 6, 7]                      # никогда ... свыше 80% времени (ранги)
zab = [3, 6, 6, 5, 9, 15, 14]

r, p = st.spearmanr(shum, zab)
print(f"Spearman R = {r:.4f}, p-level = {p:.4f} ->", "связь ЗНАЧИМА" if p < .05 else "связь незначима")

# %% Задание 1 (Statistica): возраст и результат теста (14 человек)
# ЗАДАНИЕ: Выполнить анализ зависимости между возрастом и результатом теста (10-балльная шкала)
#   с помощью коэффициента корреляции Спирмена.
import os
import matplotlib.pyplot as plt
from scipy import stats as st
os.makedirs("graphs", exist_ok=True); plt.close("all")

age = [28, 47, 32, 47, 24, 51, 32, 49, 56, 31, 27, 32, 48, 52]
test = [9, 7, 9, 6, 9, 5, 6, 6, 5, 10, 8, 8, 6, 6]

r, p = st.spearmanr(age, test)
print(f"Spearman R = {r:.4f}, p-level = {p:.5f} ->", "связь ЗНАЧИМА" if p < .05 else "связь незначима",
      "(обратная)" if r < 0 else "(прямая)")
fig, ax = plt.subplots(); ax.scatter(age, test); ax.set_xlabel("Возраст"); ax.set_ylabel("Результат теста")
ax.set_title(f"Задание 1: Spearman R = {r:.3f}")
fig.savefig("graphs/lab11_task1_spearman.png", dpi=150, bbox_inches="tight"); plt.show()

# %% Задание 2 (Statistica): тест по математике и профессиональная пригодность
# ЗАДАНИЕ: Используя ранговый коэффициент Спирмена, определить, имеется ли значимая связь между результатом
#   теста по математике (интервалы баллов) и процентом признанных негодными.
from scipy import stats as st

interval = list(range(1, 11))                                   # ниже 10, 10-20, ..., 90-100 (ранги интервалов)
negodn = [48, 32, 33, 18, 20, 18, 7, 7, 5, 4]

r, p = st.spearmanr(interval, negodn)
print(f"Spearman R = {r:.4f}, p-level = {p:.5f} ->", "связь ЗНАЧИМА" if p < .05 else "связь незначима",
      "(чем выше балл по математике, тем меньше негодных)" if r < 0 else "")
