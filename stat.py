# %% Пример 1: описательная статистика возраста + диаграмма Mean/SE/SD
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats as st

os.makedirs("graphs", exist_ok=True)
plt.close("all")

gos = pd.Series(
    [
        32,
        48,
        62,
        54,
        59,
        31,
        29,
        47,
        42,
        37,
        51,
        52,
        43,
        47,
        61,
        54,
        53,
        41,
        42,
        36,
        48,
        41,
        49,
        57,
        46,
    ]
)

ci = st.t.interval(0.95, len(gos) - 1, gos.mean(), gos.sem())
print(f"N                   = {len(gos)}")
print(f"Среднее             = {gos.mean():.2f}")
print(f"Стандартное откл.   = {gos.std():.2f}")
print(f"Дисперсия           = {gos.var():.2f}")
print(f"Стандартная ошибка  = {gos.sem():.2f}")
print(f"95% ДИ для среднего = ({ci[0]:.2f}; {ci[1]:.2f})")

fig, ax = plt.subplots(figsize=(4, 5))
ax.errorbar([0], [gos.mean()], [gos.std()], fmt="o", capsize=8, label="±SD")
ax.errorbar([0], [gos.mean()], [gos.sem()], fmt="none", lw=6, label="±SE")
ax.set_xticks([0], ["GOS"])
ax.set_xlim(-0.5, 0.5)
ax.legend()
ax.set_ylabel("Возраст")
ax.set_title("Пример 1: среднее, SE, SD")
fig.savefig("graphs/ex1_mean_se_sd.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Задание 1: описательная статистика прочности + диаграмма Mean/SE/SD
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats as st

os.makedirs("graphs", exist_ok=True)
plt.close("all")

strength = pd.Series(
    [
        68,
        75,
        34,
        92,
        57,
        29,
        38,
        49,
        84,
        68,
        71,
        42,
        67,
        73,
        37,
        51,
        97,
        42,
        83,
        43,
        54,
        68,
        67,
    ]
)

ci = st.t.interval(0.95, len(strength) - 1, strength.mean(), strength.sem())
print(f"N                   = {len(strength)}")
print(f"Среднее             = {strength.mean():.2f}")
print(f"Стандартное откл.   = {strength.std():.2f}")
print(f"Дисперсия           = {strength.var():.2f}")
print(f"Стандартная ошибка  = {strength.sem():.2f}")
print(f"95% ДИ для среднего = ({ci[0]:.2f}; {ci[1]:.2f})")

fig, ax = plt.subplots(figsize=(4, 5))
ax.errorbar([0], [strength.mean()], [strength.std()], fmt="o", capsize=8, label="±SD")
ax.errorbar([0], [strength.mean()], [strength.sem()], fmt="none", lw=6, label="±SE")
ax.set_xticks([0], ["Прочность"])
ax.set_xlim(-0.5, 0.5)
ax.legend()
ax.set_ylabel("Прочность")
ax.set_title("Задание 1: среднее, SE, SD")
fig.savefig("graphs/task1_mean_se_sd.png", dpi=150, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(4, 5))  # классический вариант
ax.boxplot(strength, tick_labels=["Прочность"], showmeans=True)
ax.set_ylabel("Прочность")
ax.set_title("Задание 1: диаграмма размаха")
fig.savefig("graphs/task1_boxplot.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Пример 2: таблица частот и гистограмма возраста (шаг 10, старт 20)
import os
import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("graphs", exist_ok=True)
plt.close("all")

gos = pd.Series(
    [
        32,
        48,
        62,
        54,
        59,
        31,
        29,
        47,
        42,
        37,
        51,
        52,
        43,
        47,
        61,
        54,
        53,
        41,
        42,
        36,
        48,
        41,
        49,
        57,
        46,
    ]
)

bins = range(20, 80, 10)
t = pd.cut(gos, bins, right=False).value_counts(sort=False).to_frame("Частота")
t["Накопл. частота"] = t["Частота"].cumsum()
t["Процент"] = 100 * t["Частота"] / t["Частота"].sum()
t["Накопл. процент"] = t["Процент"].cumsum()
print(t.round(2))

fig, ax = plt.subplots()
ax.hist(gos, bins, edgecolor="k")
ax.set_xticks(list(bins))
ax.set_title("Пример 2: гистограмма возраста")
ax.set_xlabel("Возраст")
ax.set_ylabel("Частота")
fig.savefig("graphs/ex2_hist.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Задание 2 (частоты): таблица частот и гистограмма прочности
import os
import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("graphs", exist_ok=True)
plt.close("all")

strength = pd.Series(
    [
        68,
        75,
        34,
        92,
        57,
        29,
        38,
        49,
        84,
        68,
        71,
        42,
        67,
        73,
        37,
        51,
        97,
        42,
        83,
        43,
        54,
        68,
        67,
    ]
)

bins = range(20, 110, 10)
t = pd.cut(strength, bins, right=False).value_counts(sort=False).to_frame("Частота")
t["Накопл. частота"] = t["Частота"].cumsum()
t["Процент"] = 100 * t["Частота"] / t["Частота"].sum()
t["Накопл. процент"] = t["Процент"].cumsum()
print(t.round(2))

fig, ax = plt.subplots()
ax.hist(strength, bins, edgecolor="k")
ax.set_xticks(list(bins))
ax.set_title("Задание 2: гистограмма прочности")
ax.set_xlabel("Прочность")
ax.set_ylabel("Частота")
fig.savefig("graphs/task2_hist.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Пример 3: дискретная величина — таблица частот и гистограмма
import os
import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("graphs", exist_ok=True)
plt.close("all")

books = pd.Series([3, 5, 1, 6, 3, 5, 7, 2, 6, 1, 4, 2, 6, 5, 3, 9, 7, 3, 5, 2, 4])

t = books.value_counts().sort_index().to_frame("Частота")  # All distinct values
t["Накопл. частота"] = t["Частота"].cumsum()
t["Процент"] = 100 * t["Частота"] / t["Частота"].sum()
t["Накопл. процент"] = t["Процент"].cumsum()
print(t.round(2))

fig, ax = plt.subplots()
t["Частота"].plot.bar(ax=ax, edgecolor="k", width=0.9)
ax.tick_params(axis="x", rotation=0)
ax.set_title("Пример 3: количество книг")
ax.set_xlabel("Книг")
ax.set_ylabel("Читателей")
fig.savefig("graphs/ex3_hist.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nБез накопленных величин:")
print(t[["Частота", "Процент"]].round(2))

# %% Пример 4: сравнение независимых выборок (F-критерий -> t-критерий Стьюдента)
import pandas as pd
from scipy import stats as st

gos = pd.Series(
    [
        32,
        48,
        62,
        54,
        59,
        31,
        29,
        47,
        42,
        37,
        51,
        52,
        43,
        47,
        61,
        54,
        53,
        41,
        42,
        36,
        48,
        41,
        49,
        57,
        46,
    ]
)
neg = pd.Series(
    [48, 29, 37, 32, 47, 52, 38, 34, 41, 32, 47, 53, 42, 35, 37, 29, 51, 42, 53]
)

print(f"GOS:   N={len(gos)}, среднее={gos.mean():.2f}, SD={gos.std():.2f}")
print(f"NEGOS: N={len(neg)}, среднее={neg.mean():.2f}, SD={neg.std():.2f}")

F = gos.var() / neg.var()
d1, d2 = len(gos) - 1, len(neg) - 1
p_var = 2 * min(st.f.cdf(F, d1, d2), st.f.sf(F, d1, d2))
equal = p_var > 0.05
print(
    f"\nF-критерий: F = {F:.3f}, p = {p_var:.3f} -> дисперсии "
    f"{'не различаются' if equal else 'различаются'}"
)

t, p = st.ttest_ind(gos, neg, equal_var=equal)
print(f"t-критерий ({'Стьюдента' if equal else 'Уэлча'}): t = {t:.3f}, p = {p:.4f}")
print("Вывод: различие средних", "значимо" if p < 0.05 else "незначимо")

# %% Пример 5: диаграмма размаха для двух выборок
import os
import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("graphs", exist_ok=True)
plt.close("all")

gos = pd.Series(
    [
        32,
        48,
        62,
        54,
        59,
        31,
        29,
        47,
        42,
        37,
        51,
        52,
        43,
        47,
        61,
        54,
        53,
        41,
        42,
        36,
        48,
        41,
        49,
        57,
        46,
    ]
)
neg = pd.Series(
    [48, 29, 37, 32, 47, 52, 38, 34, 41, 32, 47, 53, 42, 35, 37, 29, 51, 42, 53]
)
g, names = [gos, neg], ["GOS", "NEGOS"]
x = range(len(g))

fig, ax = plt.subplots()
ax.errorbar(
    x, [s.mean() for s in g], [s.std() for s in g], fmt="o", capsize=8, label="±SD"
)
ax.errorbar(
    x, [s.mean() for s in g], [s.sem() for s in g], fmt="none", lw=6, label="±SE"
)
ax.set_xticks(x, names)
ax.set_xlim(-0.5, len(g) - 0.5)
ax.legend()
ax.set_ylabel("Возраст")
ax.set_title("Пример 5: среднее, SE, SD")
fig.savefig("graphs/ex5_mean_se_sd.png", dpi=150, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
ax.boxplot(g, tick_labels=names, showmeans=True)
ax.set_ylabel("Возраст")
ax.set_title("Пример 5: классическая диаграмма размаха")
fig.savefig("graphs/ex5_boxplot.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Задание 2: сравнение прочности двух материалов
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats as st

os.makedirs("graphs", exist_ok=True)
plt.close("all")

m1 = pd.Series(
    [
        68,
        75,
        34,
        92,
        57,
        29,
        38,
        49,
        84,
        68,
        71,
        42,
        67,
        73,
        37,
        51,
        97,
        42,
        83,
        43,
        54,
        68,
        67,
    ]
)
m2 = pd.Series(
    [
        63,
        58,
        69,
        74,
        65,
        74,
        63,
        80,
        75,
        69,
        71,
        41,
        77,
        69,
        49,
        58,
        77,
        79,
        52,
        49,
        63,
        69,
        52,
        77,
    ]
)

print(f"Материал 1: N={len(m1)}, среднее={m1.mean():.2f}, SD={m1.std():.2f}")
print(f"Материал 2: N={len(m2)}, среднее={m2.mean():.2f}, SD={m2.std():.2f}")

F = m1.var() / m2.var()
d1, d2 = len(m1) - 1, len(m2) - 1
p_var = 2 * min(st.f.cdf(F, d1, d2), st.f.sf(F, d1, d2))
equal = p_var > 0.05
print(
    f"\nF-критерий: F = {F:.3f}, p = {p_var:.4f} -> дисперсии "
    f"{'не различаются' if equal else 'различаются'}"
)

t, p = st.ttest_ind(m1, m2, equal_var=equal)
print(f"t-критерий ({'Стьюдента' if equal else 'Уэлча'}): t = {t:.3f}, p = {p:.4f}")
if p < 0.05:
    print(
        "Вывод: различие значимо; прочнее материал", 1 if m1.mean() > m2.mean() else 2
    )
else:
    print("Вывод: различие незначимо, сказать какой материал прочнее нельзя")

g, names = [m1, m2], ["Материал 1", "Материал 2"]
x = range(len(g))
fig, ax = plt.subplots()
ax.errorbar(
    x, [s.mean() for s in g], [s.std() for s in g], fmt="o", capsize=8, label="±SD"
)
ax.errorbar(
    x, [s.mean() for s in g], [s.sem() for s in g], fmt="none", lw=6, label="±SE"
)
ax.set_xticks(x, names)
ax.set_xlim(-0.5, len(g) - 0.5)
ax.legend()
ax.set_ylabel("Прочность")
ax.set_title("Задание 2: среднее, SE, SD")
fig.savefig("graphs/task2_mean_se_sd.png", dpi=150, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
ax.boxplot(g, tick_labels=names, showmeans=True)
ax.set_ylabel("Прочность")
ax.set_title("Задание 2: классическая диаграмма размаха")
fig.savefig("graphs/task2_boxplot.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Пример 6: текучесть кадров — а) статистики, б) диаграммы, в) сравнение групп
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats as st

os.makedirs("graphs", exist_ok=True)
plt.close("all")

df = pd.DataFrame(
    {
        "FS": list("НГНГННГНГГГГННГГНГГГННГГНГГ"),
        "TEKKAD": [
            5.1,
            4.8,
            4.6,
            6.2,
            4.7,
            5.2,
            5.8,
            3.7,
            4.9,
            7.1,
            6.7,
            5.4,
            4.2,
            6.1,
            6.8,
            7.4,
            3.6,
            6.4,
            6.8,
            5.5,
            4.8,
            6.3,
            6.1,
            6.8,
            5.2,
            7.1,
            5.9,
        ],
    }
)
print("Предприятий:", len(df))

# а) статистики по группам и по всем
names = {"count": "N", "mean": "Среднее", "std": "SD", "var": "Дисперсия", "sem": "SE"}
by = df.groupby("FS").TEKKAD
tbl = by.agg(list(names)).rename(columns=names)
tbl.loc["Все"] = df.TEKKAD.agg(list(names)).values
tbl["ДИ 95% от"] = [
    st.t.interval(0.95, n - 1, m, s)[0]
    for n, m, s in zip(tbl.N, tbl["Среднее"], tbl.SE)
]
tbl["ДИ 95% до"] = [
    st.t.interval(0.95, n - 1, m, s)[1]
    for n, m, s in zip(tbl.N, tbl["Среднее"], tbl.SE)
]
print("\nа) Статистики (Г - государственные, Н - негосударственные):")
print(tbl.round(3))

# б) диаграммы размаха по группам
x = range(len(by.mean()))
fig, ax = plt.subplots()
ax.errorbar(x, by.mean(), by.std(), fmt="o", capsize=8, label="±SD")
ax.errorbar(x, by.mean(), by.sem(), fmt="none", lw=6, label="±SE")
ax.set_xticks(x, by.mean().index)
ax.set_xlim(-0.5, len(x) - 0.5)
ax.legend()
ax.set_ylabel("Текучесть, %")
ax.set_title("Пример 6: среднее, SE, SD")
fig.savefig("graphs/ex6_mean_se_sd.png", dpi=150, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots()
df.boxplot("TEKKAD", by="FS", ax=ax)
ax.set_ylabel("Текучесть, %")
ax.set_xlabel("Форма собственности")
ax.set_title("Пример 6: диаграмма размаха")
fig.suptitle("")
fig.savefig("graphs/ex6_boxplot.png", dpi=150, bbox_inches="tight")
plt.show()

# в) сравнение групп
a, b = df.TEKKAD[df.FS == "Г"], df.TEKKAD[df.FS == "Н"]
F = a.var() / b.var()
d1, d2 = len(a) - 1, len(b) - 1
p_var = 2 * min(st.f.cdf(F, d1, d2), st.f.sf(F, d1, d2))
equal = p_var > 0.05
print(
    f"\nв) F-критерий: F = {F:.3f}, p = {p_var:.3f} -> дисперсии "
    f"{'не различаются' if equal else 'различаются'}"
)
t, p = st.ttest_ind(a, b, equal_var=equal)
print(f"t-критерий ({'Стьюдента' if equal else 'Уэлча'}): t = {t:.3f}, p = {p:.5f}")
print(
    "Вывод: различие",
    "значимо" if p < 0.05 else "незначимо",
    f"(у Г среднее {a.mean():.2f}, у Н {b.mean():.2f})",
)

# %% Пример 7: зависимые выборки (загрязнение до / после)
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st

os.makedirs("graphs", exist_ok=True)
plt.close("all")

do = [88, 86, 92, 75, 69, 80, 97, 74, 83, 69, 93, 80, 68, 101, 93, 79, 67]
posle = [85, 77, 91, 74, 70, 77, 92, 75, 83, 65, 90, 72, 65, 94, 90, 79, 64]

t, p = st.ttest_rel(do, posle)
print(f"Среднее до = {np.mean(do):.2f}, после = {np.mean(posle):.2f}")
print(f"t-критерий для зависимых выборок: t = {t:.3f}, df = {len(do) - 1}, p = {p:.4f}")
print("Вывод: снижение", "значимо" if p < 0.05 else "незначимо")

fig, ax = plt.subplots(figsize=(4, 5))
ax.plot([0, 1], [do, posle], "o-", color="gray", alpha=0.5)
ax.plot([0, 1], [np.mean(do), np.mean(posle)], "o-", color="red", lw=3, label="Среднее")
ax.set_xticks([0, 1], ["До", "После"])
ax.legend()
ax.set_ylabel("Концентрация")
ax.set_title("Пример 7: до и после")
fig.savefig("graphs/ex7_paired.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Задание 3: прочность до и после обработки
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st

os.makedirs("graphs", exist_ok=True)
plt.close("all")

before = [
    68,
    75,
    34,
    92,
    57,
    29,
    38,
    49,
    84,
    68,
    71,
    42,
    67,
    73,
    37,
    51,
    97,
    42,
    83,
    43,
    54,
    68,
    67,
]
after = [
    72,
    70,
    38,
    97,
    61,
    28,
    38,
    52,
    87,
    72,
    77,
    44,
    72,
    70,
    35,
    58,
    96,
    42,
    87,
    46,
    58,
    65,
    69,
]

t, p = st.ttest_rel(before, after)
t1, p1 = st.ttest_rel(after, before, alternative="greater")  # проверка роста
print(f"Среднее до = {np.mean(before):.2f}, после = {np.mean(after):.2f}")
print(f"Двусторонний:  t = {t:.3f}, df = {len(before) - 1}, p = {p:.4f}")
print(f"Односторонний (после > до): t = {t1:.3f}, p = {p1:.4f}")
print(
    "Вывод: обработка",
    "значимо повысила" if p1 < 0.05 else "не повысила значимо",
    "прочность",
)

fig, ax = plt.subplots(figsize=(4, 5))
ax.plot([0, 1], [before, after], "o-", color="gray", alpha=0.5)
ax.plot(
    [0, 1], [np.mean(before), np.mean(after)], "o-", color="red", lw=3, label="Среднее"
)
ax.set_xticks([0, 1], ["До", "После"])
ax.legend()
ax.set_ylabel("Прочность")
ax.set_title("Задание 3: до и после обработки")
fig.savefig("graphs/task3_paired.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Пример 8: средний вес упаковок против 1 кг
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st

os.makedirs("graphs", exist_ok=True)
plt.close("all")

pack = [
    1.15,
    1.09,
    0.87,
    1.18,
    1.20,
    1.08,
    1.12,
    0.96,
    0.89,
    0.97,
    0.91,
    0.94,
    0.92,
    1.07,
    1.16,
]

t, p = st.ttest_1samp(pack, 1)
print(
    f"N = {len(pack)}, среднее = {np.mean(pack):.3f}, SD = {np.std(pack, ddof=1):.3f}, "
    f"SE = {st.sem(pack):.3f}"
)
print(f"t = {t:.3f}, df = {len(pack) - 1}, p = {p:.4f}")
print(
    "Вывод: среднее", "отличается от 1 кг" if p < 0.05 else "можно считать равным 1 кг"
)

fig, ax = plt.subplots(figsize=(4, 5))
ax.boxplot(pack, tick_labels=["Вес, кг"], showmeans=True)
ax.axhline(1, color="r", label="1 кг")
ax.legend()
ax.set_ylabel("Вес, кг")
ax.set_title("Пример 8: вес упаковок")
fig.savefig("graphs/ex8_boxplot.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Задание 4: средний вес изделий против 50 г
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st

os.makedirs("graphs", exist_ok=True)
plt.close("all")

w = [52, 47, 39, 48, 51, 46, 48, 52, 47, 49, 52, 48]

t, p = st.ttest_1samp(w, 50)
print(
    f"N = {len(w)}, среднее = {np.mean(w):.2f}, SD = {np.std(w, ddof=1):.2f}, SE = {st.sem(w):.2f}"
)
print(f"t = {t:.3f}, df = {len(w) - 1}, p = {p:.4f}")
print(
    "Вывод: среднее", "отличается от 50 г" if p < 0.05 else "можно считать равным 50 г"
)

fig, ax = plt.subplots(figsize=(4, 5))
ax.boxplot(w, tick_labels=["Вес, г"], showmeans=True)
ax.axhline(50, color="r", label="50 г")
ax.legend()
ax.set_ylabel("Вес, г")
ax.set_title("Задание 4: вес изделий")
fig.savefig("graphs/task4_boxplot.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Пример 9: сравнение частот дефектов (технологии A и B)
from statsmodels.stats.proportion import proportions_ztest

x, n = [83, 46], [284, 217]  # дефекты и объёмы выборок
z, p = proportions_ztest(x, n)
print(f"Q1 = {x[0] / n[0]:.4f}, Q2 = {x[1] / n[1]:.4f}")
print(f"z = {z:.3f}, p = {p:.4f}")
print("Вывод: частоты", "различаются значимо" if p < 0.05 else "различаются незначимо")

# %% Задание 5: сравнение частот отказов (материалы М1 и М2)
from statsmodels.stats.proportion import proportions_ztest

x, n = [11, 16], [100, 140]  # отказы и объёмы выборок
z, p = proportions_ztest(x, n)
print(f"Q1 = {x[0] / n[0]:.4f}, Q2 = {x[1] / n[1]:.4f}")
print(f"z = {z:.3f}, p = {p:.4f}")
print("Вывод: частоты", "различаются значимо" if p < 0.05 else "различаются незначимо")
