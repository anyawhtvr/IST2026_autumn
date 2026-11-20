# HW 3 — Методы оптимизации в машинном обучении

Градиентный спуск и метод Ньютона: реализация, эксперименты на квадратичных
функциях и сравнение на задаче логистической регрессии (данные LIBSVM).

## Файлы

| файл | содержимое |
|---|---|
| `optimization.py` | линейный поиск (`Constant`, `Armijo`, `Wolfe`), `gradient_descent`, `newton` |
| `oracles.py` | оракулы `QuadraticOracle`, `LogRegL2Oracle`, разностные `grad_finite_diff` / `hess_finite_diff` |
| `experiments.py` | код экспериментов и графиков для отчёта (задания 3, 4, 7, 8) |
| `plot_trajectory_2d.py` | линии уровня и траектории методов (из задания) |
| `presubmit_tests.py` | предварительные тесты (из задания) |
| `homework-practice-03-Optimization_in_ML.ipynb` | отчёт: условие, выводы формул, эксперименты, выводы |

## Данные

Наборы данных не хранятся в репозитории (они большие и общедоступные).
Скачайте их с [LIBSVM datasets](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html)
и положите в папку `data/`:

```
data/w8a.txt              # w8a
data/gisette_scale.bz2    # gisette
data/real-sim.bz2         # real-sim
```

## Запуск

Нужны Python 3, `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `pandas`, `pytest`, `jupyter`.

```
python -m pytest presubmit_tests.py    # тесты
jupyter notebook homework-practice-03-Optimization_in_ML.ipynb   # отчёт
```

Все эксперименты воспроизводимы (зафиксирован `seed`). Полный прогон отчёта
занимает ~10 минут (дольше всего — градиентный спуск на gisette).
