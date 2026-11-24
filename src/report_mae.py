"""Build the Russian WGS84 MAE report, the full machine-readable table and a Pareto chart."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
FOLDS = range(1, 6)
CSV_NAME = "mae-cv-results.csv"
CHART_NAME = "mae-pareto.svg"
REPORT_NAME = "report-mae.md"


def _ru_n(x):
    return f"{x:g}".replace(".", ",")


_POST_ANCHORS = {
    "maepost:v4_rank_spherical_median": "исходной сферической медианы V4 по рангам ячеек",
    "maepost:median/peer/p1.4": "медианы ячеек с межсеточной поддержкой и степенью вероятностей 1,4",
    "maepost:basin/h225/w0.8": "оценки по пику плотности шириной 225 км с максимальным сдвигом 80 %",
    "maepost:localatom/sharp/w0.32": "медианы с локальной модой весом 32 % и заострёнными вероятностями",
    "maepost:localgate/h200/w0.6": "медианы со сдвигом к локальной моде до 60 % при масштабе согласия 200 км",
}


def family(spec):
    if spec["kind"] == "robust":
        return {"mae_median": "Обучаемые геометрические медианы",
                "irls": "Робастная регрессия IRLS"}[spec["parameters"]["kind"]]
    return {"posterior": "Оценки по распределению ячеек",
            "blend": "Фиксированные смеси двух прогнозов",
            "gate": "Нейросетевое объединение кандидатов"}[spec["kind"]]


def _posterior_text(p):
    kind = p["kind"]
    if kind == "baseline":
        return "Воспроизводит исходную сферическую медиану V4 по 30 центрам, используя десять ведущих ячеек каждой сетки с их сохранёнными вероятностями."
    if kind == "median":
        scheme = {"equal": "три сетки получают равные доли",
                  "fine": "сетка из 1 024 ячеек получает половину веса, две сетки из 512 ячеек — по четверти",
                  "peer": "ячейки получают больший вес при поддержке двух других сеток"}[p["scheme"]]
        return f"Вычисляет сферическую медиану 30 центров. Вероятности внутри каждой сетки возводятся в степень {_ru_n(p['power'])}, {scheme}."
    if kind == "basin":
        return (f"Находит кластер вокруг пика сглаженной плотности с шириной {_ru_n(p['km'])} км "
                f"и сдвигает медиану к его центру до {_ru_n(100*p['fraction'])} % пути в зависимости от массы кластера.")
    if kind == "medoid":
        choices = {"centers": "среди 30 центров ведущих ячеек",
                   "nearby": "среди 30 центров ячеек, локальных моды и среднего и центра кластера шириной 225 км"}[p["candidates"]]
        return f"Выбирает {choices} точку с наименьшим ожидаемым угловым расстоянием. Степень вероятностей ячеек — {_ru_n(p['power'])}."
    if kind == "localatom":
        text = {"ordinary": "Вероятности ячеек сохраняются.",
                "sharp": "Вероятности ячеек возводятся в степень 1,5.",
                "trust": "Доля моды уменьшается при слабой поддержке ячеек и низкой концентрации локальных соседей."}[p["scheme"]]
        return f"Включает локальную моду поиска соседей в сферическую медиану как отдельную точку с исходной долей {_ru_n(100*p['fraction'])} %. {text}"
    if kind == "localgate":
        return (f"Сдвигает медиану к локальной моде не более чем на {_ru_n(100*p['fraction'])} % пути. "
                f"Поправка уменьшается при слабой локальной поддержке и расхождении прогнозов на масштабе {_ru_n(p['km'])} км.")
    if kind == "tail":
        return f"Добавляет для каждой сетки точку, представляющую пропущенные ячейки, умножает её массу на {_ru_n(p['share'])} и вычисляет медиану 33 точек."
    if kind == "smooth":
        return f"Вычисляет сферическую медиану ведущих ячеек, сглаживая абсолютное расстояние на масштабе {_ru_n(p['km'])} км."
    if kind == "likelihood":
        return f"Повышает веса ячеек в окрестности локальной моды гауссовым ядром шириной {_ru_n(p['km'])} км и вычисляет взвешенную сферическую медиану."
    if kind == "adaptive":
        return "Меняет степень вероятностей ячеек по энтропии и согласию ведущих ячеек трёх сеток, затем вычисляет сферическую медиану."
    if kind == "ellipsoid":
        return "Уточняет сферическую медиану двумя шагами алгоритма Вайсфельда с прямой и обратной геодезическими задачами WGS84."
    raise KeyError(kind)


def _gate_text(p):
    recipe, decode = p["recipe"], p["decode"]
    if recipe in ("linear", "mean"):
        head = "Линейная модель" if recipe == "linear" else "Многослойная сеть"
        text = f"{head} предсказывает веса кандидатов по признакам запроса и усредняет их смещения, обучаясь по среднему угловому расстоянию."
    elif recipe == "median":
        text = "Сеть обучается по среднему угловому расстоянию и оценивает веса кандидатов для сглаженной геометрической медианы."
    elif recipe == "residual":
        text = "Сеть вычисляет взвешенную медиану и добавляет ограниченную двумерную поправку, размер которой зависит от разброса кандидатов."
    elif recipe == "laplace":
        action = {"density": "вычисляет медиану со сглаживанием по ширине каждого ядра",
                  "mode": "выбирает кандидата с максимальной плотностью смеси"}[decode]
        text = f"Сеть обучает веса и ширины сферических ядер Лапласа, затем {action}."
    else:
        score = {"risk": "изменение ожидаемой ошибки каждого кандидата относительно первого эксперта",
                 "oracle": "вероятность оказаться близко к истинной обучающей точке",
                 "cdf": "ожидаемое расстояние по вероятностям превышения восьми порогов от 20 до 15 000 км"}[recipe]
        action = {"hard": "выбирает одного кандидата", "mean": "усредняет смещения кандидатов",
                  "median": "вычисляет взвешенную медиану кандидатов"}[decode]
        text = f"Сеть оценивает {score} и по этой оценке {action}."
    if p["temperature"] != 1:
        text += f" При прогнозе температура {_ru_n(p['temperature'])} усиливает ведущие веса."
    return text


def describe(spec):
    if spec["kind"] == "blend":
        return (f"Сдвигает прогноз {_POST_ANCHORS[spec['posterior']]} на {_ru_n(100*spec['weight'])} % "
                f"сферической дуги к прогнозу эксперта {spec['expert']}.")
    p = spec["parameters"]
    if spec["kind"] == "posterior":
        return _posterior_text(p)
    if spec["kind"] == "gate":
        return _gate_text(p)
    anchor = {"base": "прогноза V4", "median": "сферической медианы V4 по рангам ячеек"}[spec["pool"]]
    pool = {"core": "первые четыре эксперта", "all": "все эксперты"}[p["pool"]]
    if p["kind"] == "mae_median":
        risk = {"global": "обратной средней геодезической ошибке каждого эксперта",
                "spread": "обратной средней ошибке при сопоставимом разбросе прогнозов",
                "agreement": "обратной средней ошибке с дополнительным снижением веса изолированных прогнозов"}[p["risk"]]
        output = {"median": "Возвращает геометрическую медиану.",
                  "blend": "Возвращает смесь медианы (80 %) и взвешенного среднего (20 %) в той же плоскости."}[p["location"]]
        return (f"Использует {pool} в плоскости WGS84 вокруг {anchor}. Веса основаны на {risk} "
                f"со степенью {_ru_n(p['power'])}. {output}")
    if p["kind"] != "irls":
        raise KeyError(p["kind"])
    axes = {"shared": "общие для направлений восток и север", "separate": "отдельные для востока и севера"}[p["mode"]]
    context = {"constant": "постоянные", "spread": "зависят от расхождения экспертов",
               "geo_spread": "зависят от прогнозируемого положения и расхождения экспертов"}[p["context"]]
    return (f"Использует {pool} для поправки к координатам {anchor}. IRLS приближает минимум сглаженного "
            f"абсолютного расстояния в плоскости WGS84. Коэффициенты {axes}, {context}. "
            f"Сглаживание {_ru_n(p['smooth_km'])} км, регуляризация {_ru_n(p['alpha'])}, доля поправки {_ru_n(100*p['gain'])} %.")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def optional(path):
    return load(path) if path.is_file() else None


def number(x, digits=2):
    return f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")


def signed(x):
    return f"{x:+.2f}".replace(".", ",").replace("-", "−")


def mdtable(headings, values):
    def line(row):
        return "| " + " | ".join(str(x).replace("|", "&#124;").replace("\n", " ") for x in row) + " |"
    return "\n".join([line(headings), line(["---"] * len(headings)), *(line(row) for row in values)])


def pareto(rows):
    """Keep every configuration on a nondominated pair, including exact ties."""
    frontier = set()
    best_rmse = math.inf
    for _, tied_rows in itertools.groupby(rows, key=lambda r: r["cv_mae_km"]):
        group = list(tied_rows)
        lowest = min(r["cv_rmse_km"] for r in group)
        if lowest < best_rmse:
            frontier.update(r["name"] for r in group if r["cv_rmse_km"] == lowest)
            best_rmse = lowest
    return frontier


def checked_rows(lock):
    specs = lock["specifications"]
    rows = sorted(lock["candidates"], key=lambda r: (r["cv_mae_km"], r["cv_rmse_km"], r["name"]))
    if not rows or {r["name"] for r in rows} != set(specs) or len(rows) != len(specs):
        raise ValueError("Кандидаты и их спецификации не совпадают")
    if rows[0]["name"] != lock["winner"] or len(rows) != lock["n_variants"]:
        raise ValueError("Победитель или число вариантов расходится с протоколом отбора")
    for r in rows:
        for col in ("mae", "rmse"):
            folds = r[f"fold_{col}_km"]
            overall = r[f"cv_{col}_km"]
            if len(folds) != 5 or not all(math.isfinite(x) and x >= 0 for x in [overall, *folds]):
                raise ValueError(f"Неполный набор измерений {r['name']}: {col}")
            pooled = statistics.fmean(folds) if col == "mae" else math.sqrt(statistics.fmean(v*v for v in folds))
            if not math.isclose(overall, pooled, rel_tol=1e-7, abs_tol=.005):
                raise ValueError(f"Несогласованные метрики фолдов {r['name']}: {col}")
    return rows


def check_sources(lock, lock_hash, results, provenance):
    for record in (results, provenance):
        if record is None:
            continue
        keys = {"winner": lock["winner"], "cv_sha256": lock_hash,
                "split_sha256": lock["split_sha256"], "rmse_cv_sha256": lock["rmse_cv_sha256"]}
        for name, expected in keys.items():
            if name in record and record[name] != expected:
                raise ValueError(f"Финальный файл относится к другой версии эксперимента ({name})")


def write_csv(path, rows, specs, front):
    param_names = sorted({k for spec in specs.values() for k in spec.get("parameters", {})})
    fields = ["rank", "configuration", "family", "description_ru", "pareto", "cv_mae_km", "cv_rmse_km",
              *[f"fold_{i}_{metric}_km" for metric in ("mae", "rmse") for i in FOLDS],
              "anchor_pool", "blend_posterior", "blend_expert", "blend_fraction",
              *["param_" + k for k in param_names], "spec_json"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        out = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        out.writeheader()
        for rank, row in enumerate(rows, 1):
            name = row["name"]
            spec = specs[name]
            vals = dict(rank=rank, configuration=name, family=family(spec), description_ru=describe(spec),
                        pareto=int(name in front), cv_mae_km=row["cv_mae_km"], cv_rmse_km=row["cv_rmse_km"],
                        anchor_pool=spec.get("pool", ""), blend_posterior=spec.get("posterior", ""),
                        blend_expert=spec.get("expert", ""), blend_fraction=spec.get("weight", ""),
                        spec_json=json.dumps(spec, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
            for metric in ("mae", "rmse"):
                vals.update({f"fold_{i}_{metric}_km": row[f"fold_{metric}_km"][i-1] for i in FOLDS})
            vals.update({"param_"+k: v for k, v in spec.get("parameters", {}).items()})
            out.writerow(vals)


def chart(path, rows, specs, front):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    groups = defaultdict(list)
    for r in rows:
        groups[family(specs[r["name"]])].append(r)
    colors = ["#1f7281", "#d16d3a", "#54429c", "#217743", "#b03d68", "#5d7782", "#9a621d", "#4d6294"]
    with mpl.rc_context({"svg.fonttype": "none", "font.family": "DejaVu Sans", "svg.hashsalt": "wgs84-mae"}):
        fig, ax = plt.subplots(figsize=(10.5, 6.7), layout="constrained")
        for i, (name, group) in enumerate(sorted(groups.items())):
            ax.scatter([r["cv_mae_km"] for r in group], [r["cv_rmse_km"] for r in group],
                       label=f"{name} ({len(group)})", s=24, color=colors[i % len(colors)],
                       alpha=.65, linewidths=.2, edgecolors="white")
        frontier = sorted({(r["cv_mae_km"],r["cv_rmse_km"]) for r in rows if r["name"] in front})
        ax.plot(*zip(*frontier), color="#293643", marker="o", markersize=4, linewidth=1.15,
                markerfacecolor="none", label="Граница Парето", zorder=5)
        won = rows[0]
        ax.scatter(won["cv_mae_km"], won["cv_rmse_km"], marker="*", s=164, color="#fcce53",
                   edgecolors="#382a18", linewidths=1, label="Выбор по MAE", zorder=6)
        ax.set_title("Ошибка геолокации на пяти внешних фолдах", loc="left", fontweight="bold", pad=10)
        ax.set_xlabel("Среднее расстояние до истинной точки, MAE (км)")
        ax.set_ylabel("Среднеквадратичная ошибка, RMSE (км)")
        fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.grid(color="#e4e9ec", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -.15), ncols=2, frameon=False, fontsize=9)
        fig.savefig(path, format="svg", metadata={"Date": None})
        plt.close(fig)


def hist_table(lock, winner, v4):
    refs = lock["references"]
    entries = [("Победитель текущего отбора по MAE", winner),
               ("Минимальная MAE среди прежних вариантов V4", refs["old_v4_mae_best"]),
               ("Минимальная MAE в прежнем полном поиске", refs["old_full_mae_best"]),
               ("Победитель прежнего полного отбора по RMSE", refs["rmse_winner"])]
    if v4:
        entries.append(("Исторический победитель V4 по RMSE", v4["cv"]))
    return mdtable(["Серия", "Конфигурация", "CV MAE, км", "CV RMSE, км", "Δ MAE¹, км", "Δ RMSE¹, км"],
                   ((label, f"`{r['name']}`", number(r["cv_mae_km"]), number(r["cv_rmse_km"]),
                     signed(winner["cv_mae_km"] - r["cv_mae_km"]), signed(winner["cv_rmse_km"]-r["cv_rmse_km"]))
                    for label, r in entries))


def test_section(results, v4):
    if results is None:
        return ["В экспорте пока есть только результаты кросс-валидации. Оценка на исторической тестовой части появится после финального прогона."]
    final = results["test"]
    model_v4 = f"V4 (`{v4['winner']}`)" if v4 else "V4, победитель по RMSE"
    series = [("Победитель текущего отбора по MAE", final),
              ("Победитель прежнего полного отбора по RMSE", results["historical_rmse_model"]),
              (model_v4, results["historical_v4"])]
    table = mdtable(["Модель", "Строк", "Тестовая MAE, км", "Тестовая RMSE, км", "Δ MAE¹, км", "Δ RMSE¹, км"],
                    ((label, number(r["n"], 0), number(r["mae_km"]), number(r["rmse_km"]),
                      signed(final["mae_km"]-r["mae_km"]), signed(final["rmse_km"]-r["rmse_km"])) for label, r in series))
    return [table,
            "¹ В столбцах Δ указана ошибка нового победителя минус ошибка модели в той же строке. "
            "Отрицательное значение означает улучшение по соответствующей метрике.",
            "Координаты тестовых объектов открываются для расчёта метрик после фиксации победителя и его прогнозов. "
            "Предыдущие отчёты уже содержали агрегаты этой же тестовой выборки. Она служит исторической контрольной выборкой, "
            "а полностью независимая оценка будущих версий потребует новых объектов."]


def render(lock, rows, front, results, provenance, replay, v4):
    win = rows[0]
    specs = lock["specifications"]
    top = win["name"]
    if results:
        test = results["test"]
        lead = (f"На историческом тесте из {number(test['n'],0)} объектов средняя ошибка составила **{number(test['mae_km'])} км**, "
                f"RMSE — **{number(test['rmse_km'])} км**.")
    else:
        lead = "Тестовые метрики в полученном экспорте ещё отсутствуют."
    groups = defaultdict(list)
    for row in rows:
        groups[family(specs[row["name"]])].append(row)
    tied = [r["name"] for r in rows if (r["cv_mae_km"], r["cv_rmse_km"]) == (win["cv_mae_km"],win["cv_rmse_km"])]
    doc = ["# Геолокация по эмбеддингам. Отбор по средней ошибке WGS84",
           f"Победитель по MAE — **`{top}`**. Среднее расстояние до истинной точки на внешних фолдах составило "
           f"**{number(win['cv_mae_km'])} км**, RMSE — **{number(win['cv_rmse_km'])} км**. {lead}",
           describe(specs[top]),
           f"Всего сравнено конфигураций — **{len(rows)}**, на границе Парето — **{len(front)}**. "
           "Параметры и все измеренные ошибки доступны в [полной таблице CSV](mae-cv-results.csv).",
           "## Объекты, обучение и отбор",
           "Эксперимент использует 250 000 объектов и два вектора изображения, GeoCLIP размерности 512 и StreetCLIP размерности 768. "
           "На разработку выделено 212 500 строк, на исторический тест — 37 500. В каждом из пяти внешних фолдов 170 000 строк "
           "доступны для обучения и 42 500 служат для проверки. Каждая строка разработки получает один прогноз вне обучения.",
           "Внутри каждого внешнего обучающего набора три разбиения дают прогнозы базовых моделей для обучения ансамбля. "
           "Обучаемые комбинации получают истинные координаты этих внутренних строк. Для объектов внешнего фолда доступны "
           "предсказания экспертов, вероятности географических ячеек и признаки уверенности. После выбора правила для обучаемого "
           "финального ансамбля используются прогнозы всех 212 500 объектов разработки, полученные вне их обучающих фолдов.",
           "Пусть $d_i$ — длина кратчайшей геодезической WGS84 от прогноза до известной координаты, в километрах. "
           "Тогда $\\mathrm{MAE}=\\sum_i d_i/n$ и $\\mathrm{RMSE}=\\sqrt{\\sum_i d_i^2/n}$. "
           "Например, промах на 1 000 км вносит в сумму расстояний в десять раз больше, чем промах на 100 км, "
           "а в сумму квадратов — в сто раз больше. Итоговую RMSE вычисляют из квадратов ошибок всех строк пяти фолдов.",
           "Победителем становится вариант с минимальной общей MAE. При равенстве служат последовательными критериями общая RMSE "
           "и идентификатор конфигурации. Поскольку те же внешние фолды определяют выбор среди многих вариантов, CV победителя "
           "может быть оптимистичной. Различия в сотые доли километра сами по себе не устанавливают устойчивого преимущества."]
    if len(tied) > 1:
        doc.append(f"Одинаковая пара итоговых ошибок встретилась у {len(tied)} конфигураций. Их порядок определён идентификаторами.")
    doc.extend(["## Какие комбинации сравнивались",
                "Методы используют одни и те же базовые прогнозы. Несколько строк одной семьи могут отличаться степенью сглаживания, "
                "порогом расстояния или способом получения координаты из общих обученных весов. Поэтому число конфигураций "
                "характеризует объём поиска, а часть вычислений разделяется между вариантами.",
                mdtable(["Семейство", "Вариантов", "Лидер семьи по MAE", "CV MAE, км", "CV RMSE, км"],
                        ((label,len(items),f"`{items[0]['name']}`",number(items[0]["cv_mae_km"]),number(items[0]["cv_rmse_km"]))
                         for label,items in sorted(groups.items(), key=lambda x: x[1][0]["cv_mae_km"]))),
                "Геометрическая медиана приближает точку с наименьшей суммой расстояний до гипотез с учётом их весов. "
                "Для правил по распределениям гипотезами служат ведущие ячейки, а для обучаемых медиан — координаты экспертов. "
                "Методы IRLS подбирают поправку в двух горизонтальных направлениях. Шесть итераций взвешенной регрессии снижают влияние "
                "больших ошибок, параметр сглаживания задаёт переход от почти квадратичной потери к почти линейной.",
                "Нейросетевые варианты выбирают одну координату или объединяют несколько. Например, модель CDF обучается вероятностям того, "
                "что ошибка каждой гипотезы превысит заданные расстояния. По этим вероятностям она оценивает среднюю ошибку кандидата "
                "для конкретного запроса. Сферические приближения при обучении проверяют общей метрикой на эллипсоиде WGS84.",
                "В базовом наборе используются следующие сохранённые прогнозы, в порядке входа в ансамбль.",
                "\n".join(f"{i}. `{name}`" for i, name in enumerate(lock["members"],1)),
                "Набор с медианным якорем добавляет первым исторический сферический медианный прогноз V4. "
                "Шлюз также получает семь прогнозов V3, следующие варианты обработки распределений и первые четыре центра "
                "по вероятности каждого из трёх классификаторов ячеек.",
                "\n".join(f"- `{name}`" for name in lock["posterior_gate_experts"]),
                "## Сравнение с предыдущими поисками",
                hist_table(lock, win, v4),
                "¹ Δ — ошибка нового победителя минус ошибка указанной модели на тех же фолдах. "
                "Исторический минимум MAE в V4 и исторический выбор V4 по RMSE относятся к двум отдельно обозначенным процедурам отбора.",
                "## Тест на прежнем разделении", *test_section(results, v4),
                "## MAE и RMSE по конфигурациям",
                "![Средняя и среднеквадратичная ошибка на внешних фолдах](mae-pareto.svg)",
                "Каждая точка соответствует конфигурации. Движение влево означает уменьшение средней ошибки, вниз — уменьшение RMSE. "
                "На границе Парето другие варианты не превосходят данную пару одновременно по двум метрикам с хотя бы одним строгим "
                "улучшением. Совпадающие точки видны в CSV отдельными строками.",
                "### Десять первых конфигураций по MAE",
                mdtable(["Место", "Конфигурация", "CV MAE, км", "CV RMSE, км", "Параметры и способ прогноза"],
                        ((i, f"`{r['name']}`", number(r["cv_mae_km"]),number(r["cv_rmse_km"]),describe(specs[r["name"]]))
                         for i,r in enumerate(rows[:10],1))),
                "### Победитель по внешним фолдам",
                mdtable(["Фолд", "MAE, км", "RMSE, км"], ((i,number(win["fold_mae_km"][i-1]),number(win["fold_rmse_km"][i-1])) for i in FOLDS)),
                "<details>\n<summary>Открыть результаты всех конфигураций по фолдам</summary>",
                "В каждой ячейке фолда указаны MAE / RMSE в километрах. CSV сохраняет исходную численную точность и полную спецификацию JSON.",
                mdtable(["Место", "Конфигурация", *(f"Фолд {i}" for i in FOLDS)],
                        ((i, f"`{r['name']}`", *(f"{number(a)} / {number(b)}" for a,b in zip(r["fold_mae_km"],r["fold_rmse_km"])))
                         for i,r in enumerate(rows,1))),
                "</details>", "## Воспроизводимость",
                f"Контрольная сумма разбиения SHA-256 — `{lock['split_sha256']}`. Контрольная сумма прежнего протокола отбора по RMSE — "
                f"`{lock['rmse_cv_sha256']}`."])
    if provenance:
        checkpoint = provenance.get("checkpoint")
        if checkpoint:
            doc.append(f"Файл параметров победителя — `{checkpoint}`, SHA-256 — `{provenance['checkpoint_sha256']}`.")
        elif "checkpoint" in provenance:
            doc.append("Для выбранного правила достаточно сохранённых компонентов ансамбля и его спецификации. Отдельные обучаемые параметры верхнего уровня отсутствуют.")
        if "fit_seconds" in provenance:
            doc.append(f"Финальная подготовка правила и расчёт тестовых прогнозов заняли {number(provenance['fit_seconds'],1)} с.")
    if results and "predictions_sha256" in results:
        doc.append(f"SHA-256 зафиксированного файла тестовых прогнозов — `{results['predictions_sha256']}`.")
    if replay:
        fields = [("median_km","медиана"),("mean_km","среднее"),("p95_km","95-й процентиль"),("max_km","максимум")]
        observed = [f"{label} {number(replay[key],6)} км" for key,label in fields if key in replay]
        if observed:
            size = f" на {number(replay['n'],0)} строках" if "n" in replay else ""
            doc.append("Расхождение повторного прогноза из сохранённых параметров с зафиксированными координатами"
                       f"{size} составило " + ", ".join(observed) + ".")
    doc.append("Протокол фолдов и правило отбора заданы в [mae_search.py](../src/screen/mae_search.py). "
               "Прогноз по ячейкам реализован в [mae_posterior.py](../src/screen/mae_posterior.py), "
               "устойчивая регрессия и геометрические медианы — в [mae_rules.py](../src/screen/mae_rules.py) "
               "и [mae_median.py](../src/screen/mae_median.py), а нейросетевые модели — в [mae_gate.py](../src/screen/mae_gate.py).")
    doc.append("Обучение и финальная проверка доступны в [ноутбуке MAE](../train250k_mae.ipynb). "
               "Связь между этапами показана на [схеме эксперимента](workflow-mae.svg). "
               "Повторить построение из экспортированных JSON можно командой "
               "`python src/report_mae.py --input artifacts/mae_cv --docs docs`. Для графика нужен Matplotlib. "
               "Предыдущий эксперимент описан в [отчёте об отборе по RMSE](report-500.md).")
    return "\n\n".join(doc) + "\n"


def generate(source, docs):
    path = source / "cv-summary.json"
    lock = load(path)
    rows = checked_rows(lock)
    results = optional(source / "results.json")
    provenance = optional(source / "final-provenance.json")
    replay = optional(source / "replay-check.json")
    check_sources(lock, hashlib.sha256(path.read_bytes()).hexdigest(), results, provenance)
    v4 = optional(REPO / "artifacts/v4/results.json")
    if v4 and v4.get("split_sha256") != lock["split_sha256"]:
        v4 = None
    front = pareto(rows)
    content = render(lock, rows, front, results, provenance, replay, v4)
    docs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=docs) as tmp:
        out = Path(tmp)
        write_csv(out / CSV_NAME, rows, lock["specifications"], front)
        chart(out / CHART_NAME, rows, lock["specifications"], front)
        (out / REPORT_NAME).write_text(content, encoding="utf-8")
        for name in (CSV_NAME, CHART_NAME, REPORT_NAME):
            (out / name).replace(docs / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    local = REPO.parent / "geocv-mae"
    parser.add_argument("--input", "--mae-dir", type=Path, default=local if local.is_dir() else REPO / "artifacts/mae_cv")
    parser.add_argument("--docs", "--out-dir", type=Path, default=REPO / "docs")
    args = parser.parse_args()
    generate(args.input, args.docs)
    print(args.docs / REPORT_NAME)


if __name__ == "__main__":
    main()
