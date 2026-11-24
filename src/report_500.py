"""Build the Russian experiment report from exported JSON, without ML dependencies."""

import argparse
import ast
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import report_tuning


FAMILIES = {
    "neural": "Прямая регрессия по эмбеддингам",
    "geo": "Географические модели и вероятностные ячейки",
    "retrieval": "Поиск по геопривязанным изображениям",
    "meta": "Метаобучение по прогнозам экспертов V3",
    "ensemble": "Ансамбли прогнозов и правила смешивания",
}
IDEAS = {
    "features": ("F", "Признаки"), "networks": ("N", "Нейросети"),
    "geo": ("G", "Географические модели"), "retrieval": ("R", "Поиск"),
    "ensembles": ("E", "Ансамбли и метаобучение"),
}
PAPER = {
    "distance_smoothing": "Географическое сглаживание целевой вероятности ячеек в духе PIGEON.",
    "distance_risk": "Классификатор ячеек со штрафом за ожидаемый квадрат сферического расстояния между классами.",
    "hierarchical_cells": "Совместное предсказание грубого и тонкого географических классов по обучающему иерархическому разбиению.",
    "fourier_location": "Сопоставление вектора изображения с многомасштабными признаками Фурье координат центров, по мотивам GeoCLIP.",
    "candidate_ranker": "Переранжирование координатных гипотез по прогнозируемой географической близости, по мотивам GeoRanker.",
}
COLUMNS = ["stage", "role", "rank", "configuration", "family", "feature_view",
           "feature_description_ru", "method", "method_description_ru", "n_evaluated",
           "mae_km", "rmse_km", "delta_reference_rmse_km", "mse_km2", "seconds_recorded",
           *[f"fold_{i}_rmse_km" for i in range(1, 6)], *report_tuning.COLUMNS]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def optional(path):
    return load(path) if path.is_file() else None


def atomic_text(path, content):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def num(value, precision=2):
    return "—" if value is None else f"{value:,.{precision}f}".replace(",", " ").replace(".", ",")


def diff(value):
    return "—" if value is None else ("+" if value > 0 else "") + num(value)


def cell(value):
    return str(value).replace("|", "&#124;").replace("\n", " ")


def table(lines, labels, rows):
    lines.extend(["", "| " + " | ".join(labels) + " |", "|" + "|".join("---" for _ in labels) + "|"])
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in rows)
    lines.append("")


def src_dict(path, name):
    if not path.is_file():
        return None
    for statement in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(statement, ast.Assign) and any(isinstance(x, ast.Name) and x.id == name for x in statement.targets):
            return ast.literal_eval(statement.value)
    return None


def registry(folder):
    known = optional(folder / "registry.json")
    if known is None:
        known = {}
        source = folder if (folder / "features.py").exists() else Path(__file__).resolve().parent
        for family in FAMILIES:
            models = source / ({"geo": "geo_models", "neural": "neural_models", "retrieval": "retrieval_models",
                                "meta": "meta_models", "ensemble": "ensemble_models"}[family] + ".py")
            views = (src_dict(models, "VIEWS") if family == "meta" else
                     src_dict(models, "GROUPS") if family == "ensemble" else
                     src_dict(source / "features.py", "FEATURES"))
            methods = src_dict(models, "METHODS")
            if not views or not methods:
                return {}
            for view, view_desc in views.items():
                for method, method_desc in methods.items():
                    known[f"{family}/{view}/{method}"] = dict(family=family, features=view, method=method,
                        feature_description=view_desc, model_description=method_desc)
    require(len(known) == 500, "Реестр должен содержать 500 конфигураций")
    require(Counter(v["family"] for v in known.values()) == Counter({k: 100 for k in FAMILIES}),
            "Некорректное число конфигураций семейств")
    for name, v in known.items():
        require(name == "/".join((v["family"], v["features"], v["method"])), f"Некорректный ключ {name}")
    return known


def record(stage, name, metrics, *, desc, view_desc="", family="", view="", method="", role="candidate",
           base=None, n=None, folds=False):
    mae = metrics.get("cv_mae_km", metrics.get("mae_km"))
    rmse = metrics.get("cv_rmse_km", metrics.get("rmse_km"))
    require(all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in (mae, rmse)),
            f"Неполные или некорректные метрики для {stage}/{name}")
    require(mae <= rmse + 1e-6, f"MAE больше RMSE для {name}")
    actual_n = metrics.get("n", n)
    require(isinstance(actual_n, int) and actual_n > 0 and (n is None or actual_n == n),
            f"Некорректное число строк для {stage}/{name}: {actual_n}")
    if "mse_km2" in metrics:
        require(math.isclose(metrics["mse_km2"], rmse**2, rel_tol=1e-6, abs_tol=1e-7),
                f"RMSE и MSE расходятся для {name}")
    out = dict(stage=stage, role=role, configuration=name, family=family, feature_view=view,
               feature_description_ru=view_desc, method=method or name, method_description_ru=desc,
               n_evaluated=actual_n, mae_km=mae, rmse_km=rmse,
               delta_reference_rmse_km=rmse-base if base is not None else None,
               mse_km2=metrics.get("mse_km2"), seconds_recorded=metrics.get("seconds"))
    if folds:
        values = metrics.get("fold_rmse_km", [])
        require(len(values) == 5 and all(math.isfinite(v) and v >= 0 for v in values),
                f"Ожидались пять внешних RMSE для {name}")
        require(math.isclose(math.sqrt(sum(v*v for v in values) / 5), rmse, rel_tol=1e-6, abs_tol=1e-6),
                f"Сводный RMSE пяти равновеликих фолдов расходится для {name}")
        out.update({f"fold_{i}_rmse_km": value for i, value in enumerate(values, 1)})
    return out


def ranked(rows):
    require(len(rows) == len({r["configuration"] for r in rows}), "Повторная конфигурация внутри одного этапа")
    rows.sort(key=lambda r: (r["rmse_km"], r["configuration"]))
    for idx, row in enumerate(rows, 1):
        row["rank"] = idx
    return rows


def screen_data(folder, catalog):
    summary = optional(folder / "summary.json")
    if summary:
        data = summary["candidates"]
        require(summary["n"] == len(data) == 500, "Неполный итог скрининга")
        require({r["name"] for r in data} == set(catalog), "Результаты скрининга не совпадают с реестром")
        ref = summary.get("reference_v4")
    else:
        data = [load(p) for p in sorted(folder.glob("*__*.json")) if not p.name.endswith(".error.json")]
        ref = optional(folder / "reference-v4.json")
    refrow = None if ref is None else record("screen", "reference_v4", ref, n=12500, role="reference",
                                           desc="Победитель V4, обученный на полной обучающей части этого внешнего фолда.")
    if data:
        require(bool(catalog), "Для измеренных строк скрининга нужен registry.json или исходные определения")
    out = []
    for r in data:
        name = r["name"]
        d = catalog[name]
        out.append(record("screen", name, r, desc=d["model_description"], view_desc=d["feature_description"],
                   family=d["family"], view=d["features"], method=d["method"], n=12500,
                   base=refrow["rmse_km"] if refrow else None))
    return ranked(out), refrow


def paper_desc(key):
    if key == "reference_v4":
        return "Контроль V4 на том же разбиении."
    suffix = "_plus_v4"
    base = key.removesuffix(suffix)
    return PAPER.get(base, key) + (" Новый прогноз получает 0,33 веса при смешивании с V4." if key.endswith(suffix) else "")


def full_desc(name, catalog, summary=None):
    summary = summary or {}
    if name.startswith("tuned:"):
        return report_tuning.describe(summary["tuned_fusion"]["rules"][name])
    for prefix, key in (("strong:", "strong_fusion"), ("diverse:", "diversity_fusion")):
        if name.startswith(prefix):
            return summary[key]["rules"][name]
    if name == "reference_v4":
        return "Повторно обученная контрольная конфигурация V4."
    if name.startswith("literature:"):
        return paper_desc(name.partition(":")[2])
    if name.startswith("blend:"):
        return "Сферическая интерполяция от V4 к финалисту с долей финалиста 0,33. " + full_desc(name.partition(":")[2], catalog, summary)
    if name.startswith("joint:"):
        method = name.partition(":")[2]
        desc = next((v["model_description"] for v in catalog.values() if v["family"] == "ensemble" and v["method"] == method), method)
        return "Совместное правило по специалистам. " + desc + " Затем добавлены метамодель и V4 с фиксированными долями."
    if name in catalog:
        d = catalog[name]
        return d["feature_description"] + " " + d["model_description"]
    return "Идентификатор из сводки полной перекрёстной проверки."


def cv_data(stage, summary, descriptions, reference, families=None):
    if not summary:
        return []
    data = summary["candidates"]
    by_name = {x["name"]: x for x in data}
    require(reference in by_name, f"Отсутствует контроль {reference} в {stage}")
    base = by_name[reference].get("cv_rmse_km", by_name[reference].get("rmse_km"))
    out = []
    for name, r in by_name.items():
        family = r.get("family", "") if families is None else families.get(name, "")
        out.append(record(stage, name, r, desc=descriptions(name), family=family, n=212500, folds=True,
                          base=base, role="reference" if name == reference else "candidate"))
    require(len(out) == len(data), f"Повторные записи в {stage}")
    out = ranked(out)
    if "winner" in summary:
        require(by_name[summary["winner"]].get("cv_rmse_km", by_name[summary["winner"]].get("rmse_km"))
                == out[0]["rmse_km"], f"Указанный победитель расходится с таблицей {stage}")
    return out


def history_data(folder, cv, stage, desc):
    result = optional(folder / "results.json")
    if result is None:
        return None
    require(cv is not None and result["winner"] == cv["winner"], f"Итоговый победитель не совпадает с CV этапа {stage}")
    if result.get("split_sha256") and cv.get("split_sha256"):
        require(result["split_sha256"] == cv["split_sha256"], f"Итоговый тест и CV имеют разные разбиения в {stage}")
    return record(stage, result["winner"], result["test"], desc=desc(result["winner"]), n=37500, role="selected_test")


def idea_data(folder):
    rows = []
    for basename, (prefix, _) in IDEAS.items():
        parts = load(folder / (basename + ".json"))
        require([v["id"] for v in parts] == [f"{prefix}{i:03}" for i in range(1, 101)],
                f"Неполная или повторная нумерация гипотез {prefix}")
        for v in parts:
            require(all(isinstance(v.get(k), str) and v[k].strip() for k in
                        ("id", "title", "mechanism", "hypothesis", "required_inputs", "budget")),
                    f"Неполная запись {v.get('id')}")
            require(v["budget"] in {"quick", "medium", "heavy"}, f"Неизвестный бюджет {v['id']}")
        rows.extend(parts)
    require(len({r["title"].casefold().strip() for r in rows}) == 500, "Повтор названия в концептуальном каталоге")
    return rows


def cv_table(lines, rows):
    table(lines, ["№", "Конфигурация", "Что проверено", "MAE, км", "RMSE, км", "Δ RMSE, км", "RMSE фолдов 1–5, км"],
          ([r["rank"], "`"+r["configuration"]+"`", r["method_description_ru"], num(r["mae_km"]),
            num(r["rmse_km"]), diff(r["delta_reference_rmse_km"]),
            " · ".join(num(r[f"fold_{i}_rmse_km"], 1) for i in range(1,6))] for r in rows))


def screen_sections(lines, screen, ref, groups):
    lines.extend(["## Скрининг конфигураций", "",
        "Каждая строка задаётся семейством, представлением входа или набором экспертов и способом построения ответа. "
        "В заголовке каждой таблицы указано общее представление, в строках — конкретный метод. "
        "Номер показывает место среди всех загруженных конфигураций скрининга."])
    if not screen:
        lines.extend(["", "В локальном наборе артефактов пока нет измерений этого этапа.", ""])
        return
    if ref:
        lines.extend(["", f"Контрольный V4 на тех же 12 500 изображениях получил MAE {num(ref['mae_km'])} км и "
            f"RMSE {num(ref['rmse_km'])} км. Δ в таблицах равна RMSE конфигурации минус RMSE этого контроля."])
    else:
        lines.extend(["", "Контрольный результат V4 на этих 12 500 изображениях отсутствует в загруженных файлах."])
    lines.extend(["", "Прямые модели обучены на 60 000 примерах. Метамодели и ансамбли используют отдельные 20 000 "
        "калибровочных строк. Признаки V3 получены из ранее обученных моделей внешнего фолда, а контрольный V4 "
        "использует все 170 000 обучающих строк фолда."])
    summary = []
    for key, title in FAMILIES.items():
        batch = [r for r in screen if r["family"] == key]
        if batch:
            best = batch[0]
            summary.append([title, len(batch), "`"+best["configuration"]+"`", num(best["rmse_km"]),
                            num(statistics.median(r["rmse_km"] for r in batch)),
                            sum(r["rmse_km"] < ref["rmse_km"] for r in batch) if ref else "—"])
    table(lines, ["Семейство", "Измерено", "Лидер по RMSE", "Лучший RMSE, км", "Медианный RMSE, км", "Лучше V4"], summary)
    lines.extend(["Время приведено в секундах как его записал исполнитель. Для поиска общая длительность десяти "
        "декодеров разделена на десять. Время ансамбля охватывает объединение уже готовых прогнозов. "
        "Подготовка признаков, прежнее обучение V3 и часть общих расчётов вынесены за пределы этих замеров.", ""])
    for family, title in FAMILIES.items():
        batch = [r for r in screen if r["family"] == family]
        if not batch:
            continue
        lines.extend([f"### {title}", ""])
        for view in sorted({r["feature_view"] for r in batch}):
            part = [r for r in batch if r["feature_view"] == view]
            lines.extend([f"#### {family}/{view}", "", part[0]["feature_description_ru"]])
            if family == "ensemble" and groups and view in groups:
                lines.extend(["", "Участники по калибровочной выборке — " + ", ".join("`"+v+"`" for v in groups[view]) + "."])
            table(lines, ["№", "Метод", "Механизм", "MAE, км", "RMSE, км", "Δ RMSE, км", "Время, с"],
                  ([r["rank"], "`"+r["method"]+"`", r["method_description_ru"], num(r["mae_km"]),
                    num(r["rmse_km"]), diff(r["delta_reference_rmse_km"]), num(r["seconds_recorded"])] for r in part))


def read_diversity_top(folder):
    sel=load(folder/"selection.json")
    query=optional(folder/"results.json")
    text=(f"На 20 000 калибровочных строках по вкладу в прогноз V4 ранжированы {sel['raw_models']} базовых конфигураций. "
          f"Жадный отбор сохранил {len(sel['selected'])} специалистов.")
    if query:
        joint=query["greedy_joint"]
        text += (f" На 12 500 контрольных строках после фиксации весов ансамбль получил RMSE {num(joint['rmse_km'])} км, "
                 f"а его соответствующий исходный V4 — {num(query['v4']['rmse_km'])} км. "
                 f"MAE изменилась с {num(query['v4']['mae_km'])} до {num(joint['mae_km'])} км.")
    example = next((r for r in sel["candidate_calibration"]
                    if r["name"] == "retrieval/shrink_whiten/ret_kernel_mode"), None)
    if example:
        text += (f" Поисковая модель `{example['name']}` заняла {example['standalone_rank']}-е место "
                 f"самостоятельно с RMSE {num(example['cal_standalone']['rmse_km'])} км. При добавлении к V4 "
                 f"она снизила калибровочный RMSE на {num(example['cal_gain_signed_km'])} км. Корреляция "
                 f"её направленных ошибок с V4 составила {num(example['error_corr_vector'],3)}.")
    return text


def generate(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    catalog = registry(args.screen_dir)
    screen, screen_ref = screen_data(args.screen_dir, catalog)
    groups = optional(args.screen_dir / "ensemble-components.json")
    v4_summary = load(args.v4_dir / "cv-summary.json")
    v4_descriptions = load(args.v4_dir / "descriptions-ru.json")
    require(len(v4_descriptions) == v4_summary["n_new_variants"] == 50, "Ожидалось 50 новых вариантов V4")
    require(set(v4_descriptions) == {r["name"] for r in v4_summary["candidates"] if r["name"] != "incumbent_v3"},
            "Описания V4 расходятся с результатами")
    v4desc = lambda key: "Контрольный победитель V3." if key == "incumbent_v3" else v4_descriptions[key]
    v4 = cv_data("v4_cv", v4_summary, v4desc, "incumbent_v3")
    old = history_data(args.v4_dir, v4_summary, "v4_test", v4desc)
    lit_summary = optional(args.research_dir / "literature-summary.json")
    full_summary = optional(args.full_dir / "cv-summary.json")
    ideas = idea_data(args.research_dir)
    require(all(s["split_sha256"] == v4_summary["split_sha256"] for s in (lit_summary, full_summary) if s),
            "Сигнатуры разбиения сравниваемых полных CV расходятся")
    papers = cv_data("literature_cv", lit_summary, paper_desc, "reference_v4")
    if papers:
        expected_papers = {"reference_v4", *PAPER, *(n + "_plus_v4" for n in PAPER)}
        require({r["configuration"] for r in papers} == expected_papers, "Неполная таблица проверки работ")
    full = cv_data("finalists_cv", full_summary, lambda n: full_desc(n, catalog, full_summary), "reference_v4")
    final = history_data(args.full_dir, full_summary, "final_test", lambda n: full_desc(n, catalog, full_summary))
    tuned = [row for row in full if row["configuration"].startswith("tuned:")]
    tuning_specs = (full_summary or {}).get("tuned_fusion", {}).get("rules", {})
    if tuned or tuning_specs:
        require(len(tuned) == len(tuning_specs) == full_summary["tuned_fusion"]["n_variants"],
                "Количество измерений дополнительного подбора расходится с реестром")
        require({r["configuration"] for r in tuned} == set(tuning_specs), "Неполный реестр подбора ансамблей")
        for row in tuned:
            row["stage"] = "ensemble_tuning_cv"
            row["family"] = "tuned_fusion"
            row.update(report_tuning.csv_params(tuning_specs[row["configuration"]]))
    if final and old:
        final["delta_reference_rmse_km"] = final["rmse_km"] - old["rmse_km"]
    stage_counts = [len(screen), len(v4_descriptions), len(papers) - 1 if papers else 0]
    lines = ["# Геолокация по двум эмбеддингам — скрининг и полная проверка", "", "## Результаты", ""]
    if final:
        lines += [f"По пяти фолдам выбрана конфигурация `{final['configuration']}`. На историческом тесте из "
            f"37 500 изображений она получила **RMSE {num(final['rmse_km'])} км** и **MAE {num(final['mae_km'])} км**.", ""]
    elif full:
        lines += [f"Полная проверка выбрала `{full[0]['configuration']}`.", ""]
    if screen:
        first = screen[0]
        lines += [f"На 12 500 изображениях скрининга минимальный RMSE среди {len(screen)} измеренных конфигураций "
                  f"получила `{first['configuration']}` — **{num(first['rmse_km'])} км**. Её MAE составила "
                  f"{num(first['mae_km'])} км.", ""]
        if screen_ref:
            improved = sum(r["rmse_km"] < screen_ref["rmse_km"] for r in screen)
            lines += [f"V4 на этих же изображениях получил RMSE {num(screen_ref['rmse_km'])} км и MAE "
                f"{num(screen_ref['mae_km'])} км. У лидера скрининга Δ RMSE к V4 составила "
                f"{diff(first['rmse_km'] - screen_ref['rmse_km'])} км. Число конфигураций с меньшим RMSE, чем у V4, — "
                f"{improved} из {len(screen)}.", ""]
    if full:
        winner = full[0]
        control = next(r for r in full if r["role"] == "reference")
        lines += [f"На полном CV у выбранной конфигурации RMSE {num(winner['rmse_km'])} км "
            f"и MAE {num(winner['mae_km'])} км. Относительно контроля V4 на этом этапе "
            f"Δ RMSE равна {diff(winner['rmse_km'] - control['rmse_km'])} км и "
            f"Δ MAE равна {diff(winner['mae_km'] - control['mae_km'])} км.", ""]
    if papers:
        best_paper = papers[0]
        ctrl = next(x for x in papers if x["role"] == "reference")
        lines += [f"В проверке идей из публикаций лучший результат по RMSE показала `{best_paper['configuration']}`. "
            f"Относительно V4 на тех же пяти фолдах её Δ RMSE составила {diff(best_paper['rmse_km']-ctrl['rmse_km'])} км, "
            f"Δ MAE — {diff(best_paper['mae_km']-ctrl['mae_km'])} км.", ""]
    if old and final is None:
        lines += [f"Для ориентира, исторический тест V4 на 37 500 изображениях дал RMSE {num(old['rmse_km'])} км "
            f"и MAE {num(old['mae_km'])} км у `{old['configuration']}`. "
            "Итоговый тест нового поиска пока отсутствует в загруженных артефактах.", ""]
    lines += ["Все измеренные конфигурации с описаниями и численными результатами доступны в [CSV](results-500.csv). "
              "Самостоятельные исследовательские предложения собраны в [каталоге гипотез](ideas-500.json).", "",
              "## Данные, метрики и выбор модели", "",
              "Каждое из 250 000 изображений представлено 512 компонентами GeoCLIP и 768 компонентами StreetCLIP. "
              "Историческое стратифицированное разбиение выделяет 212 500 изображений для разработки и 37 500 для теста. "
              "Пять внешних фолдов разработки содержат по 42 500 контрольных и 170 000 обучающих строк. "
              "Стратификация по стране сохраняет доли стран, и близкие географические точки могут попасть в разные фолды.", "",
              "Для каждого примера единичный вектор ECEF переводится в геодезические широту и долготу. "
              "Расстояние до истинных координат вычисляется обратной геодезической задачей на эллипсоиде WGS84. "
              "MAE — среднее расстояние в километрах. RMSE — квадратный корень из среднего квадрата расстояний. "
              "Большие промахи влияют на выбор по RMSE сильнее. Общий RMSE пяти фолдов рассчитывается по всем квадратам "
              "ошибок с учётом размера фолдов. Внутренние функции обучения могут использовать сферические приближения.", "",
              "Скрининг использует внешний фолд № 1, в коде `fold=0`, и зерно `904317`. "
              "Из обучающей части случайно выбраны непересекающиеся 60 000 строк для базовых моделей и 20 000 для "
              "калибровки. Из контрольной части случайно выбраны 12 500 для общего рейтинга. Метапризнаки V3 для "
              "обучающих строк получены вне соответствующего внутреннего фолда. Для контрольных строк их базовые "
              "модели обучены в пределах внешней обучающей части. В ансамблях состав группы и веса выбираются по "
              "20 000 калибровочным примерам. Поиск и нейросетевые операции используют GPU; часть предобработки, "
              "геодезических расчётов и объединения прогнозов выполняется на CPU.", "",
              "Финалисты отбираются как объединение общего топ-5 скрининга и лидеров каждого из пяти семейств. "
              "Повторяющиеся кандидаты учитываются один раз. Их повторно сравнивают на полных пяти фолдах. "
              "В эти фолды входят изображения, участвовавшие в отборе при скрининге, поэтому оценка финалистов "
              "характеризует внутренний выбор модели. Исторические тестовые агрегаты прошлых версий были известны до "
              "этой серии экспериментов. Это ограничивает независимость повторной оценки на тех же 37 500 изображениях.", ""]
    lines += ["## Исторический тест", ""]
    test_rows = [r for r in (old, final) if r]
    if test_rows:
        table(lines, ["Версия", "Выбранный метод", "Строк", "MAE, км", "RMSE, км", "Δ RMSE к V4, км"],
              (["V4" if r is old else "Новый поиск", "`"+r["configuration"]+"`", num(r["n_evaluated"], 0),
                num(r["mae_km"]), num(r["rmse_km"]), "контроль" if r is old else diff(r["delta_reference_rmse_km"])]
               for r in test_rows))
    else:
        lines += ["В загруженных материалах отсутствуют файлы итоговой тестовой оценки.", ""]
    if final:
        final_info = load(args.full_dir / "results.json")
        if final_info.get("predictions_sha256"):
            lines += [f"Контрольная сумма SHA-256 файла тестовых прогнозов — `{final_info['predictions_sha256']}`.", ""]
        if final_info.get("cv_sha256"):
            lines += [f"Контрольная сумма зафиксированного решения CV — `{final_info['cv_sha256']}`.", ""]
    lines += ["## Пять фолдов финалистов", "",
              "Для совместных правил базовые специалисты прямой регрессии, ячеек, поиска и модели из публикаций "
              "дают прогнозы на трёх вложенных разбиениях для обучения весов. В совместном прогнозе их совокупная "
              "номинальная доля составляет 1/2. Метаэксперт и контрольный V4 добавляются последовательной сферической "
              "интерполяцией с номинальными долями 1/6 и 1/3. У отдельных кандидатов `blend:` доля финалиста "
              "в интерполяции с V4 составляет 0,33. Δ в этом разделе считается к повторно оценённому V4.", ""]
    selection = full_summary.get("selection", {}) if full_summary else (optional(args.full_dir / "selection.json") or {})
    if selection.get("finalists"):
        lines += ["Отобранные конфигурации — " + ", ".join("`"+k+"`" for k in selection["finalists"]) + ".", ""]
    if selection.get("joint_members"):
        lines += ["Компоненты совместных правил — " + ", ".join("`"+k+"`" for k in selection["joint_members"]) + ".", ""]
    if full:
        for label,title in (("strong_fusion", "Усиленные ансамбли"), ("diversity_fusion", "Ансамбли с дополнительными специалистами")):
            meta = full_summary.get(label)
            if meta:
                lines += [title + " — " + ", ".join("`"+k+"`" for k in meta["members"]) + ".", ""]
        cv_table(lines, [r for r in full if r["stage"] != "ensemble_tuning_cv"])
    else:
        folds_done = len(list(args.full_dir.glob("fold-[0-4].json")))
        lines += [f"Сводка этого этапа пока отсутствует. В локальном каталоге результатов сохранено {folds_done} из 5 "
                  "отдельных файлов фолдов. Общий рейтинг появится при загрузке полной сводки.", ""]
    if tuned:
        metadata = full_summary["tuned_fusion"]
        best = tuned[0]
        previous = metadata.get("previous_cv_winner")
        lines += ["## Детальный подбор параметров ансамбля", "",
            f"Измерено {len(tuned)} дополнительных конфигураций, каждая на всех пяти внешних фолдах. "
            f"Лучший вариант этой сетки — `{best['configuration']}`, его RMSE {num(best['rmse_km'])} км "
            f"и MAE {num(best['mae_km'])} км. Номер строки в таблице показывает место в общем рейтинге полной CV.", ""]
        if previous:
            before = next((r for r in full if r["configuration"] == previous), None)
            if before:
                lines += [f"У лучшей конфигурации очередной сетки Δ RMSE к предыдущему лидеру полной CV "
                    f"`{previous}` составляет {diff(best['rmse_km']-before['rmse_km'])} км. "
                    "Эта разница измерена на фолдах, участвовавших в подборе.", ""]
        lines += ["Опорная точка — V4. Геодезические смещения остальных моделей вычисляются на эллипсоиде WGS84. "
            "Обозначение core соответствует V4, метамодели, модели со штрафом расстояния и поисковому специалисту; "
            "all добавляет отобранных дополнительных специалистов. Коэффициенты Ridge могут быть общими для двух "
            "координат или отдельными для востока и севера. α задаёт относительную силу притяжения к исходным весам "
            "с учётом масштаба матрицы признаков. Параметр хвоста усиливает обучающие строки с большими ошибками V4. "
            "Предел шага обрезает длину поправки перед переносом на эллипсоид. При оценке все примеры имеют равный вес.", "",
            "Члены полного набора — " + ", ".join("`"+v+"`" for v in metadata["members"]) + ".", ""]
        if report_tuning.chart(tuned,tuning_specs,args.out_dir/"tuning-rmse.svg"):
            lines += ["![RMSE ансамбля при изменении регуляризации](tuning-rmse.svg)", "",
                "На графике фиксированы географический контекст и расхождение экспертов, нулевое усиление хвоста "
                "и предел шага 20 000 км. Каждая линия меняет только α.", ""]
        cv_table(lines, tuned)
    diversity_path = args.diversity_dir
    if (diversity_path/"selection.json").is_file():
        from report_diversity import build as build_diversity
        if build_diversity(diversity_path,args.out_dir,full_summary):
            front = read_diversity_top(diversity_path)
            lines += ["## Дополняющие ошибки специалистов", "", front, "",
                "У каждой строки калибровки признаки V3 получены вне её собственного внутреннего фолда. "
                "Базовые модели, создавшие признаки для других строк, могли видеть её метку. В отдельной "
                "пятифолдовой проверке усиленных и дополненных ансамблей для обучения правил объединения "
                "используются прогнозы специалистов на вложенных контрольных блоках.", "",
                "По каждому из 300 исходных специалистов доступны корреляции знаковых ошибок, индивидуальный "
                "результат и прирост от смешивания с V4. [Подробный анализ](diversity-analysis.md) · "
                "[Числа в CSV](diversity-results.csv).", ""]
    if papers:
        lines += ["## Проверка идей из публикаций", "",
            "Пять адаптаций обучены на готовых визуальных эмбеддингах. Для каждой оценены самостоятельный прогноз "
            "и фиксированная сферическая смесь с V4. Δ рассчитана к V4 на тех же пяти фолдах. Географическое "
            "сглаживание и штраф расстояния относятся к разным целям обучения. Ранжировщик оценивает кандидатов "
            "по диагностическим признакам и расстояниям обучающих строк. Источники и различия реализации описаны "
            "в [обзоре публикаций](literature-notes.md).", ""]
        cv_table(lines, papers)
    screen_sections(lines, screen, screen_ref, groups)
    lines += ["## Предыдущие 50 конфигураций V4", "",
        "Все 50 новых вариантов V4 оценены на каждом из тех же пяти внешних фолдов. Отдельная контрольная "
        "строка показывает победителя V3. Δ в этой таблице рассчитана к V3. Выигравший `cal_interaction` "
        "оценивает смещения шести экспертов от многомасштабного прогноза и умножает их на признаки уверенности, "
        "разброса и согласия. Регуляризованная регрессия переносит итоговую поправку на WGS84. "
        "Подробности предыдущей серии приведены в [отчёте V4](report-v4.md).", ""]
    cv_table(lines, v4)
    lines += ["## Каталог исследовательских предложений", "",
        "Каталог содержит 500 сформулированных гипотез. У каждого предложения есть идентификатор, механизм, "
        "ожидаемый эффект, необходимые входы и оценка бюджета. Метрики в этом отчёте привязаны к идентификаторам "
        "исполненных конфигураций. Например, поисковая гипотеза с префиксом `R` и строка рейтинга "
        "`retrieval/<представление>/<агрегатор>` принадлежат разным схемам идентификаторов.", ""]
    table(lines, ["Направление", "Идентификаторы", "Предложений"], ([title, f"{prefix}001–{prefix}100", 100] for prefix, title in IDEAS.values()))
    lines += ["[Скачать каталог из 500 гипотез в JSON](ideas-500.json).", "",
        "## Воспроизведение отчёта", "",
        "Для сборки из экспортированных файлов нужен Python 3.10 или новее. График тонкой настройки строится при наличии Matplotlib. В каталог скрининга поместите "
        "`summary.json`, `registry.json` и, при наличии, `ensemble-components.json`. В каталог V4 поместите "
        "сводку CV, русские описания и исторический тест. Каталог исследований содержит пять JSON-каталогов гипотез "
        "и сводку проверки идей из публикаций. Для полного результата нужны `cv-summary.json` и `results.json` "
        "из каталога финалистов.", "", "```bash",
        "python src/report_500.py --screen-dir artifacts/screen500 --v4-dir artifacts/v4 \\",
        "  --research-dir artifacts/v5 --full-dir artifacts/full500 --diversity-dir artifacts/diversity500 --out-dir docs",
        "```", "", "Числа в Markdown округлены до двух знаков, значения по фолдам — до одного. "
        "CSV хранит численную точность исходных JSON. Пустые поля обозначают показатели, "
        "которые не были опубликованы на соответствующем этапе.", ""]
    workflow = args.out_dir / "workflow-final500.svg"
    notebook = args.out_dir.parent / "train250k_final500.ipynb"
    if workflow.is_file() and notebook.is_file():
        lines += ["Процесс обучения, отбора и тестирования показан на [схеме эксперимента](workflow-final500.svg). "
                  "Запуск доступен в [ноутбуке эксперимента](../train250k_final500.ipynb).", ""]
    if (args.out_dir / "architecture-final.md").is_file():
        lines += ["Компоненты ансамблей и тракт прогноза описаны в [разборе итоговой архитектуры](architecture-final.md).", ""]
    all_rows = screen + ([screen_ref] if screen_ref else []) + v4 + papers + full + test_rows
    csv_path = args.out_dir / "results-500.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    temporary.replace(csv_path)
    atomic_text(args.out_dir / "ideas-500.json", json.dumps(ideas, ensure_ascii=False, indent=2) + "\n")
    atomic_text(args.out_dir / "report-500.md", "\n".join(lines))
    print(f"Скрининг: {stage_counts[0]}/500; V4: {stage_counts[1]} новых; адаптации публикаций: {stage_counts[2]}; "
          f"финалисты и смеси: {len(full)}; итоговый тест нового поиска: {bool(final)}.")


def main():
    repo = Path(__file__).resolve().parent.parent
    work = repo.parent
    def source(local_name, exported_name):
        local = work/local_name
        return local if local.is_dir() else repo/"artifacts"/exported_name
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", type=Path, default=source("geocv-screen", "screen500"))
    parser.add_argument("--v4-dir", type=Path, default=source("geocv-v4", "v4"))
    parser.add_argument("--research-dir", type=Path, default=source("geocv-v5", "v5"))
    parser.add_argument("--full-dir", type=Path, default=source("geocv-full", "full500"))
    parser.add_argument("--diversity-dir", type=Path, default=source("geocv-diversity", "diversity500"))
    parser.add_argument("--out-dir", type=Path, default=repo/"docs")
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
