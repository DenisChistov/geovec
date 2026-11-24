"""Build the Russian diversity analysis using measured run artifacts and the standard library."""
import csv
import io
import json
import math
from html import escape
from pathlib import Path


FAMILIES = {
    "neural": ("Нейросети", "#7653a3"),
    "geo": ("Геоклассификаторы", "#087f76"),
    "retrieval": ("Поиск по эталонам", "#bf6521"),
}


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x, digits=2):
    if x is None:
        return "—"
    return f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")


def _flatten(record, prefix=""):
    result = {}
    for key, val in record.items():
        field = f"{prefix}{key}"
        if isinstance(val, dict):
            result.update(_flatten(val, field + "."))
        elif isinstance(val, list):
            result[field] = json.dumps(val, separators=(",", ":"), ensure_ascii=False)
        else:
            result[field] = val
    return result


def _ticks(lo, hi):
    if hi <= lo:
        lo, hi = lo - 1, hi + 1
    target = (hi - lo) / 5
    power = 10**math.floor(math.log10(target))
    step = next(s * power for s in (1, 2, 5, 10) if s * power >= target)
    a, b = math.floor(lo / step), math.ceil(hi / step)
    return [i * step for i in range(a, b + 1)]


def _scatter(rows, baseline, highlighted, weak):
    xvals = [r["cal_standalone"]["rmse_km"] for r in rows]
    yvals = [r["cal_gain_signed_km"] for r in rows]
    xt = _ticks(min(*xvals, baseline), max(*xvals, baseline))
    yt = _ticks(min(*yvals, 0), max(*yvals, 0))
    x0, x1, y0, y1 = 102, 907, 350, 39
    def sx(v):
        return x0 + (x1 - x0) * (v - xt[0]) / (xt[-1] - xt[0])
    def sy(v):
        return y0 + (y1 - y0) * (v - yt[0]) / (yt[-1] - yt[0])
    def line(xa, ya, xb, yb, color="#dfe4e6", extra=""):
        return f'<line x1="{xa:.3f}" y1="{ya:.3f}" x2="{xb:.3f}" y2="{yb:.3f}" stroke="{color}" {extra}/>'
    def label(x, y, s, attr=''):
        return f'<text x="{x:.2f}" y="{y:.2f}" {attr}>{escape(str(s))}</text>'
    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 466" width="960" height="466" role="img" aria-labelledby="t d">',
           '<title id="t">Самостоятельная ошибка и выигрыш при добавлении к V4</title>',
           '<desc id="d">Каждая точка соответствует одной базовой модели. Выше нуля находятся смеси с меньшим калибровочным RMSE, чем у V4.</desc>',
           '<rect width="960" height="466" fill="white"/>',
           '<g font-family="DejaVu Sans,Arial,sans-serif" fill="#23343c" font-size="13">']
    for t in xt:
        out += [line(sx(t), y1, sx(t), y0), label(sx(t), y0+20, _fmt(t, 0), 'text-anchor="middle"')]
    for t in yt:
        digits = max(0, -math.floor(math.log10(yt[1]-yt[0])))
        out += [line(x0, sy(t), x1, sy(t)), label(x0-10, sy(t)+4, _fmt(t, digits), 'text-anchor="end"')]
    out += [line(x0, sy(0), x1, sy(0), "#536873", 'stroke-width="1.5"'),
            line(sx(baseline), y1, sx(baseline), y0, "#536873", 'stroke-dasharray="5 4"'),
            label((x0+x1)/2, y0+45, "Самостоятельный RMSE на калибровке, км", 'text-anchor="middle"'),
            '<text transform="translate(25 191) rotate(-90)" text-anchor="middle">Снижение RMSE при добавлении к V4, км</text>']
    for r in sorted(rows, key=lambda v: (v["name"] in highlighted or v["name"] == weak, v["name"])):
        name = r["name"]
        x, y = sx(r["cal_standalone"]["rmse_km"]), sy(r["cal_gain_signed_km"])
        color = FAMILIES[name.split("/")[0]][1]
        title = f'{name} · RMSE {_fmt(r["cal_standalone"]["rmse_km"])} км · выигрыш {_fmt(r["cal_gain_signed_km"])} км'
        if name in highlighted:
            out.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="6.4" fill="white" stroke="#152b38" stroke-width="2"/>')
        if name == weak:
            out.append(f'<path d="M {x:.3f} {y-9:.3f} l 9 9 -9 9 -9 -9 Z" fill="none" stroke="#a1164a" stroke-width="2"/>')
        out.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="3.3" fill="{color}" fill-opacity=".82"><title>{escape(title)}</title></circle>')
    for i, (name, color) in enumerate(FAMILIES.values()):
        x = 100 + 211*i
        out += [f'<circle cx="{x}" cy="425" r="4" fill="{color}"/>', label(x+10, 430, name)]
    out += [line(724, 425, 750, 425, "#536873", 'stroke-dasharray="5 4"'), label(759,430,"RMSE базы V4"),
            '<circle cx="100" cy="449" r="5" stroke="#152b38" stroke-width="2" fill="none"/>',
            label(110,454,"Отобран в совместную смесь или полную CV")]
    if weak:
        out += ['<path d="M 512 442 l 7 7 -7 7 -7 -7 Z" stroke="#a1164a" stroke-width="2" fill="none"/>',
                label(524,454,"Лучший выигрыш среди уступающих V4 в одиночку")]
    return "\n".join(out + ['</g></svg>'])


def _table(headers, rows):
    return ["| " + " | ".join(map(str, headers)) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows]]


def _csv(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _extended(lines, directory, supplement):
    output, table = [], []
    labels = {"raw": "Базовые модели", "meta": "Метамодели", "v4": "Кандидаты V4"}
    for phase, label in labels.items():
        path = directory / "extended" / (phase + ".json")
        if not path.is_file():
            continue
        obj = _read(path)
        meta, records = obj["metadata"], obj["rows"]
        if (meta["phase"] != phase or meta["fixed_candidate_weight"] != .25
                or not meta["exploratory_only"] or len(records) != obj["n_candidates"]
                or len(records) != len({r["name"] for r in records}) or not records):
            raise ValueError(f"Incomplete post-lock diagnostics for {phase}")
        reference = meta["reference"]
        if any(row["reference"] != reference for row in records):
            raise ValueError(f"Mixed reference fits in {phase}")
        best = min(records, key=lambda r: (r["fixed_quarter_blend"]["rmse_km"], r["name"]))
        num_better = sum(r["quarter_blend_gain_vs_reference_km"] > 0 for r in records)
        table.append([label, len(records), num_better, _fmt(obj["reference_metrics"]["rmse_km"]),
                      f'`{best["name"]}`', _fmt(best["fixed_quarter_blend"]["rmse_km"]),
                      _fmt(best["quarter_blend_gain_vs_reference_km"]), _fmt(best.get("error_corr_vector"),3)])
        for row in records:
            output.append({"diagnostic_phase": phase, "evaluation_n": meta["n"], "candidate_weight": .25,
                           "reference_fit": reference, "selection_sha256": meta["selection_sha256"],
                           "split_sha256": meta["split_sha256"],
                           **_flatten(obj["reference_metrics"], "reference_metrics."), **_flatten(row)})
    if table:
        lines += ["", "## Фиксированная доля кандидата после отбора", "",
                  "Для каждого кандидата точку V4 сдвинули на 25 % геодезического расстояния к его прогнозу. Таблица показывает лучший результат каждой завершённой группы на повторно использованной подвыборке из 12 500 снимков. Это исследовательское сравнение после фиксации участников полной CV.", "",
                  "Для базовых моделей и метамоделей контроль V4 обучен на 60 000 строках. Для кандидатов V4 использован контроль, обученный на 170 000 строках внешнего фолда. Выигрыш отсчитывается от контроля своей группы.", "",
                  *_table(["Группа", "Проверено", "С улучшением", "RMSE контроля, км", "Лучшая добавка", "RMSE смеси, км", "Выигрыш, км", "Корреляция векторов ошибок"],
                          table), "", "Для каждой конфигурации опубликованы контроль, самостоятельные показатели, результат смеси и показатели согласия ошибок. [Все измерения после фиксации выбора](diversity-extended-results.csv)"]
    if supplement:
        replay = supplement.get("screen_ensemble_replay", {})
        records = replay.get("rows", [])
        if records and replay.get("status") == "ready":
            n = supplement["n_evaluate"]
            for row in records:
                output.append({"diagnostic_phase": "ensemble", "evaluation_n": n, "candidate_weight": .25,
                               "reference_fit": replay["reference"], **_flatten(replay["reference_metrics"], "reference_metrics."),
                               **_flatten(row)})
            best = min(records, key=lambda row: (row["fixed_quarter_blend"]["rmse_km"], row["name"]))
            lines += ["", f'Для {len(records)} повторно рассчитанных ансамблей испытана та же доля 25 %. На этой подвыборке наименьший RMSE смеси получил `{best["name"]}` — {_fmt(best["fixed_quarter_blend"]["rmse_km"])} км, выигрыш относительно его контроля {_fmt(best["quarter_blend_gain_vs_reference_km"])} км. [Все строки, включая ансамбли](diversity-extended-results.csv)']
    complete = {phase: sum(1 for row in output if row["diagnostic_phase"] == phase)
                for phase in ("raw", "meta", "ensemble")}
    if complete == {"raw": 300, "meta": 100, "ensemble": 100}:
        lines += ["", "В CSV представлены все 500 первоначальных конфигураций скрининга: 300 базовых моделей, 100 метамоделей и 100 ансамблей с идентификатором фазы `ensemble`. Отдельная фаза `v4`, если доступна, относится к предыдущему эксперименту."]
    return output


def _full_section(lines, full, directory, summary_dir):
    meta = full.get("diversity_fusion") if full else None
    if not meta:
        return
    scores = [r for r in full["candidates"] if r["name"] in meta["rules"]]
    scores.sort(key=lambda r: (r["cv_rmse_km"], r["name"]))
    lines += ["", "## Пять внешних фолдов", "",
              "В полный рейтинг вошли варианты с дополнительными экспертами. Итог по фолдам получен объединением ошибок всех строк перед извлечением квадратного корня.", "",
              *_table(["Правило", "MAE, км", "RMSE, км"], [
                  [f'`{r["name"]}`', _fmt(r["cv_mae_km"]), _fmt(r["cv_rmse_km"])] for r in scores])]
    rel = meta.get("diagnostics", "diversity/complementarity.json")
    candidates = [d / rel for d in (summary_dir, directory.parent / "full500", directory.parent / "geocv-full", directory.parent) if d]
    candidates.extend([directory / "complementarity.json", directory / "diversity" / "complementarity.json"])
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return
    detail = _read(path)
    lines += ["", f'Для {_fmt(detail["n"], 0)} строк каждый контроль обучен заново на четырёх основных экспертах. В таблице выигрыш равен RMSE контроля минус RMSE расширенного набора. Доля улучшений считает строки, на которых расширенный набор дал меньшую ошибку.', "",
              *_table(["Правило", "Четыре эксперта, км", "Расширенный набор, км", "Выигрыш, км", "Доля улучшений"], [
                  [f'`{r["name"]}`', _fmt(a["four_experts_refit"]["cv_rmse_km"]), _fmt(a["full"]["cv_rmse_km"]),
                   _fmt(-a["delta_rmse_km"]), _fmt(100*a["fraction_rows_improved"], 1)+" %"]
                  for r in scores if (a := detail["paired_ablation"].get(r["name"]))])]


def build(diversity_dir, docs_dir, full_summary=None):
    """Generate Markdown, CSV and a portable SVG once the calibration selection exists."""
    directory, docs = Path(diversity_dir), Path(docs_dir)
    selection_file = directory / "selection.json"
    if not selection_file.is_file():
        return False
    s = _read(selection_file)
    rows = sorted(s["candidate_calibration"], key=lambda r: (r["signed_gain_rank"],r["name"]))
    if (s["raw_models"] != 300 or len(rows) != 300 or len({r["name"] for r in rows}) != 300
            or any(r["name"].split("/")[0] not in FAMILIES for r in rows)):
        raise ValueError("Expected all 300 distinct calibrated raw configurations")
    chosen, routed = set(s["selected"]), set(s["full_cv_additions"])
    highlight = chosen | routed
    base = s["v4_calibration"]["rmse_km"]
    weaker = [r for r in rows if r["cal_standalone_gap_to_v4_km"] > 0 and r["cal_gain_signed_km"] > 0]
    weak = weaker[0]["name"] if weaker else None
    best = rows[0]
    eval_data = _read(directory/"results.json") if (directory/"results.json").is_file() else None
    eval_rows = eval_data["selected_or_routed"] if eval_data else {}
    values = []
    for r in rows:
        name = r["name"]
        flat = {"name": name, "family": name.split("/")[0], "selected": int(name in chosen),
                "routed_full_cv": int(name in routed), "new_for_full_cv": int(name in s["new_for_full_cv"])}
        flat.update(_flatten(r))
        if name in eval_rows:
            flat.update(_flatten(eval_rows[name], "screen_evaluation."))
        values.append(flat)
    lines = ["# Какие модели дополняют V4", "",
             f'Из 500 конфигураций исходного скрининга здесь рассмотрены 300 базовых моделей. Лучший измеренный выигрыш от отдельного добавления на калибровке дал `{best["name"]}`: {_fmt(best["cal_gain_signed_km"])} км относительно V4 с RMSE {_fmt(base)} км. При самостоятельном применении эта модель дала {_fmt(best["cal_standalone"]["rmse_km"])} км.', "",
             f'Для V4 использовали калибратор, обученный на 60 000 строках скрининга. На отдельных {_fmt(s["calibration_n"], 0)} строках оценивали вес каждого базового прогноза относительно V4. Каждая из четырёх частей калибровки получила прогноз с весом, найденным по трём остальным. Отбор среди 300 вариантов всё равно использует эту калибровку. В столбце выигрыша положительное число означает снижение RMSE WGS84.', "",
             "Базовые прогнозы V3 для каждой обучающей и калибровочной строки построены вне обучения на самой строке. Её метка при этом могла участвовать в обучении V3-моделей, подготовивших признаки других строк калибровки или обучения V4. Строгий вложенный протокол полной CV проверяет комбинацию на внешних фолдах отдельно.", "",
             '![Самостоятельный RMSE моделей и выигрыш в смеси с V4](diversity-scatter.svg)', "",
             f'У {len(weaker)} моделей самостоятельный RMSE выше базового, но добавление со знаковым коэффициентом улучшает калибровочный результат. '
             + (f'Ромб отмечает `{weak}`.' if weak else ''), "",
             "## Калибровка и выбор состава", "",
             "Коэффициент со знаком допускает экстраполяцию от прогноза модели. Положительная смесь допускает только движение V4 к её прогнозу. Корреляции ошибок измеряются от истинной точки к обоим прогнозам в общей локальной системе восток–север. Совпадение длин промахов и совпадение направлений описывают разные свойства. Выигрыш в километрах проверяет, уменьшила ли ошибка смеси фактический RMSE.", "",
             *_table(["Кандидат", "Самостоятельный RMSE, км", "Выигрыш со знаком, км", "Выигрыш при весе ≥ 0, км", "Корреляция направленных ошибок"], [
                 [f'`{r["name"]}`', _fmt(r["cal_standalone"]["rmse_km"]), _fmt(r["cal_gain_signed_km"]),
                  _fmt(r["cal_gain_positive_km"]), _fmt(r["error_corr_vector"], 3)]
                 for r in rows if r["signed_gain_rank"] <= 5 or r["name"] in highlight or r["name"] == weak]), "",
             f'Совместная смесь включает {len(chosen)} моделей после пошагового отбора. Приём каждой следующей модели требует снижения перекрёстно оценённого RMSE не менее чем на 0,05 км. Для полной пятифолдовой проверки направлено {len(routed)} дополнений с учётом уже доступных моделей и бюджета новых обучений.', "",
             *_table(["Шаг", "Кандидат", "Выигрыш к предыдущему шагу, км", "Принят"], [
                 [v["round"], f'`{v["candidate"]}`', _fmt(v["marginal_cal_gain_km"]), "да" if v["selected"] else "нет"]
                 for v in s["greedy_steps"]]), "",
             "Дополнения для полной CV — " + ", ".join(f'`{n}`' for n in s["full_cv_additions"]) + ".", "",
             "Все 300 строк, коэффициенты, показатели по востоку и северу, длинам ошибок и худшему децилю V4 доступны в [CSV](diversity-results.csv)."]
    example = next((row for row in rows if row["name"] == "retrieval/shrink_whiten/ret_kernel_mode"), None)
    if example:
        lines += ["", f'У `{example["name"]}` самостоятельный RMSE составил {_fmt(example["cal_standalone"]["rmse_km"])} км, {example["standalone_rank"]}-е место среди 300. Корреляция направленных ошибок с V4 равна {_fmt(example["error_corr_vector"],3)}. Добавление этого прогноза по отдельности снизило калибровочный RMSE на {_fmt(example["cal_gain_signed_km"])} км.']
    if eval_data:
        n = eval_data["v4"]["n"]
        lines += ["", "## Проверка после фиксации выбора", "",
                  f'Веса переобучены на всей калибровке и применены к {_fmt(n, 0)} строкам первого внешнего фолда. Эти строки ранее участвовали в ранжировании исходных 500 конфигураций. У V4 RMSE {_fmt(eval_data["v4"]["rmse_km"])} км и MAE {_fmt(eval_data["v4"]["mae_km"])} км. У зафиксированной совместной смеси RMSE {_fmt(eval_data["greedy_joint"]["rmse_km"])} км и MAE {_fmt(eval_data["greedy_joint"]["mae_km"])} км.', "",
                  *_table(["Кандидат", "Самостоятельный RMSE, км", "RMSE смеси со знаком, км", "Выигрыш, км"],
                          [[f'`{name}`', _fmt(v["standalone"]["rmse_km"]), _fmt(v["signed"]["rmse_km"]),
                            _fmt(v["gain_signed_km"])] for name,v in eval_rows.items()])]
        interval = eval_data.get("joint_paired_uncertainty")
        if interval:
            lo_r, hi_r = interval["delta_rmse_km_ci95"]
            lo_a, hi_a = interval["delta_mae_km_ci95"]
            lines += ["", f'Для разности «совместная смесь минус V4» парный бутстрэп по {_fmt(n,0)} строкам с {_fmt(interval["resamples"],0)} повторными выборками дал 95%-й интервал ΔRMSE от {_fmt(lo_r)} до {_fmt(hi_r)} км и ΔMAE от {_fmt(lo_a)} до {_fmt(hi_a)} км. Отрицательная разность означает улучшение. Ресэмплинг учитывает парность двух прогнозов на одном снимке. Неопределённость переноса на новые географические блоки требует независимой проверки с разбиением по территориям.']
    supp = _read(directory/"supplement.json") if (directory/"supplement.json").is_file() else None
    if supp:
        groups = [("Сохранённые правила группы decorrelated", "saved_decorrelated"),
                  ("Прогнозы полного внешнего фолда", "full_outer_cache"),
                  ("Варианты по публикациям", "literature_outer_cache"),
                  ("Повторно рассчитанные ансамбли скрининга", "screen_ensemble_replay")]
        lines += ["", "## Дополнительная диагностика", "",
                  f'После фиксации выбора сравнивали доступные прогнозы на тех же {_fmt(supp["n_evaluate"],0)} строках. У каждой группы использован её сохранённый контроль. Числа ниже показывают объём доступных сравнений.', "",
                  *_table(["Источник", "Сравнений", "Состояние"], [
                      [title, len(supp[k]["rows"]), {"ready":"готово", "cache_missing":"нет сохранённых прогнозов",
                       "component_list_missing":"нет списка участников", "component_predictions_missing":"неполный набор прогнозов"}.get(supp[k].get("status"),"сохранено")]
                      for title,k in groups if k in supp])]
    extended = _extended(lines, directory, supp)
    summary_dir = None
    if isinstance(full_summary, (Path, str)):
        path = Path(full_summary)
        summary_dir = path.parent
        full_summary = _read(path) if path.is_file() else None
    _full_section(lines, full_summary, directory, summary_dir)
    svg = _scatter(rows, base, highlight, weak)
    docs.mkdir(parents=True, exist_ok=True)
    outputs = {"diversity-analysis.md": "\n".join(lines)+"\n", "diversity-results.csv": _csv(values),
               "diversity-scatter.svg": svg}
    if extended:
        outputs["diversity-extended-results.csv"] = _csv(extended)
    for name, text in outputs.items():
        target = docs/name
        staging = target.with_suffix(target.suffix+".tmp")
        staging.write_text(text, encoding="utf-8")
        staging.replace(target)
    return True
