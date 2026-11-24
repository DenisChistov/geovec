"""Russian parameter descriptions and a controlled comparison for ensemble tuning."""

CONTEXTS = {
    "constant":"постоянные коэффициенты", "spread":"общее расхождение прогнозов",
    "geo":"положение V4 на сфере", "geo_spread":"положение V4 и общий разброс",
    "expert":"общий разброс и расхождение каждого эксперта с V4",
    "geo_expert":"положение V4, общий разброс и расхождение каждого эксперта с V4",
}
PARAMETERS = ("kind", "pool", "mode", "context", "alpha", "tail", "cap", "meta", "paper")
COLUMNS = ["tuning_" + k for k in PARAMETERS]


def describe(s):
    def fmt(x):
        return f"{x:g}".replace(".", ",")
    if s["kind"] == "fixed":
        return (f"Фиксированная касательная смесь от V4. Доля метамодели {fmt(s['meta'])}, "
                f"модели со штрафом расстояния {fmt(s['paper'])}, остаточная доля V4 {fmt(1-s['meta']-s['paper'])}.")
    pool = "четыре основных эксперта" if s["pool"] == "core" else "основные и дополнительные эксперты"
    mode = "общие веса для востока и севера" if s["mode"] == "shared" else "раздельные веса для востока и севера"
    return (f"Ridge, {pool}, {mode}. Контекст — {CONTEXTS[s['context']]}. "
            f"Регуляризация α = {fmt(s['alpha'])}, усиление хвоста {fmt(s['tail'])}, "
            f"предел шага {fmt(s['cap'])} км.")


def csv_params(spec):
    return {"tuning_" + k: v for k,v in spec.items() if k in PARAMETERS}


def chart(rows, specs, target):
    try:
        import matplotlib as mpl
        mpl.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        return False
    curves = {}
    for row in rows:
        s = specs[row["configuration"]]
        if s["kind"] == "ridge" and s["context"] == "geo_expert" and s["tail"] == 0 and s["cap"] == 20000:
            curves.setdefault((s["pool"],s["mode"]), []).append((s["alpha"],row["rmse_km"]))
    if not curves:
        return False
    with mpl.rc_context({"svg.fonttype":"none", "font.family":"DejaVu Sans", "svg.hashsalt":"geo-tuning500"}):
        fig, ax = plt.subplots(figsize=(9.5,4.7), layout="constrained")
        colors={"core":"#176c88","all":"#ae493e"}
        for (pool,mode), data in sorted(curves.items()):
            x,y=zip(*sorted(data))
            label=("Основные" if pool=="core" else "Все эксперты") + (" · общий вес" if mode=="shared" else " · раздельные веса")
            ax.semilogx(x,y,linestyle="-" if mode=="shared" else "--",marker="o" if mode=="shared" else "s",
                        color=colors[pool],label=label,linewidth=1.8,markersize=4)
        ax.set_title("Регуляризация поправки к V4 при одинаковом контексте",loc="left",fontweight="bold",pad=11)
        ax.set_xlabel("Относительная сила регуляризации α")
        ax.set_ylabel("Общий RMSE пяти фолдов, км")
        ticks=sorted({point[0] for data in curves.values() for point in data})
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,_: f"{value:g}".replace(".",",")))
        ax.grid(axis="y",color="#dde3e6",linewidth=.8)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(frameon=False,loc="best",fontsize=9,ncols=2)
        tmp=target.with_suffix(".tmp.svg")
        fig.savefig(tmp,format="svg",metadata={"Date":None})
        plt.close(fig)
        tmp.replace(target)
    return True
