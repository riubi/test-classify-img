import glob
import json
import os

import pandas as pd
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, PageBreak, SimpleDocTemplate, Spacer

from report_lib import P, bullets, center, fig, h1, h2, table, title
from report_text import TEXT

GITHUB = "https://github.com/riubi/test-classify-img"
RUNS = "results/runs"
FIG = "results/figures"

summary = pd.read_csv("results/summary.csv").set_index("run")
split_sizes = pd.read_csv("results/split_sizes.csv", index_col=0)
with open("results/selection.json") as f:
    sel = json.load(f)
BASE, BEST_CNN = sel["base_cnn"], sel["best_cnn"]
RES = ["resnet_fc", "resnet_layer4", "resnet_layer3_4", "resnet_full"]
LAYERS = {"resnet_fc": "FC", "resnet_layer4": "layer4 + FC", "resnet_layer3_4": "layer3-4 + FC",
          "resnet_full": "Full fine-tuning"}


def f3(x):
    return f"{x:.3f}"


def run_row(run, changed):
    s = summary.loc[run]
    return [run, changed, int(s.best_epoch), int(s.epochs), f3(s.train_f1), f3(s.val_f1),
            f3(s.test_accuracy), f3(s.test_f1)]


RUN_HEADER = ["запуск", "что изменено", "best epoch", "эпох", "train F1*", "val F1", "test Acc", "test macro F1"]
RUN_W = [0.17, 0.25, 0.08, 0.07, 0.1, 0.1, 0.1, 0.13]

story = []

story += [Spacer(1, 2 * cm), P("Сессионный проект", center),
          P("Классификация планктона WHOI-Plankton при ограниченном количестве данных", title),
          Spacer(1, 0.6 * cm),
          P("Ruslan Papina, группа М25-555", center),
          P("Вариант P: классы, в которых от 101 до 200 изображений", center),
          P(f'Код проекта: <link href="{GITHUB}" color="blue">{GITHUB}</link>', center),
          Spacer(1, 1 * cm)]
story += [P(TEXT["intro"])]

story += [PageBreak(), P("1. Данные и выбор классов", h1)]
story += [P(t) for t in TEXT["data"]]
rows = [["класс", "что это", "всего", "train", "val", "test"]]
for c in split_sizes.index:
    r = split_sizes.loc[c]
    rows.append([c, TEXT["class_desc"].get(c, ""), int(r.total), int(r.train), int(r.val), int(r.test)])
story += [P("Таблица 1. Выбранные классы и разбиение", h2)]
story += table(rows, [0.24, 0.4, 0.09, 0.09, 0.09, 0.09])
story += [P(t) for t in TEXT["split"]]
story += fig(f"{FIG}/class_counts.png", 1.0,
             "Рис. 1. Число изображений во всех 103 классах (логарифмическая шкала). "
             "Оранжевым выделены классы варианта.")
story += fig(f"{FIG}/examples.png", 0.75, "Рис. 2. Примеры изображений выбранных классов после предобработки "
                                          "(паддинг до квадрата и resize 224×224)")

story += [P("2. Предобработка", h1)]
story += [P(TEXT["prep"][0])]
story += bullets(TEXT["prep_steps"])
story += [P(t) for t in TEXT["prep"][1:]]

story += [P("3. Аугментации", h1), P(TEXT["aug_intro"])]
story += [P("Таблица 2. Аугментации для train (схема basic)", h2)]
story += table([["преобразование", "параметры", "зачем"]] + TEXT["aug_used"], [0.25, 0.27, 0.48])
story += [P("Что не использую и почему", h2)]
story += bullets(TEXT["aug_not_used"])
story += [P(TEXT["aug_strong"])]
story += fig(f"{FIG}/aug_examples.png", 0.9, "Рис. 3. Исходное изображение и случайные варианты basic и strong")
story += [P("Эксперимент: своя CNN без аугментаций и с аугментациями", h2), P(TEXT["aug_exp_setup"])]
story += table([RUN_HEADER] + [run_row(r, c) for r, c in [("cnn_noaug", "без аугментаций"), ("cnn_aug", "basic"),
                                                          ("cnn_aug_strong", "strong")]], RUN_W)
story += [P(TEXT["train_f1_note"])]
story += fig(f"{FIG}/aug_curves.png", 1.0, "Рис. 4. Кривые обучения своей CNN с разными аугментациями")
story += [P(t) for t in TEXT["aug_result"]]
if "cnn_aug_flip" in summary.index:
    story += [P("Проверка: аугментации без поворотов на 90°", h2), P(TEXT["aug_flip_intro"])]
    story += table([RUN_HEADER] + [run_row(r, c) for r, c in [("cnn_noaug", "без аугментаций"), ("cnn_aug", "basic"),
                                                              ("cnn_aug_flip", "flip")]], RUN_W)
    story += fig(f"{FIG}/aug_flip_curves.png", 1.0, "Рис. 4а. Своя CNN: basic и flip")
    story += [P(t) for t in TEXT["aug_flip_result"]]

story += [PageBreak(), P("4. Собственная CNN", h1), P(TEXT["cnn_arch_intro"])]
story += table([["блок", "слои", "размер выхода", "параметров"]] + TEXT["cnn_arch"], [0.14, 0.5, 0.18, 0.18])
story += [P(t) for t in TEXT["cnn_train"]]
story += [P("Исследование параметров", h2), P(TEXT["cnn_study_intro"])]
story += table([RUN_HEADER] + [run_row(r, c) for r, c in TEXT["cnn_runs"] if r in summary.index], RUN_W)
story += fig(f"{FIG}/cnn_params.png", 1.0, "Рис. 5. Macro F1 на val и test для всех запусков своей CNN")
for key, cap in TEXT["cnn_curves"]:
    story += fig(f"{FIG}/cnn_{key}_curves.png", 1.0, cap)
story += [P(t) for t in TEXT["cnn_result"]]

story += [PageBreak(), P("5. ResNet18 и transfer learning", h1)]
story += [P(t) for t in TEXT["resnet_intro"]]
rows = [["режим", "обучаются", "заморожено", "обучаемых параметров"]]
for r, train_part, frozen in TEXT["resnet_modes"]:
    rows.append([r, train_part, frozen, f"{int(summary.loc[r].trainable_params):,}".replace(",", " ")])
story += table(rows, [0.18, 0.3, 0.32, 0.2])
story += [P(t) for t in TEXT["resnet_train"]]
story += table([RUN_HEADER] + [run_row(r, LAYERS[r]) for r in RES], RUN_W)
story += fig(f"{FIG}/resnet_curves.png", 1.0, "Рис. 6. Кривые обучения ResNet18 в четырёх режимах")
story += [P(t) for t in TEXT["resnet_result"]]
if os.path.exists(f"{FIG}/resnet_aug.png"):
    story += [P("Дополнительно: ResNet18 с аугментациями basic", h2), P(TEXT["resnet_aug_intro"])]
    story += table([RUN_HEADER] + [run_row(r + "_aug", LAYERS[r] + ", basic") for r in RES
                                   if r + "_aug" in summary.index], RUN_W)
    story += fig(f"{FIG}/resnet_aug.png", 0.9, "Рис. 6а. ResNet18 без аугментаций и с basic")
    story += [P(t) for t in TEXT["resnet_aug_result"]]

story += [PageBreak(), P("6. Метрики на test", h1), P(TEXT["metrics_intro"])]
if BEST_CNN == BASE:
    final, names = [BASE], ["Custom CNN"]
else:
    final, names = [BASE, BEST_CNN], ["Custom CNN (база)", "Custom CNN (лучшая)"]
final += RES
names += ["ResNet18 " + LAYERS[r] for r in RES]
rows = [["модель", "запуск", "Accuracy", "Balanced Acc", "Precision (macro)", "Recall (macro)", "Macro F1"]]
for r, n in zip(final, names):
    s = summary.loc[r]
    rows.append([n, r, f3(s.test_accuracy), f3(s.test_balanced_acc), f3(s.test_precision), f3(s.test_recall),
                 f3(s.test_f1)])
story += table(rows, [0.24, 0.16, 0.11, 0.12, 0.13, 0.12, 0.12])

pc = pd.read_csv("results/per_class_f1.csv", index_col=0)
rows = [["класс"] + names]
for c in pc.index:
    rows.append([c] + [f"{v:.3f}" for v in pc.loc[c]])
story += [P("F1 по классам на test", h2)]
story += table(rows, [0.22] + [0.78 / len(names)] * len(names), font_size=7)
story += fig(f"{FIG}/per_class_f1.png", 0.8, "Рис. 7. F1 по классам для итоговых моделей")
story += [P(t) for t in TEXT["per_class_result"]]

conf = pd.read_csv("results/top_confusions.csv")
rows = [["модель", "истинный класс", "предсказан как", "сколько", "доля класса"]]
for r, n in zip(final, names):
    for _, c in conf[conf.run == r].head(3).iterrows():
        rows.append([n, c.true, c.pred, int(c["count"]), f"{c.share_of_true:.0%}"])
story += [P("Самые частые ошибки (по 3 на модель, test)", h2)]
story += table(rows, [0.26, 0.24, 0.24, 0.12, 0.14])
story += [P(t) for t in TEXT["confusion_result"]]

story += [P("Confusion matrix и кривые обучения итоговых моделей", h2)]
for r, n in zip(final, names):
    cap = f"{n}: loss, accuracy и macro F1 по эпохам (пунктир - лучшая эпоха) и confusion matrix на test"
    story += [KeepTogether(fig(f"{RUNS}/{r}/curves.png", 1.0) + fig(f"{RUNS}/{r}/cm_test.png", 0.5, cap))]

story += [PageBreak(), P("7. Сравнение моделей", h1), P(TEXT["compare_intro"])]
rows = [["Model", "Trainable<br/>layers", "Trainable<br/>params", "Aug", "Batch<br/>size", "Activation",
         "Dropout", "Optimizer", "LR", "Best<br/>epoch", "Accuracy", "Macro<br/>F1"]]
for r, n in zip(final, names):
    s = summary.loc[r]
    cnn = s.model == "Custom CNN"
    rows.append([n if cnn else "ResNet18", "все (с нуля)" if cnn else LAYERS[r],
                 f"{int(s.trainable_params):,}".replace(",", " "), s.aug,
                 int(s.batch_size), s.activation, f"{s.dropout:g}" if cnn else "-",
                 s.optimizer + (f", wd={s.wd:g}" if s.wd else ""), s.lr, int(s.best_epoch), f3(s.test_accuracy),
                 f3(s.test_f1)])
story += table(rows, font_size=6.5)
story += [P(TEXT["compare_common"])]
story += fig(f"{FIG}/final_compare.png", 0.85, "Рис. 8. Macro F1 итоговых моделей на val и test")

story += [P("Переобучение", h2)]
rows = [["модель", "train F1*", "val F1", "test F1", "train loss*", "val loss", "test loss"]]
for r, n in zip(final, names):
    s = summary.loc[r]
    rows.append([n, f3(s.train_f1), f3(s.val_f1), f3(s.test_f1), f3(s.train_loss), f3(s.val_loss),
                 f3(s.test_loss)])
story += table(rows, [0.3, 0.115, 0.115, 0.115, 0.12, 0.115, 0.12])
story += [P(t) for t in TEXT["overfit_result"]]

if os.path.exists("results/seeds_summary.csv"):
    sd = pd.read_csv("results/seeds_summary.csv").set_index("run")
    rows = [["модель", "запусков", "val F1", "test Accuracy", "test Macro F1"]]
    for r, n in zip(final, names):
        if r in sd.index:
            x = sd.loc[r]
            rows.append([n, int(x.n_seeds), f"{x.val_f1_mean:.3f} ± {x.val_f1_std:.3f}",
                         f"{x.test_acc_mean:.3f} ± {x.test_acc_std:.3f}",
                         f"{x.test_f1_mean:.3f} ± {x.test_f1_std:.3f}"])
    story += [P("Повторы с другими seed", h2), P(TEXT["seeds_intro"])]
    story += table(rows)
    story += [P(t) for t in TEXT["seeds_result"]]

extra = sorted(glob.glob("results/extra/*_patience50/metrics.json"))
if extra:
    rows = [["seed", "с early stopping: best / эпох", "test F1", "без early stopping: best / эпох", "test F1"]]
    for p in extra:
        name = p.split("/")[2].replace("_patience50", "")
        seed = name.rsplit("_s", 1)[1]
        with open(f"results/seeds/{name}/metrics.json") as f:
            a = json.load(f)
        with open(p) as f:
            b = json.load(f)
        rows.append([seed, f"{a['best_epoch']} / {a['epochs_trained']}", f3(a["test"]["f1_macro"]),
                     f"{b['best_epoch']} / {b['epochs_trained']}", f3(b["test"]["f1_macro"])])
    story += [P("Early stopping у своей CNN", h2), P(TEXT["es_intro"])]
    story += table(rows, [0.1, 0.27, 0.15, 0.3, 0.18])
    story += [P(t) for t in TEXT["es_result"]]

story += [P("8. Выводы", h1)]
for q, a in TEXT["conclusions"]:
    story += [P(f"<b>{q}</b>"), P(a)]

story += [P("Приложение. Все запуски", h1), P(TEXT["appendix"])]
rows = [["запуск", "aug", "act", "блоков", "bs", "opt", "lr", "drop", "best ep", "train F1*", "val F1", "test Acc",
         "test Bal.Acc", "test P", "test R", "test F1", "мин"]]
for r, s in summary.iterrows():
    blocks = "" if pd.isna(s.channels) else len(str(s.channels).split(","))
    rows.append([r, s.aug, s.activation, blocks, int(s.batch_size), s.optimizer, s.lr,
                 "" if pd.isna(s.dropout) else f"{s.dropout:g}", int(s.best_epoch), f3(s.train_f1), f3(s.val_f1),
                 f3(s.test_accuracy), f3(s.test_balanced_acc), f3(s.test_precision), f3(s.test_recall),
                 f3(s.test_f1), f"{s.time_min:.1f}"])
story += table(rows, font_size=6)
story += [P("Воспроизведение", h2)]
story += bullets(TEXT["reproduce"])

os.makedirs("report", exist_ok=True)
doc = SimpleDocTemplate("report/report.pdf", leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                        bottomMargin=1.6 * cm, title="Классификация планктона WHOI-Plankton", author="Ruslan Papina")


def page_num(canvas, doc_):
    canvas.setFont("DejaVu", 8)
    canvas.drawRightString(doc_.pagesize[0] - 1.8 * cm, 1 * cm, str(doc_.page))


doc.build(story, onFirstPage=page_num, onLaterPages=page_num)
print("report/report.pdf готов")
