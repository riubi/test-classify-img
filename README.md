# Классификация планктона WHOI-Plankton при малом количестве данных

Сессионный проект, Ruslan Papina, группа М25-555. Вариант **P**: берутся все классы, в которых от 101 до 200 изображений.
Сравниваются собственная CNN (с нуля) и ResNet18 с весами ImageNet в четырёх режимах дообучения.

Отчёт: [report/report.pdf](report/report.pdf)

## Данные

Датасет [project-oceania/whoi-plankton](https://huggingface.co/datasets/project-oceania/whoi-plankton),
3 563 596 снимков IFCB, 103 класса. В диапазон 101-200 попадает 7 классов (1043 изображения):

| класс | изображений |
|---|---|
| Cerataulina_flagellate | 142 |
| Chaetoceros_flagellate | 149 |
| Emiliania_huxleyi | 148 |
| Pleuronema_sp | 115 |
| Strombidium_oculatum | 152 |
| Strombidium_wulffi | 157 |
| dino_large1 | 180 |

Число изображений по классам взято из карточки официального датасета (`results/class_counts.csv`).
Доступ к самим parquet-файлам project-oceania выдаётся вручную владельцами. Пока заявка была на рассмотрении,
картинки выгружались из открытой копии [nf-whoi/whoi-plankton](https://huggingface.co/datasets/nf-whoi/whoi-plankton).
Это тот же архив WHOI без класса `mix` и двух классов на 5 и 4 картинки. Для всех 7 классов число картинок
совпадает с официальным (в `prepare_data.py` стоит проверка).

Разбиение train / val / test = 70 / 15 / 15, стратифицированное, seed 42: `splits/split.csv`.
Одно и то же разбиение используется во всех экспериментах.

## Результаты

Test macro F1, среднее ± std по трём seed (42, 1, 2), выборки одинаковые для всех моделей:

| модель | обучаемые слои | test macro F1 | val macro F1 |
|---|---|---|---|
| Custom CNN, 4 блока (база) | все | 0.825 ± 0.028 | 0.797 ± 0.023 |
| Custom CNN, 5 блоков, без early stopping | все | 0.882 ± 0.025 | 0.886 ± 0.008 |
| ResNet18 | fc | 0.920 ± 0.020 | 0.929 ± 0.000 |
| ResNet18 | layer4 + fc | 0.948 ± 0.012 | 0.956 ± 0.016 |
| ResNet18 | layer3 + layer4 + fc | 0.940 ± 0.003 | 0.966 ± 0.003 |
| ResNet18 | вся сеть | 0.930 ± 0.018 | 0.958 ± 0.013 |

Подробности, все запуски, графики и выводы - в отчёте. Метрики и графики каждого запуска лежат в `results/runs/`,
сводная таблица - `results/summary.csv`.

## Структура

```
prepare_data.py      подсчёт классов, выгрузка картинок, разбиение
data.py              датасет, предобработка, аугментации
models.py            своя CNN и ResNet18 с режимами заморозки
train.py             обучение одной конфигурации + оценка на test
run_experiments.sh   все эксперименты подряд
run_seeds.py         повтор итоговых моделей с seed 1 и 2
summarize.py         сводная таблица и графики по всем запускам
make_report.py       сборка report/report.pdf
results/             метрики, графики, confusion matrix по каждому запуску
```

## Запуск

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
hf auth login
python prepare_data.py
bash run_experiments.sh
python summarize.py
python run_seeds.py
python summarize.py
python make_report.py
```

`hf auth login` нужен, потому что карточка gated-датасета читается только с токеном. `prepare_data.py` работает
около 15 минут и качает только нужные row group'ы (~1 ГБ).

Обучение запускалось на MacBook Air M2 (MPS). На MPS часть операций недетерминирована,
поэтому при повторном запуске метрики могут немного отличаться.
