# Анализ и факторизация акцентной информации в дискретных представлениях речи

Эксперимент по исследованию того, как дискретные представления речи (HuBERT, EnCodec, WavLM) кодируют информацию об акценте говорящего. Используются датасет L2-ARCTIC и разреженные автоэнкодеры (SAE).

Подробное задание — в [task.md](task.md).

## Этапы эксперимента

1. **Подготовка окружения и данных** — распаковка L2-ARCTIC, организация по акцентам, загрузчик данных (PyTorch Dataset).
2. **Извлечение скрытых представлений** — скрипт `extract_features.py` для HuBERT (слои 6, 9, 12), затем EnCodec и WavLM.
3. **Обучение SAE** — обучение по одному SAE на слой (конфиг и интеграция с AudioSAE).
4. **Анализ признаков** — специализация признаков, тепловые карты, классификация акцентов, поиск характерных аудио-фрагментов.
5. **Сравнение моделей** — сводная таблица и графики по HuBERT, EnCodec, WavLM.

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

При использовании AudioSAE для обучения SAE — клонировать репозиторий в `third_party/audiosae/` и установить его зависимости отдельно (см. документацию AudioSAE).

## Подготовка данных

1. Распаковать zip-архивы в `data/l2_arctic/`:
   ```bash
   python scripts/unpack_l2_arctic.py
   ```
2. (Опционально) Разложить файлы по папкам акцентов:
   ```bash
   python scripts/organize_by_accent.py
   ```

## Запуск

- **Извлечение признаков** (HuBERT, слои 6, 9, 12):
  ```bash
  python src/extract_features.py --config configs/extract.yaml
  ```
- **Обучение SAE** (после появления кода AudioSAE):
  ```bash
  python src/train_sae.py --config configs/sae.yaml
  ```
- **Анализ** — скрипты в `src/analysis/` и `src/compare_models.py` используют конфиги и пути из `configs/` и `outputs/`.

Конфигурация: `configs/data.yaml`, `configs/extract.yaml`, `configs/sae.yaml`.

## Структура репозитория

- `configs/` — конфигурации путей и гиперпараметров
- `data/l2_arctic/` — датасет L2-ARCTIC (архивы → распакованные папки дикторов → по акцентам)
- `src/` — код: загрузка данных, извлечение признаков, обучение SAE, анализ
- `scripts/` — распаковка архивов, организация по акцентам
- `outputs/` — признаки, чекпоинты SAE, графики, таблицы, аудио-примеры (не в git)
