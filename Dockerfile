FROM python:3.12-slim

# uv з офіційного образу — без curl | sh і без окремого шару з pip.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Спершу лише залежності. Шар перебудовується тільки при зміні lock-файлу,
#    а не при кожній правці коду.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) Потім код і сам пакет.
COPY src/ src/
RUN uv sync --frozen --no-dev

# Дані монтуються ззовні (-v ./data:/app/data), щоб образ не залежав від
# конкретного вивантаження інбоксу.
COPY data/ data/

# --no-sync: усе вже встановлено на етапі збірки, у рантаймі мережа не потрібна.
ENTRYPOINT ["uv", "run", "--no-sync", "python", "-m", "triage"]
CMD ["--input", "data/input_requests.csv", "--out", "out"]
