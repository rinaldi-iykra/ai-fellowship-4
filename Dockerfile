FROM python:3.12.7-slim

COPY . .

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r project/requirements.txt

ENV DEBIAN_FRONTEND=dialog

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8050", "--timeout", "900", "project.app.main:server"]