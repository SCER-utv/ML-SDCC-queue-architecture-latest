# Usa un'immagine Python ufficiale e leggera
FROM python:3.10-slim

# Imposta la directory di lavoro
WORKDIR /app

# Copia i requisiti e installa le dipendenze
# (La tua immagine base all'inizio, assicurati che sia "slim" e NON "alpine",
# ad esempio: FROM python:3.10-slim)

# Copia i requisiti
COPY requirements.txt .

# 1. Aggiorna pip, setuptools e wheel (I "motori" di installazione)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 2. Installa le librerie
RUN pip install --no-cache-dir -r requirements.txt

# Copia l'intera struttura del progetto nel container
COPY . .

# Cruciale: aggiunge la root al PYTHONPATH affinché gli import
# come 'from src.model.model_factory import ...' funzionino ovunque
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Non impostiamo un CMD fisso, lo decideremo al lancio del container