# Usa una imagen base de Python
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia archivos
COPY . .

# Instala dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto 8080 que usa Cloud Run
EXPOSE 8080

# Comando para ejecutar la app con gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
