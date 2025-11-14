FROM node:20-alpine AS frontend-build
WORKDIR /app
COPY clientapp/package*.json ./
RUN npm install
COPY clientapp/ ./
RUN npm run build

# Stage 2: Flask with React build
FROM python:3.11-slim
WORKDIR /app

COPY FarmerApp.Api/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY FarmerApp.Api/ ./
COPY --from=frontend-build /app/build ./

EXPOSE 8080
CMD ["python", "app.py"]