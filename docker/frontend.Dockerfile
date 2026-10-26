# BadmintonVision AI — static frontend served by nginx, proxying /api to the backend service.
# Build from the repository root:  docker build -f docker/frontend.Dockerfile -t badmintonvision-frontend .
FROM node:22-alpine AS build
WORKDIR /src
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# Leave empty to call the API on the same origin (nginx proxies /api); set it to point the
# static build at a separately hosted API instead.
ARG VITE_API_BASE=
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

FROM nginx:1.27-alpine
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /src/dist /usr/share/nginx/html
EXPOSE 80
