FROM nginx:1.27-alpine

COPY src/docker/gateway/nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
