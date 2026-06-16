FROM nginx:1.25-alpine

# Remove the default Nginx welcome page configuration
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom configurations and certificates into the container
COPY nginx/nginx.conf /etc/nginx/nginx.conf
COPY nginx/certs/nginx.crt /etc/nginx/certs/nginx.crt
COPY nginx/certs/nginx.key /etc/nginx/certs/nginx.key

# Export ports for HTTP and HTTPS
EXPOSE 80
EXPOSE 443

CMD ["nginx", "-g", "daemon off;"]
