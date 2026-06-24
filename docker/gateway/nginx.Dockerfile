FROM nginx:1.25-alpine

# Remove the default Nginx welcome page configuration
RUN rm /etc/nginx/conf.d/default.conf

# Copy the custom configuration. TLS certificates are mounted at runtime.
COPY nginx/nginx.conf /etc/nginx/nginx.conf

# Export ports for HTTP and HTTPS
EXPOSE 80
EXPOSE 443

CMD ["nginx", "-g", "daemon off;"]
