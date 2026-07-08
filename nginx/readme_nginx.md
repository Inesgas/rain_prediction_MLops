# double authentification via nginx
one way - certification for access
the second way - protected viy passwords for defined parts of the api.


# Nginx Reverse Proxy & Load Balancer

This directory contains the configuration for the Nginx web server, acting as the central API gateway (Reverse Proxy) in front of the FastAPI instances (`rain-api`).

## Security & Architecture Features

* **HTTPS Enforcement:** Automatic redirection of all HTTP traffic (Port 80) to secure HTTPS (Port 443) using SSL/TLS. (see nginx.conf-file)

* **Load Balancing:** Distributes incoming traffic across 3 FastAPI replicas utilizing the `least_conn` (least connections) algorithm. This protects the system from DDoS-atack. For our example there were only 3 gateways created (Port 8502 )
(see nginx.conf-file)

* **Layered Authentication:** 
  * **Public Endpoints:** `/health`, `/locations`, `/docs`, `/openapi.json` (accessible for monitoring and API documentation).

  * **Protected Endpoints (Basic Auth via `.htpasswd`):** `/predict`, `/predict/batch`, `/model`, `/metrics`.

* **DDoS & Bot Protection:** Implements rate limiting (`100r/m` with burst buffer) on prediction endpoints and restricts payload sizes (`client_max_body_size 1M`).
(see nginx.conf-file)

## Directory Structure

```text
nginx/
├── nginx.conf          # Main Nginx configuration file
├── .htpasswd           # Encrypted user credentials (Basic Auth / our passwords)
└── certs/
    ├── nginx.crt       # SSL certificate
    └── nginx.key       # Private SSL key
```

## Docker-Compose Integration (Example)

Ensure that the configuration, credentials, and certificates are correctly mounted as volumes within your `docker-compose.yml`:

```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/.htpasswd:/etc/nginx/.htpasswd:ro
      - ./nginx/certs:/etc/nginx/certs:ro
    depends_on:
      - fastapi

```

## Useful Commands

### Add a new user for Basic Authentication
```bash
htpasswd -B .htpasswd <new_username>
```

### Test Nginx configuration inside a running container
```bash
docker compose exec nginx nginx -t
```
