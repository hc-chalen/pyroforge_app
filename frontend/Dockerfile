# ---------------------------------------------------------------------------
# PyroForge frontend — multi-stage build.
#   dev        → Vite dev server with HMR (used by docker-compose.override.yml)
#   build      → compiles the production static bundle
#   production → serves that bundle via nginx (default target for deploys)
# ---------------------------------------------------------------------------

# --- Shared base with installed dependencies ---------------------------------
FROM node:22-alpine AS base
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .

# --- Dev: hot-reloading Vite dev server --------------------------------------
FROM base AS dev
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# --- Build: emit static assets into /app/dist --------------------------------
# VITE_API_URL is baked into the bundle at build time (Vite inlines import.meta.env).
FROM base AS build
ARG VITE_API_URL=
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

# --- Production: nginx serving the static bundle -----------------------------
FROM nginx:1.27-alpine AS production
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
