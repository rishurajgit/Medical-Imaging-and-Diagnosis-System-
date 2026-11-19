FROM node:22-alpine AS builder

WORKDIR /app
COPY package.json bun.lock* package-lock.json* ./
RUN npm install

COPY . .
RUN npm run build

FROM node:22-alpine

ENV NODE_ENV=production \
    PORT=3000

WORKDIR /app
COPY package.json ./
RUN npm install --omit=dev && npm install tsx

COPY --from=builder /app/dist ./dist
COPY server.ts ./server.ts
COPY src ./src

EXPOSE 3000

CMD ["npx", "tsx", "server.ts"]
