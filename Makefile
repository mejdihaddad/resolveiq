.PHONY: run-docker-compose up down ps logs test frontend-build compose-check

# Start the complete local stack. Qdrant data remains in its bind-mounted folder.
run-docker-compose: up

up:
	docker compose up --build -d

down:
	docker compose down

ps:
	docker compose ps

logs:
	docker compose logs --tail=100

test:
	uv run pytest

frontend-build:
	npm --prefix frontend run build

compose-check:
	docker compose config -q
