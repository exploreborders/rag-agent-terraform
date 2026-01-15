.PHONY: help setup dev test up down clean status logs

# Default target
help: ## Show available commands
	@echo "RAG Agent - Development Commands"
	@echo "================================"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make setup    # Setup environment (first time only)"
	@echo "  make build    # Build Docker images (first time or when code changes)"
	@echo "  make up       # Deploy and start all services"
	@echo "  make dev      # Start development server"
	@echo ""
	@echo "🛑 Cleanup:"
	@echo "  make down     # Stop all services"
	@echo "  make clean    # Clean build artifacts"
	@echo ""
	@echo "🧪 Testing & Code Quality:"
	@echo "  make test     # Run tests"
	@echo "  make lint     # Check code quality"
	@echo ""
	@echo "🔍 Debugging:"
	@echo "  make status   # Check service status"
	@echo "  make logs     # Show service logs"

setup: ## Setup development environment (venv + dependencies)
	python3.11 -m venv venv || python3 -m venv venv
	./venv/bin/pip install -e .[dev]
	cp .env.example .env 2>/dev/null || true
	@echo "✅ Setup complete. Use 'make build && make up' to start services."

build: ## Build Docker images for deployment
	@echo "🏗️ Building Docker images..."
	export DOCKER_BUILDKIT=1 && \
	docker build \
		--target production \
		--cache-from rag-agent-app:latest \
		--tag rag-agent-app:latest \
		-f docker/app/Dockerfile \
		.
	@echo "✅ App image built"
	cd mcp-coordinator && \
	export DOCKER_BUILDKIT=1 && \
	docker build \
		--target production \
		--cache-from rag-agent-mcp-coordinator:latest \
		--tag rag-agent-mcp-coordinator:latest \
		-f Dockerfile \
		.
	@echo "✅ MCP Coordinator image built"
	@echo "🎯 Images ready for deployment!"

up: ## Deploy and start all services (fast Terraform)
	@echo "🚀 Starting Terraform deployment..."
	@cd terraform && terraform init >/dev/null 2>&1
	@cd terraform && terraform apply -auto-approve
	@echo "⏳ Waiting for services..."
	@sleep 10
	@echo "✅ Services running!"
	@echo "🌐 App: http://localhost:8000"
	@echo "📊 Grafana: http://localhost:3000 (admin/admin)"

dev: ## Start development server
	./venv/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test: ## Run tests
	./venv/bin/pytest tests/ -v --cov=app --cov-report=term-missing

lint: ## Check and fix code quality
	./venv/bin/black .
	./venv/bin/isort .
	./venv/bin/flake8 .
	./venv/bin/mypy .

down: ## Stop all services
	@echo "🧹 Destroying Terraform infrastructure..."
	@cd terraform && terraform destroy -auto-approve >/dev/null 2>&1
	@echo "✅ Services stopped and cleaned up"

status: ## Check service status
	@echo "📊 Docker containers:"
	@docker ps --filter "name=rag-agent" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || echo "No containers running"
	@echo ""
	@echo "🌐 Service health:"
	@echo "   App: $$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || echo "DOWN")"
	@echo "   MCP: $$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8001/health 2>/dev/null || echo "DOWN")"

logs: ## Show service logs
	@echo "Showing logs for all RAG Agent containers..."
	@for container in $$(docker ps --filter "name=rag-agent" --format "{{.Names}}"); do \
		echo "=== $$container ==="; \
		docker logs $$container --tail 10 2>/dev/null || echo "No logs available"; \
		echo ""; \
	done

clean: ## Clean build artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -rf dist/ build/ .mypy_cache/ htmlcov/

