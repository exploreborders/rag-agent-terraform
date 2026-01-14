.PHONY: help setup install dev clean test lint format check deploy destroy

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

# Environment setup
setup: ## Set up development environment
	@echo "Setting up development environment..."
	python3.11 -m venv venv
	source venv/bin/activate && pip install -e .[dev]
	cp .env.example .env
	@echo "Environment setup complete. Run 'source venv/bin/activate' to activate."

install: ## Install dependencies
	@echo "Installing dependencies..."
	source venv/bin/activate && pip install -e .[dev]

dev: ## Start development server
	@echo "Starting development server..."
	source venv/bin/activate && cd src && python3.11 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Testing
test: ## Run unit tests only (excludes integration tests)
	@echo "Running unit tests..."
	source venv/bin/activate && cd src && python3.11 -m pytest tests/ -v -m "not integration" --cov=app --cov-report=html

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	source venv/bin/activate && cd src && python3.11 -m pytest tests/ -v -m "not integration"

test-integration: ## Run integration tests with real infrastructure
	@echo "Running integration tests..."
	@echo "Note: Requires 'make deploy' to be run first for database/Redis infrastructure"
	source venv/bin/activate && cd src && python3.11 -m pytest tests/integration/ -v -m "integration"

test-integration-quick: ## Run quick integration test to verify infrastructure
	@echo "Running quick integration test..."
	@echo "Note: Requires 'make deploy' to be run first for database/Redis infrastructure"
	source venv/bin/activate && cd src && python3.11 -m pytest tests/integration/test_simple_integration.py -v

test-all: ## Run all tests (unit + integration) - requires full infrastructure
	@echo "Running all tests..."
	source venv/bin/activate && cd src && python3.11 -m pytest tests/ -v

test-cov: ## Run unit tests with coverage report (excludes integration tests)
	@echo "Running unit tests with coverage..."
	source venv/bin/activate && cd src && python3.11 -m pytest tests/ -m "not integration" --cov=app --cov-report=term-missing --cov-report=html

test-cov-all: ## Run all tests with coverage (requires full infrastructure)
	@echo "Running all tests with coverage..."
	@echo "Note: Requires 'make deploy' to be run first for database/Redis infrastructure"
	source venv/bin/activate && cd src && python3.11 -m pytest tests/ --cov=app --cov-report=term-missing --cov-report=html

# Code quality
lint: ## Run all linting tools
	@echo "Running linters..."
	source venv/bin/activate && black . && isort . && flake8 . && mypy .

format: ## Format code with black and isort
	@echo "Formatting code..."
	source venv/bin/activate && black . && isort .

check: ## Check code quality without making changes
	@echo "Checking code quality..."
	source venv/bin/activate && black --check . && isort --check-only . && flake8 . && mypy .

# Infrastructure
infra-init: ## Initialize Terraform
	@echo "Initializing Terraform..."
	cd terraform && terraform init

infra-plan: ## Plan Terraform changes
	@echo "Planning Terraform changes..."
	cd terraform && terraform plan

infra-apply: ## Apply Terraform changes
	@echo "Applying Terraform changes..."
	cd terraform && terraform apply -auto-approve

infra-destroy: ## Destroy Terraform infrastructure
	@echo "Destroying Terraform infrastructure..."
	cd terraform && terraform destroy

infra-validate: ## Validate Terraform configuration
	@echo "Validating Terraform configuration..."
	cd terraform && terraform validate

# Docker containers are managed by Terraform
# Use 'make deploy' and 'make destroy' for infrastructure management

# Deployment
deploy: infra-apply ## Full deployment (infrastructure + containers)
	@echo "Deployment complete. Services starting..."

destroy: ## Full teardown (fast Docker cleanup)
	@echo "🧹 Fast cleanup - stopping containers..."
	-docker stop $$(docker ps -q --filter "name=rag-agent") 2>/dev/null || true
	@echo "🗑️ Removing containers..."
	-docker rm $$(docker ps -aq --filter "name=rag-agent") 2>/dev/null || true
	@echo "🌐 Removing network..."
	-docker network rm rag-agent-network 2>/dev/null || true
	@echo "🧽 Cleaning up images..."
	-docker image prune -f
	@echo "✅ Cleanup complete! Use 'make deploy' to restart."
	@echo "Teardown complete."

# Development utilities
clean: ## Clean up generated files
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".terraform" -exec rm -rf {} +
	rm -rf dist/ build/ .mypy_cache/

clean-all: clean ## Clean everything including venv
	@echo "Cleaning everything..."
	rm -rf venv/

# Data operations
ingest-docs: ## Ingest sample documents
	@echo "Ingesting documents..."
	source venv/bin/activate && cd src && python3.11 scripts/ingest_documents.py --input ../data/documents/ --recursive

setup-db: ## Set up vector database
	@echo "Setting up vector database..."
	source venv/bin/activate && cd src && python3.11 scripts/setup_vector_db.py

evaluate: ## Evaluate RAG performance
	@echo "Evaluating RAG performance..."
	source venv/bin/activate && cd src && python3.11 scripts/evaluate_rag.py --test-set ../data/test_queries.json

seed-db: ## Seed database with sample data
	@echo "Seeding database..."
	./scripts/seed_db.sh

# Documentation
docs-serve: ## Serve documentation locally
	@echo "Serving documentation..."
	source venv/bin/activate && mkdocs serve

docs-build: ## Build documentation
	@echo "Building documentation..."
	source venv/bin/activate && mkdocs build

# Git hooks
pre-commit: ## Run pre-commit hooks
	@echo "Running pre-commit hooks..."
	source venv/bin/activate && pre-commit run --all-files

# CI/CD
ci: check test ## Run CI checks (lint + test)
	@echo "CI checks passed!"

# Monitoring
monitoring-test: ## Test monitoring setup
	@echo "Testing monitoring setup..."
	./scripts/test_monitoring.sh

monitoring-logs: ## Show monitoring container logs
	@echo "Showing monitoring logs..."
	docker logs rag-agent-prometheus-dev --tail 50
	docker logs rag-agent-grafana-dev --tail 50

monitoring-status: ## Check monitoring container status
	@echo "Checking monitoring status..."
	docker ps | grep rag-agent | grep -E "(prometheus|grafana|exporter)"

# Development workflow
workflow-dev: setup install infra-init deploy dev ## Complete development setup
	@echo "Development environment ready!"

workflow-monitoring: infra-apply monitoring-test ## Complete monitoring setup
	@echo "Monitoring environment ready!"