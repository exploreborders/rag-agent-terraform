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
test: ## Run all tests
	@echo "Running tests..."
	source venv/bin/activate && cd src && python3.11 -m pytest tests/ -v --cov=app --cov-report=html

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	source venv/bin/activate && cd src && python3.11 -m pytest tests/ -v -m "not integration"

test-integration: ## Run integration tests only
	@echo "Running integration tests..."
	source venv/bin/activate && cd src && python3.11 -m pytest tests/ -v -m "integration"

test-cov: ## Run tests with coverage report
	@echo "Running tests with coverage..."
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

destroy: infra-destroy ## Full teardown
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

# Development workflow
workflow-dev: setup install infra-init deploy dev ## Complete development setup
	@echo "Development environment ready!"