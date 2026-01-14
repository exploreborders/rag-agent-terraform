#!/bin/bash
# Monitoring Setup Test Script
# This script tests the monitoring infrastructure setup

set -e

echo "🔍 Testing RAG Agent Monitoring Setup"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a service is responding
check_service() {
    local name=$1
    local url=$2
    local expected_code=${3:-200}

    echo -n "Testing $name ($url)... "

    if curl -s --max-time 5 --connect-timeout 5 -o /dev/null -w "%{http_code}" "$url" | grep -q "^$expected_code$"; then
        echo -e "${GREEN}✅ OK${NC}"
        return 0
    else
        echo -e "${RED}❌ FAILED${NC}"
        return 1
    fi
}

# Check if containers are running
echo -e "\n${YELLOW}Checking container status...${NC}"
if command -v docker &> /dev/null; then
    if docker ps | grep -q "rag-agent-prometheus"; then
        echo -e "${GREEN}✅ Prometheus container is running${NC}"
    else
        echo -e "${RED}❌ Prometheus container is not running${NC}"
        echo "Run: cd terraform && terraform apply"
        exit 1
    fi

    if docker ps | grep -q "rag-agent-grafana"; then
        echo -e "${GREEN}✅ Grafana container is running${NC}"
    else
        echo -e "${RED}❌ Grafana container is not running${NC}"
        echo "Run: cd terraform && terraform apply"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠️ Docker not available, skipping container checks${NC}"
fi

echo -e "\n${YELLOW}Testing service endpoints...${NC}"

# Test Prometheus
if ! check_service "Prometheus" "http://localhost:9090/-/healthy"; then
    echo "Prometheus health check failed"
fi

# Test Grafana
if ! check_service "Grafana" "http://localhost:3000/api/health"; then
    echo "Grafana health check failed"
fi

# Test RAG Agent metrics endpoint
if ! check_service "RAG Agent Metrics" "http://localhost:8000/metrics"; then
    echo "RAG Agent metrics endpoint not accessible"
fi

# Test exporters
if ! check_service "PostgreSQL Exporter" "http://localhost:9187/metrics"; then
    echo "PostgreSQL exporter not accessible"
fi

if ! check_service "Redis Exporter" "http://localhost:9121/metrics"; then
    echo "Redis exporter not accessible"
fi

if ! check_service "Node Exporter" "http://localhost:9100/metrics"; then
    echo "Node exporter not accessible"
fi

echo -e "\n${YELLOW}Testing Prometheus targets...${NC}"

# Check Prometheus targets status
PROMETHEUS_URL="http://localhost:9090"
TARGETS_RESPONSE=$(curl -s "$PROMETHEUS_URL/api/v1/targets" 2>/dev/null)

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Prometheus API accessible${NC}"

    # Check for unhealthy targets
    UNHEALTHY=$(echo "$TARGETS_RESPONSE" | grep -o '"health":"[^"]*"' | grep -v '"health":"up"' | wc -l)

    if [ "$UNHEALTHY" -gt 0 ]; then
        echo -e "${RED}⚠️ $UNHEALTHY unhealthy targets detected${NC}"
        echo "$TARGETS_RESPONSE" | jq -r '.data.activeTargets[] | select(.health != "up") | "\(.labels.job): \(.lastError)"' 2>/dev/null || echo "Install jq to see detailed target status"
    else
        echo -e "${GREEN}✅ All Prometheus targets are healthy${NC}"
    fi
else
    echo -e "${RED}❌ Cannot access Prometheus API${NC}"
fi

echo -e "\n${YELLOW}Grafana Dashboard URLs:${NC}"
echo "Main Dashboard: http://localhost:3000/d/rag-agent-performance/rag-agent-performance-dashboard"
echo "Database Dashboard: http://localhost:3000/d/database-performance/database-performance-dashboard"
echo "Infrastructure Dashboard: http://localhost:3000/d/infrastructure/infrastructure-dashboard"

echo -e "\n${GREEN}🎉 Monitoring setup test complete!${NC}"
echo -e "${YELLOW}Note: If targets are unhealthy, check container networking and DNS resolution${NC}"