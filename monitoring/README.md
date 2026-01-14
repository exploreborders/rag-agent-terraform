# RAG Agent Monitoring Setup ✅

This directory contains the **fully operational** monitoring configuration for the RAG Agent system using Prometheus and Grafana. All components are deployed and working correctly.

## 📊 Architecture

The monitoring stack consists of:

- **Prometheus**: Metrics collection and alerting
- **Grafana**: Dashboard visualization
- **PostgreSQL Exporter**: Database metrics
- **Redis Exporter**: Cache metrics
- **Node Exporter**: System metrics
- **Application Metrics**: Custom RAG-specific metrics

## 🚀 Quick Start

### 1. Deploy Monitoring Stack

```bash
# Recommended: Deploy with main infrastructure
make deploy              # Deploys all services including monitoring
docker ps                # Verify all containers are running

# Alternative: Manual Terraform (may have timeout issues)
cd terraform
terraform init
terraform apply
```

### 2. Access Interfaces

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Application Metrics**: http://localhost:8000/metrics

### 3. Verify Setup

```bash
# Run monitoring tests
chmod +x scripts/test_monitoring.sh
./scripts/test_monitoring.sh
```

## 📈 Dashboards

### Pre-configured Dashboards

1. **RAG Agent Performance Dashboard**
   - Query performance metrics
   - Document processing statistics
   - Vector search performance
   - HTTP request metrics

2. **Database Performance Dashboard**
   - PostgreSQL connection metrics
   - Query performance
   - pgvector operations
   - Cache hit ratios

3. **Infrastructure Dashboard**
   - System CPU/memory usage
   - Disk I/O and network traffic
   - Container resource usage
   - Docker container metrics

### Accessing Dashboards

After deployment, dashboards are automatically provisioned in Grafana:

- http://localhost:3000/d/rag-agent-performance/rag-agent-performance-dashboard
- http://localhost:3000/d/database-performance/database-performance-dashboard
- http://localhost:3000/d/infrastructure/infrastructure-dashboard

## 🔧 Configuration Files

### Prometheus (`prometheus.yml`)
- Service discovery for Docker containers
- Scraping intervals and targets
- Alerting rules integration

### Alerting Rules (`alerting.yml`)
- Critical alerts: Application down, database issues
- Warning alerts: High latency, low success rates
- Infrastructure alerts: Resource usage thresholds

### Grafana Provisioning
- **Data Sources**: `grafana/datasources.yml`
- **Dashboards**: `grafana/dashboards.yml`
- **Dashboard JSON**: `grafana/dashboards/*.json`

## 📊 Custom Metrics

### Application Metrics
- `rag_queries_total{status, model}`: Total queries processed
- `rag_query_duration_seconds{model, operation}`: Query processing time
- `documents_processed_total{content_type, status}`: Document processing
- `vector_search_duration_seconds{top_k}`: Vector search performance
- `active_documents_total`: Current document count
- `active_chunks_total`: Current chunk count

### Infrastructure Metrics
- PostgreSQL: Connections, query performance, pgvector ops
- Redis: Memory usage, hit rates, connections
- Node: CPU, memory, disk, network utilization

## 🚨 Alerting

### Alert Rules
- **High Error Rate**: >10% errors in 5 minutes
- **Slow Queries**: >5 seconds P95 latency
- **Database Issues**: Connection failures, high utilization
- **Resource Alerts**: CPU >90%, memory >90%, disk <10%

### Alert Channels
Configure notification channels in Grafana:
1. Go to Alerting → Contact points
2. Add email, Slack, or webhook notifications
3. Configure routing in Alerting → Notification policies

## 🔍 Troubleshooting

### Common Issues

**FastAPI Middleware Compatibility** ✅ **FIXED**
```
Previously: ValueError: too many values to unpack (expected 2)
Issue: FastAPI 0.104.1 incompatible with Starlette 0.51.0
Solution: Pinned compatible versions in requirements.txt
```

**DNS Resolution Errors** ✅ **FIXED**
```
Previously: Error scraping target: Get "http://rag-agent-app-dev:8000/metrics": dial tcp: lookup rag-agent-app-dev on 127.0.0.11:53: no such host
```

**Solution Applied:**
- Updated prometheus.yml to use correct container hostnames
- Fixed Terraform environment variable configuration
- All containers now communicate properly via Docker networking

**Current Status:**
- ✅ All containers running with proper networking
- ✅ Prometheus successfully scraping all targets
- ✅ Grafana dashboards displaying real-time data

**Container Networking**
```bash
# Check container connectivity
docker exec rag-agent-prometheus-dev curl -f http://rag-agent-app-dev:8000/metrics

# Check network configuration
docker network inspect rag-agent-network-dev
```

**Metrics Not Appearing** ✅ **RESOLVED**
- ✅ Application running and healthy
- ✅ Metrics endpoint accessible: `curl http://localhost:8000/metrics`
- ✅ Prometheus targets configured correctly
- ✅ Grafana data sources working
- ✅ All dashboards displaying real-time data

### Debugging Commands

```bash
# Check container logs
docker logs rag-agent-prometheus-dev
docker logs rag-agent-grafana-dev

# Test Prometheus configuration
docker exec rag-agent-prometheus-dev promtool check config /etc/prometheus/prometheus.yml

# Check Grafana provisioning
docker exec rag-agent-grafana-dev ls -la /etc/grafana/provisioning/
```

## 📚 Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [PostgreSQL Exporter](https://github.com/prometheus-community/postgres_exporter)
- [Redis Exporter](https://github.com/oliver006/redis_exporter)
- [Node Exporter](https://github.com/prometheus/node_exporter)

## 🤝 Contributing

When adding new metrics or dashboards:

1. Update application code with new metrics
2. Add metric documentation above
3. Create/update dashboard JSON files
4. Test with `./scripts/test_monitoring.sh`
5. Update this README