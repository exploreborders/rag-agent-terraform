---
name: Infrastructure Issue
about: Report issues with deployment, infrastructure, or DevOps
title: '[INFRA] '
labels: ['infrastructure', 'devops', 'triage']
assignees: ''
---

## 🏗️ Infrastructure Issue Description
Describe the infrastructure or deployment issue you're experiencing.

## 🔧 Environment Details
- **Cloud Provider**: [e.g., AWS, GCP, Azure, Local]
- **Deployment Method**: [e.g., Terraform, Docker Compose, Kubernetes]
- **Infrastructure Component**:
  - [ ] Terraform Configuration
  - [ ] Docker Services
  - [ ] PostgreSQL Database
  - [ ] Redis Cache
  - [ ] Ollama Service
  - [ ] Network Configuration
  - [ ] CI/CD Pipeline
  - [ ] Monitoring/Logging
  - [ ] Security Configuration
  - [ ] Other (specify)

## ❌ Error Details
**Error Message:**
```
Paste the full error message here
```

**Error Location:**
- File: `path/to/file`
- Line: `line_number`
- Command: `terraform apply` or `docker-compose up`, etc.

**Full Logs:**
```
Paste relevant log output here (use code blocks for formatting)
```

## 🔄 Steps to Reproduce
1. Run command: `terraform init`
2. Execute: `terraform plan`
3. Deploy with: `terraform apply`
4. Error occurs at: `specific step`

## ✅ Expected Behavior
What should happen when the infrastructure is deployed?

## 📊 Current Status
- [ ] **Blocking**: Cannot deploy or access the system
- [ ] **Degraded**: System partially working but with issues
- [ ] **Working**: Infrastructure deployed but needs optimization
- [ ] **Planning**: Infrastructure design or configuration issue

## 🔍 Troubleshooting Done
What have you tried to resolve this issue?

- [ ] Checked Terraform version compatibility
- [ ] Verified Docker daemon is running
- [ ] Confirmed network connectivity
- [ ] Reviewed resource limits and quotas
- [ ] Checked service health endpoints
- [ ] Examined system logs
- [ ] Tested with different configurations

## 📋 Configuration
**Relevant Configuration:**
```hcl
# Terraform configuration
resource "docker_container" "example" {
  # configuration here
}
```

**Environment Variables:**
```bash
# Key environment variables
DATABASE_URL=postgresql://...
OLLAMA_BASE_URL=http://...
```

**System Resources:**
- CPU: [e.g., 4 cores]
- RAM: [e.g., 8GB]
- Disk: [e.g., 50GB]
- OS: [e.g., Ubuntu 22.04, macOS 14.0]

## 🎯 Impact
- **Severity**: [Critical/High/Medium/Low]
- **Affected Users**: [All users, Development team, Specific environment]
- **Business Impact**: [System down, Development blocked, Performance issues]

## 🔗 Related
- **Related Issues**: #123, #456
- **Terraform Version**: 1.5.7
- **Docker Version**: 24.0.0
- **Provider Versions**: kreuzwerker/docker v3.6.2

## 📝 Workaround
Is there a temporary workaround for this infrastructure issue?