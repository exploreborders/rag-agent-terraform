# Multi-Agenten-RAG-System: Implementierungsplan
## Übersicht

Dieses Dokument enthält den detaillierten Plan für die Integration eines Multi-Agenten-Systems in das bestehende RAG-Agent-System unter Verwendung von LangGraph, Docker MCP Toolkit und Redis Message Queue.

## Projekt-Kontext

### Bestehendes System
- **FastAPI Backend** mit REST-API (Port 8000)
- **PostgreSQL + pgvector** für Vektor-Datenbank
- **Redis** für Caching und Sessions
- **Ollama** für lokale AI-Modelle (llama3.2, embeddinggemma)
- **Document Processing Pipeline** (PDF, Text, Images)
- **Prometheus Metriken** und Health Checks

### Neue Anforderungen
- **4-Agenten-Architektur** mit LangGraph
- **Docker MCP Toolkit Integration** (200+ Tools)
- **Redis Message Queue** für Agenten-Kommunikation
- **API-Kompatibilität** beibehalten
- **Keine User-Management** (vereinfachte Sicherheit)
- **Docker-Deployment** für alle Services

## Entscheidungen und Annahmen

### Architektur-Entscheidungen
1. **LangGraph als Orchestrierungs-Framework** - Bietet State-Management und Graph-basierte Workflows
2. **Redis als Message Queue** - Für Agenten-Kommunikation und Task-Verteilung
3. **Docker MCP Toolkit** - Für isolierte Tool-Ausführung und Sicherheit
4. **PostgreSQL Checkpointer** - Für LangGraph State-Persistenz
5. **Vereinfachte Sicherheit** - DataSanitizer ohne User-Level-Management

### Sicherheits-Entscheidungen
1. **Keine kritischen Daten an LLM** - Persönliche Informationen und Unternehmensdaten werden gefiltert
2. **Metadata-only State** - Nur IDs, Scores und Metadaten im Graph-State
3. **Container-Isolation** - MCP Tools laufen in separaten Docker-Containern

### Tool-Auswahl
1. **MCP Search Tools**: Brave Search, ArXiv, Perplexity
2. **MCP Code Tools**: GitHub Official, Context7
3. **Priorität**: Search und Code Tools für RAG-Erweiterung

## System-Architektur

### Container-Setup (docker-compose.yml)

```yaml
services:
  rag-api:           # FastAPI App mit LangGraph
  postgres:          # Vektor-Datenbank + Checkpointer
  redis:             # Message Queue + Caching
  ollama:            # AI-Modelle
  mcp-coordinator:   # MCP Tool Orchestrierung
```

### Agenten-Architektur

#### 1. Query Processor Agent
- **Aufgabe**: Query-Analyse, Intent-Erkennung, Agenten-Dispatch
- **Input**: Raw Query
- **Output**: Sanitized Query, Agent Tasks, Intent Classification

#### 2. Retrieval Agent
- **Aufgabe**: Bestehende RAG-Suche in Vektor-Datenbank
- **Input**: Sanitized Query, Document IDs
- **Output**: Metadata-only Search Results

#### 3. MCP Research Agent
- **Aufgabe**: Externe Recherche über MCP Tools
- **Input**: Sanitized Query
- **Output**: Web Search Results, Academic Papers

#### 4. MCP Code Agent
- **Aufgabe**: Code-bezogene Suche über MCP Tools
- **Input**: Sanitized Query
- **Output**: GitHub Code, Documentation Results

#### 5. Results Aggregator Agent
- **Aufgabe**: Kombination aller Agenten-Ergebnisse
- **Input**: Alle Agenten-Results
- **Output**: Ranked, Deduplicated Results

#### 6. Response Generator Agent
- **Aufgabe**: Finale Antwort-Generierung mit Context
- **Input**: Aggregated Results
- **Output**: AI-Generated Response

#### 7. Validation Agent
- **Aufgabe**: Qualitäts- und Sicherheitsprüfung
- **Input**: Generated Response
- **Output**: Validated Final Response

## Technische Spezifikationen

### State Management

```python
class DockerMultiAgentRAGState(TypedDict):
    query: str                           # Original Query
    sanitized_query: str                 # Gefilterte Query
    agent_tasks: Dict[str, Any]         # Agenten-Zuweisungen
    agent_results: Dict[str, Any]       # Agenten-Ergebnisse
    retrieved_metadata: List[Dict]      # Sichere Suchergebnisse
    mcp_search_results: Optional[Dict]  # MCP Recherche
    mcp_code_results: Optional[Dict]    # MCP Code-Ergebnisse
    final_response: Optional[str]       # Finale Antwort
    confidence_score: float             # Vertrauens-Score
    sources: List[Dict]                 # Zitierte Quellen
```

### Sicherheits-Filter

```python
SENSITIVE_PATTERNS = {
    'personal': [
        r'\b\d{3}-\d{2}-\d{4}\b',        # SSN
        r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',  # Kreditkarten
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
    ],
    'corporate': [
        r'\b(confidential|internal|secret)\b',
        r'\b(password|token|key|credential)\b',
        r'\b(employee|salary|compensation)\b',
    ]
}
```

### MCP Tool Integration

#### Aktivierte Tools
- **Search**: brave-search, arxiv, perplexity
- **Code**: github-official, context7
- **Zukünftig**: mongodb, elasticsearch, notion

#### Tool-Ausführung
- **Container-basiert**: Jeder Tool läuft in isoliertem Docker-Container
- **Timeout**: 30 Sekunden pro Tool-Ausführung
- **Cleanup**: Automatische Container-Bereinigung
- **OAuth**: Sichere Authentifizierung für externe Services

## API-Spezifikation

### Bestehende Endpoints (Kompatibilität)
```http
POST /query                    # Legacy RAG Query
GET  /documents               # Document List
POST /documents/upload        # Document Upload
DELETE /documents/{id}        # Document Delete
GET  /health                  # Health Check
GET  /metrics                 # Prometheus Metrics
```

### Neue Endpoints
```http
POST /agents/query            # Multi-Agenten Query
POST /agents/stream           # Streaming Multi-Agenten Query
GET  /agents/status           # Agenten-Status
POST /agents/configure        # Agenten-Konfiguration
```

## Metriken und Monitoring

### Prometheus Metriken
- `rag_agent_execution_duration_seconds` - Agenten-Ausführungszeit
- `rag_agent_success_total` - Erfolgreiche Agenten-Ausführungen
- `rag_mcp_tool_usage_total` - MCP Tool Nutzung
- `rag_security_filters_applied_total` - Sicherheitsfilter-Anwendungen
- `rag_docker_containers_active` - Aktive Docker-Container

### Monitoring-Komponenten
- **Grafana Dashboards** für Agenten-Performance
- **Health Checks** für alle Agenten-Container
- **Audit Logging** für Sicherheits-Events
- **Container-Metriken** (CPU, Memory, Network)

## Implementierungs-Phasen

### Phase 1: Grundinfrastruktur (1 Woche)
1. Docker-Compose Setup mit allen Services
2. MCP-Coordinator Container implementieren
3. Redis Message Queue für Agenten-Kommunikation
4. Basis LangGraph mit PostgreSQL-Persistenz

### Phase 2: Agenten-Entwicklung (2 Wochen)
1. Query Processor Agent (Intent-Klassifikation)
2. Retrieval Agent (bestehendes RAG integrieren)
3. MCP Research Agent (Brave, ArXiv, Perplexity)
4. MCP Code Agent (GitHub, Context7)
5. Results Aggregator Agent
6. Response Generator Agent
7. Validation Agent

### Phase 3: Sicherheit & Kommunikation (1 Woche)
1. DataSanitizer implementieren (ohne User-Management)
2. Sicherheits-Gates für kritische Daten
3. OAuth für MCP Tools konfigurieren
4. Audit-Logging für Agenten-Aktivitäten

### Phase 4: API & Monitoring (1 Woche)
1. Legacy-API-Kompatibilität sicherstellen
2. Neue Multi-Agenten-Endpoints hinzufügen
3. Umfassende Metriken implementieren
4. Agenten-Performance-Tracking aktivieren

### Phase 5: Testing & Deployment (1 Woche)
1. Integrationstests für Docker-Setup
2. Performance-Benchmarks und Optimierungen
3. Sicherheitstests für Datenfilterung
4. Produktions-Deployment und Dokumentation

## Deployment-Workflow

### Container-Build
```bash
# Vollständiges System starten
docker-compose up -d

# MCP Toolkit initialisieren
docker exec rag-api python -c "from app.mcp_coordinator import initialize_tools; initialize_tools()"

# Ollama Modelle laden
docker exec ollama ollama pull llama3.2:latest
docker exec ollama ollama pull embeddinggemma:latest
```

### System-Initialisierung
```bash
# Datenbank-Migration
docker exec rag-api python -c "from app.vector_store import VectorStore; vs = VectorStore(); await vs.initialize_schema()"

# LangGraph kompilieren
docker exec rag-api python -c "from app.multi_agent_graph import create_and_compile_graph; compiled_graph = create_and_compile_graph()"
```

## Risiken und Mitigation

### Technische Risiken
1. **MCP Tool Reliability** - Container-Timeouts und Fehlerbehandlung
2. **Redis Message Queue** - Message Loss Prevention und Retry-Mechanismen
3. **LangGraph Complexity** - State-Management und Debugging

### Sicherheitsrisiken
1. **Data Leakage** - Sicherstellen, dass keine kritischen Daten LLM erreichen
2. **Container Escapes** - MCP Tool Isolation und Resource Limits
3. **API Abuse** - Rate Limiting und Authentication

### Performance-Risiken
1. **Agent Parallelization** - Ressourcen-Kontention bei vielen Agenten
2. **Database Load** - Vektor-Suche und Checkpointer-Performance
3. **Network Latency** - MCP Tool Response Times

## Erfolgskriterien

### Funktionale Anforderungen
- ✅ Bestehende API-Endpoints funktionieren weiterhin
- ✅ Multi-Agenten-Queries liefern erweiterte Ergebnisse
- ✅ MCP Tools sind erfolgreich integriert
- ✅ Kritische Daten werden gefiltert

### Nicht-Funktionale Anforderungen
- ✅ Query-Latenz < 5 Sekunden für einfache Queries
- ✅ System bleibt stabil unter Last
- ✅ Sicherheitsfilter funktionieren korrekt
- ✅ Monitoring bietet vollständige Observabilität

## Abhängigkeiten und Prerequisites

### Software-Abhängigkeiten
- Docker Desktop mit MCP Toolkit
- PostgreSQL 15+
- Redis 7+
- Python 3.11+
- Ollama für AI-Modelle

### MCP Tool Prerequisites
- Brave Search API Key (falls OAuth erforderlich)
- GitHub OAuth Token
- ArXiv API Access
- Context7 API Access

## Kommunikationsplan

### Team-Koordination
- Wöchentliche Standup-Meetings
- Code Reviews für alle Agenten-Implementierungen
- Sicherheits-Reviews für DataSanitizer

### Dokumentation
- API-Dokumentation für neue Endpoints
- Architektur-Diagramme
- Sicherheitsrichtlinien
- Deployment-Guide

## Meilensteine und Deadlines

- **Phase 1 Ende**: Woche 1 - Docker-Infrastruktur vollständig
- **Phase 2 Ende**: Woche 3 - Alle Agenten implementiert und getestet
- **Phase 3 Ende**: Woche 4 - Sicherheit und Kommunikation fertig
- **Phase 4 Ende**: Woche 5 - API und Monitoring komplett
- **Phase 5 Ende**: Woche 6 - Produktionsbereit deployed

## Kostenabschätzung

### Infrastruktur-Kosten
- Docker Container: Minimal (lokale Ausführung)
- PostgreSQL/Redis: Minimal (lokale Instanzen)
- MCP API Calls: Variable (je nach Tool-Nutzung)

### Entwicklungsaufwand
- **Senior Developer**: 4-6 Wochen Vollzeit
- **Testing & QA**: 1 Woche
- **Dokumentation**: 0.5 Wochen

## Notfallplan

### Rollback-Strategie
1. Bestehende API-Endpoints bleiben aktiv
2. LangGraph kann deaktiviert werden (Fallback auf alten RAG-Agent)
3. MCP Tools können einzeln deaktiviert werden
4. Vollständiger Rollback durch Git-Reset möglich

### Monitoring-Alerts
- Agenten-Performance-Degradation
- Sicherheitsfilter-Fehler
- Container-Health-Issues
- API-Response-Time-Spikes

---

## Änderungsprotokoll

| Datum | Version | Änderung | Autor |
|-------|---------|----------|-------|
| 2024-01-XX | 1.0 | Initiale Plan-Erstellung | System |

## Anlagen

### Anlage 1: Technische Architektur-Diagramme
### Anlage 2: API-Spezifikationen
### Anlage 3: Sicherheitsrichtlinien
### Anlage 4: Deployment-Scripts
### Anlage 5: Testfälle