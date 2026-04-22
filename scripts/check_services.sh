#!/bin/bash
# Run this after 'docker-compose up -d' to verify all services are healthy.
# Usage: bash scripts/check_services.sh

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

check() {
  local name=$1
  local url=$2
  if curl -sf "$url" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} $name"
  else
    echo -e "${RED}✗${NC} $name — not reachable at $url"
  fi
}

echo ""
echo "FraudShield — Service Health Check"
echo "─────────────────────────────────"

check "MLflow         " "http://localhost:5001"
check "Airflow        " "http://localhost:8080/health"
check "MinIO API      " "http://localhost:9000/minio/health/live"
check "MinIO Console  " "http://localhost:9001"
check "Prometheus     " "http://localhost:9090/-/healthy"
check "Grafana        " "http://localhost:3000/api/health"

# Redis
if docker exec fraudshield-redis redis-cli ping 2>/dev/null | grep -q PONG; then
  echo -e "${GREEN}✓${NC} Redis"
else
  echo -e "${RED}✗${NC} Redis"
fi

# Postgres
if docker exec fraudshield-postgres pg_isready -U fraudshield 2>/dev/null | grep -q "accepting"; then
  echo -e "${GREEN}✓${NC} PostgreSQL"
else
  echo -e "${RED}✗${NC} PostgreSQL"
fi

# Kafka
if docker exec fraudshield-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list > /dev/null 2>&1; then
  echo -e "${GREEN}✓${NC} Kafka"
else
  echo -e "${RED}✗${NC} Kafka"
fi

echo ""
echo "UIs:"
echo "  MLflow:   http://localhost:5001"
echo "  Airflow:  http://localhost:8080  (admin / admin)"
echo "  MinIO:    http://localhost:9001  (minioadmin / minioadmin123)"
echo "  Grafana:  http://localhost:3000  (admin / admin)"
echo "  Prometheus: http://localhost:9090"
echo ""
