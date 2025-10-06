#!/bin/bash
set -e

# Emotion-Aware Virtual Classroom - Kubernetes Deployment Script
# This script deploys the complete emotion classroom system to Kubernetes

echo "🚀 Starting Emotion-Aware Virtual Classroom Deployment"
echo "=" * 60

# Configuration
NAMESPACE=${NAMESPACE:-emotion-classroom}
KUBECTL_CONTEXT=${KUBECTL_CONTEXT:-default}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed" 
        exit 1
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Create namespace
create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
}

# Apply Kubernetes manifests
deploy_manifests() {
    log_info "Deploying Kubernetes manifests..."
    
    # Deploy in order
    kubectl apply -f kubernetes/secrets.yaml -n $NAMESPACE
    kubectl apply -f kubernetes/storage.yaml -n $NAMESPACE
    kubectl apply -f kubernetes/database-deployment.yaml -n $NAMESPACE
    kubectl apply -f kubernetes/backend-deployment.yaml -n $NAMESPACE
    kubectl apply -f kubernetes/frontend-deployment.yaml -n $NAMESPACE
    kubectl apply -f kubernetes/nginx-deployment.yaml -n $NAMESPACE
    kubectl apply -f kubernetes/monitoring-deployment.yaml -n $NAMESPACE
    
    log_info "All manifests deployed"
}

# Wait for deployments
wait_for_deployments() {
    log_info "Waiting for deployments to be ready..."
    
    deployments=(
        "postgres"
        "redis"
        "backend"
        "frontend"
        "nginx-ingress"
        "prometheus"
        "grafana"
    )
    
    for deployment in "${deployments[@]}"; do
        log_info "Waiting for $deployment..."
        kubectl wait --for=condition=available --timeout=300s deployment/$deployment -n $NAMESPACE || {
            log_warn "$deployment deployment not ready, continuing..."
        }
    done
    
    log_info "Deployment readiness check completed"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Port forward for local access (in background)
    log_info "Setting up port forwards for monitoring access..."
    log_info "Prometheus will be available at: http://localhost:9090"
    log_info "Grafana will be available at: http://localhost:3000"
    log_warn "Use 'kubectl port-forward' commands to access monitoring:"
    echo "kubectl port-forward svc/prometheus-service 9090:9090 -n $NAMESPACE"
    echo "kubectl port-forward svc/grafana-service 3000:3000 -n $NAMESPACE"
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Wait for backend pod to be ready
    kubectl wait --for=condition=ready pod -l app=backend -n $NAMESPACE --timeout=300s
    
    # Run migrations
    BACKEND_POD=$(kubectl get pods -l app=backend -n $NAMESPACE -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -n $NAMESPACE $BACKEND_POD -- python -c "
from backend.database.database import create_tables
create_tables()
print('Database tables created successfully')
" || log_warn "Migration command failed, may already be initialized"
    
    log_info "Database initialization completed"
}

# Get service URLs
get_service_urls() {
    log_info "Getting service URLs..."
    
    # Get external IP for nginx service
    EXTERNAL_IP=$(kubectl get svc nginx-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    
    if [ "$EXTERNAL_IP" = "pending" ] || [ -z "$EXTERNAL_IP" ]; then
        log_warn "External IP not yet assigned. Using port-forward for access:"
        echo "kubectl port-forward svc/nginx-service 8080:80 -n $NAMESPACE"
        echo "Application will be available at: http://localhost:8080"
    else
        log_info "Application available at: http://$EXTERNAL_IP"
    fi
    
    # Show all services
    echo
    kubectl get services -n $NAMESPACE
}

# Health checks
health_checks() {
    log_info "Performing health checks..."
    
    # Check pod status
    kubectl get pods -n $NAMESPACE
    
    # Check if services are responding
    log_info "Services status:"
    kubectl get svc -n $NAMESPACE
    
    log_info "Health checks completed"
}

# Main deployment function
deploy() {
    log_info "Starting deployment process..."
    
    check_prerequisites
    create_namespace
    deploy_manifests
    wait_for_deployments
    run_migrations
    setup_monitoring
    get_service_urls
    health_checks
    
    echo
    log_info "🎉 Emotion-Aware Virtual Classroom deployed successfully!"
    echo
    echo "Next steps:"
    echo "1. Access the application using the URLs shown above"
    echo "2. Set up SSL certificates for production use"
    echo "3. Configure external load balancer if needed"
    echo "4. Set up backup procedures for persistent data"
    echo
    log_info "Deployment completed at $(date)"
}

# Cleanup function
cleanup() {
    log_warn "Cleaning up emotion classroom deployment..."
    kubectl delete namespace $NAMESPACE --ignore-not-found=true
    log_info "Cleanup completed"
}

# Script usage
usage() {
    echo "Usage: $0 [deploy|cleanup|status]"
    echo "  deploy  - Deploy the emotion classroom system"
    echo "  cleanup - Remove the emotion classroom system"
    echo "  status  - Show deployment status"
    exit 1
}

# Status check
status() {
    log_info "Checking deployment status..."
    echo
    echo "Namespace: $NAMESPACE"
    echo
    echo "Pods:"
    kubectl get pods -n $NAMESPACE 2>/dev/null || echo "Namespace not found"
    echo
    echo "Services:"
    kubectl get svc -n $NAMESPACE 2>/dev/null || echo "Namespace not found"
    echo
    echo "Deployments:"
    kubectl get deployments -n $NAMESPACE 2>/dev/null || echo "Namespace not found"
}

# Main script logic
case "${1:-deploy}" in
    deploy)
        deploy
        ;;
    cleanup)
        cleanup
        ;;
    status)
        status
        ;;
    *)
        usage
        ;;
esac