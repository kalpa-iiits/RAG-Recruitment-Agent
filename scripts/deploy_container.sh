#!/bin/bash
# This script pulls and runs the latest Docker image from ECR

# Exit if any command fails
set -e

# Variables (will be set by GitHub Actions)
ECR_REGISTRY=${ECR_REGISTRY}
ECR_REPOSITORY=${ECR_REPOSITORY}
IMAGE_TAG=${IMAGE_TAG}

# Update system packages
sudo apt-get update

# Install Docker if not installed
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    sudo apt-get update
    sudo apt-get install -y docker-ce
    sudo usermod -aG docker ubuntu
fi

# Install Nginx if not installed
if ! command -v nginx &> /dev/null; then
    echo "Installing Nginx..."
    sudo apt-get install -y nginx
fi

# Install AWS CLI if not installed
if ! command -v aws &> /dev/null; then
    echo "Installing AWS CLI..."
    sudo apt-get install -y unzip
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
fi

# Create nginx config if it doesn't exist
if [ ! -f "/etc/nginx/sites-available/streamlit" ]; then
    echo "Creating Nginx configuration..."
    cat > /tmp/streamlit_nginx << 'EOL'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 86400;
    }
}
EOL
    sudo cp /tmp/streamlit_nginx /etc/nginx/sites-available/streamlit
    sudo ln -sf /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/
    sudo rm -f /etc/nginx/sites-enabled/default
    sudo nginx -t
    sudo systemctl restart nginx
fi

# Login to ECR
echo "Logging in to Amazon ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}

# Stop any running container
echo "Stopping any existing container..."
docker stop streamlit-container 2>/dev/null || true
docker rm streamlit-container 2>/dev/null || true

# Pull the latest image
echo "Pulling the latest image from ECR..."
docker pull ${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}

# Run the container
#
# Runtime config lives on the host: .dockerignore keeps .env out of every
# image layer, so nothing in the image supplies it. Started without this,
# the container comes up looking healthy and fails quietly — no
# OPENAI_API_KEY (the UI then asks each user for their own), a JWT_SECRET
# regenerated per process (every deploy logs everyone out), and a SQLite
# file inside the container that the next deploy throws away. Better to
# refuse to start than to serve that.
ENV_FILE=${ENV_FILE:-/opt/cvexpert/.env}
if [ ! -f "${ENV_FILE}" ]; then
    echo "ERROR: ${ENV_FILE} not found — create it before deploying." >&2
    echo "       See backend/.env.example for the variables it must set." >&2
    exit 1
fi

echo "Starting the container..."
# Bound to loopback, not 0.0.0.0: nginx terminates TLS and proxies to it, so
# publishing the port on every interface would put the API on the public
# internet on :8501, unencrypted and around the proxy.
docker run -d --name streamlit-container -p 127.0.0.1:8501:8501 --restart always \
    --env-file "${ENV_FILE}" \
    ${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}

echo "Deployment completed successfully!"