#!/bin/bash

SERVICE_NAME="claude-proxy"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo_info "Stopping $SERVICE_NAME service..."
systemctl --user stop "$SERVICE_NAME.service" 2>/dev/null || true

echo_info "Disabling $SERVICE_NAME service..."
systemctl --user disable "$SERVICE_NAME.service" 2>/dev/null || true

echo_info "Removing service file..."
rm -f "$SYSTEMD_USER_DIR/$SERVICE_NAME.service"

echo_info "Reloading systemd user daemon..."
systemctl --user daemon-reload

echo_info "✅ Service $SERVICE_NAME has been stopped and removed!"
