#!/bin/bash

set -e

# Configuration
SERVICE_NAME="claude-proxy"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo_error "uv is not installed. Please install uv first."
    exit 1
fi

# Check if .env file exists
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo_error ".env file not found. Please create one based on .env.example"
    exit 1
fi

# Create systemd user directory if it doesn't exist
mkdir -p "$SYSTEMD_USER_DIR"

echo_info "Installing dependencies..."
cd "$SCRIPT_DIR"
uv sync

echo_info "Creating systemd service file..."

# Create the service file (fixed version without User= directive and with proper path setup)
cat > "$SYSTEMD_USER_DIR/$SERVICE_NAME.service" << EOF
[Unit]
Description=Claude to OpenAI Proxy Server
After=network.target

[Service]
Type=simple
WorkingDirectory=$SCRIPT_DIR
Environment=PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
EnvironmentFile=$SCRIPT_DIR/.env
ExecStart=/usr/bin/env uv run uvicorn server:app --host 0.0.0.0 --port 8088
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
EOF

echo_info "Service file created at $SYSTEMD_USER_DIR/$SERVICE_NAME.service"

# Stop the service if it's already running
echo_info "Stopping any existing service..."
systemctl --user stop "$SERVICE_NAME.service" 2>/dev/null || true

# Reload systemd user daemon
echo_info "Reloading systemd user daemon..."
systemctl --user daemon-reload

# Enable the service
echo_info "Enabling $SERVICE_NAME service..."
systemctl --user enable "$SERVICE_NAME.service"

# Start the service
echo_info "Starting $SERVICE_NAME service..."
systemctl --user start "$SERVICE_NAME.service"

# Check service status
sleep 3
if systemctl --user is-active --quiet "$SERVICE_NAME.service"; then
    echo_info "✅ Service $SERVICE_NAME is running successfully!"
    echo_info "Service status:"
    systemctl --user status "$SERVICE_NAME.service" --no-pager -l
else
    echo_error "❌ Service $SERVICE_NAME failed to start!"
    echo_error "Service status:"
    systemctl --user status "$SERVICE_NAME.service" --no-pager -l
    echo_error "Recent logs:"
    journalctl --user -u "$SERVICE_NAME.service" --no-pager -l -n 20
    exit 1
fi

echo_info "🎉 Setup complete!"
echo_info ""
echo_info "Your service is now running on http://0.0.0.0:8088"
echo_info ""
echo_info "Useful commands:"
echo_info "  Check status:    systemctl --user status $SERVICE_NAME"
echo_info "  View logs:       journalctl --user -u $SERVICE_NAME -f"
echo_info "  Restart service: systemctl --user restart $SERVICE_NAME"
echo_info "  Stop service:    systemctl --user stop $SERVICE_NAME"
echo_info "  Disable service: systemctl --user disable $SERVICE_NAME"
echo_info ""
echo_info "To enable linger (start service on boot without login):"
echo_info "  sudo loginctl enable-linger $USER"
