#!/usr/bin/env bash
# ========================================================================
# Install systemd services for Atlas Solver + Monitor Daemon
# Run as root: bash deploy/install_service.sh
# ========================================================================
set -euo pipefail

APP_DIR="${APP_DIR:-/srv/atlas/OCR_Atlas_annotation_final_project}"

echo "========================================"
echo " Atlas Service Installer"
echo "========================================"
echo "APP_DIR: $APP_DIR"
echo ""

# Ensure output directories exist
mkdir -p "$APP_DIR/outputs/hetzner_production"

# --- 1. Install Atlas Solver Service ---
echo "[1/4] Installing atlas-solver.service..."
cp "$APP_DIR/deploy/atlas-solver.service" /etc/systemd/system/atlas-solver.service

# --- 2. Install Atlas Monitor Service ---
echo "[2/4] Installing atlas-monitor.service..."
cp "$APP_DIR/deploy/atlas-monitor.service" /etc/systemd/system/atlas-monitor.service

# --- 3. Reload and enable ---
echo "[3/4] Reloading systemd and enabling services..."
systemctl daemon-reload
systemctl enable atlas-solver.service
systemctl enable atlas-monitor.service

# --- 4. Summary ---
echo "[4/4] Done!"
echo ""
echo "========================================"
echo " Services installed successfully"
echo "========================================"
echo ""
echo "Solver service:"
echo "  Start:   systemctl start atlas-solver"
echo "  Stop:    systemctl stop atlas-solver"
echo "  Status:  systemctl status atlas-solver"
echo "  Logs:    journalctl -u atlas-solver -f"
echo ""
echo "Monitor daemon:"
echo "  Start:   systemctl start atlas-monitor"
echo "  Stop:    systemctl stop atlas-monitor"
echo "  Status:  systemctl status atlas-monitor"
echo "  Logs:    tail -f $APP_DIR/outputs/monitor_daemon.log"
echo ""
echo "Quick commands:"
echo "  Start both:   systemctl start atlas-solver atlas-monitor"
echo "  Stop both:    systemctl stop atlas-solver atlas-monitor"
echo "  Restart both: systemctl restart atlas-solver atlas-monitor"
echo ""
echo "NOTE: Stop any manually-running solver before starting the service:"
echo "  pkill -f atlas_web_auto_solver; pkill -f google-chrome; pkill -f Xvfb"
echo "  rm -f /tmp/.X99-lock /tmp/.X11-unix/X99"
