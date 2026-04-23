#!/usr/bin/env bash
# ========================================================================
# Install systemd service for Atlas Solver
# Run as root: bash install_service.sh
# ========================================================================
set -euo pipefail

APP_DIR="${APP_DIR:-/srv/atlas/OCR_Atlas_annotation_final_project}"
APP_USER="${APP_USER:-atlas}"

cat > /etc/systemd/system/atlas-solver.service <<EOF
[Unit]
Description=Atlas Solver - Automated Episode Labeling
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${APP_USER}
Group=${APP_USER}
WorkingDirectory=${APP_DIR}
ExecStart=/bin/bash ${APP_DIR}/deploy/run_solver.sh
Restart=on-failure
RestartSec=30
StandardOutput=append:${APP_DIR}/outputs/solver_service.log
StandardError=append:${APP_DIR}/outputs/solver_service.log
Environment=HOME=/home/${APP_USER}
Environment=NODE_OPTIONS=--no-warnings

# Resource limits
LimitNOFILE=65536
MemoryMax=12G
CPUQuota=700%

# Security hardening
NoNewPrivileges=false
ProtectSystem=false

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable atlas-solver.service

echo "[service] installed and enabled atlas-solver.service"
echo "[service] start:   systemctl start atlas-solver"
echo "[service] status:  systemctl status atlas-solver"
echo "[service] logs:    journalctl -u atlas-solver -f"
echo "[service] stop:    systemctl stop atlas-solver"
