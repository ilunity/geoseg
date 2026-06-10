#!/bin/sh
set -e

if [ ! -f /app/checkpoints/lulc_checkpoint_1.pth ]; then
  echo "Seeding checkpoints volume from image..."
  mkdir -p /app/checkpoints/custom
  cp -n /app/checkpoints_default/*.pth /app/checkpoints/ 2>/dev/null || true
  if [ -f /app/checkpoints_default/custom/registry.json ]; then
    cp -n /app/checkpoints_default/custom/registry.json /app/checkpoints/custom/
  fi
fi

exec "$@"
