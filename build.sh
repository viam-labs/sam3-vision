#!/bin/sh
set -e
cd "$(dirname "$0")"
tar --exclude='__pycache__' --exclude='*.pyc' -czf module.tar.gz \
    meta.json run.sh setup.sh requirements.txt src
echo "Built module.tar.gz"
