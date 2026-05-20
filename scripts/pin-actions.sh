#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") [-c|--check] [-h|--help]"
  echo ""
  echo "Options:"
  echo "  -c, --check  check only, do not modify files"
  echo "  -h, --help   show this message"
}

CHECK=false
for arg in "$@"; do
  case "$arg" in
    -c|--check) CHECK=true ;;
    -h|--help)  usage; exit 0 ;;
    *) echo "Error: unknown option '$arg'"; usage; exit 1 ;;
  esac
done

cd "$(dirname "$0")/.."

if [[ "$CHECK" == true ]]; then
  echo "Checking action pins..."
  pinact run --check

  echo "Auditing workflows with zizmor..."
  zizmor .github/workflows/ --pedantic

  echo "All checks passed"
else
  echo "Pinning actions to latest SHAs..."
  pinact run

  echo "Fixing workflows with zizmor..."
  zizmor .github/workflows/ --fix=all --pedantic

  echo "Done"
fi
