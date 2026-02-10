#!/usr/bin/env bash
set -euo pipefail

# Autonomous Eden Setup Script
# Checks dependencies, pulls models, creates directories, validates configuration.

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }

echo "========================================="
echo "  Autonomous Eden"
echo "  Setup"
echo "========================================="
echo ""

# ---- 1. Check Docker ----
echo "--- Checking dependencies ---"

if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | head -1)
    ok "Docker: $DOCKER_VERSION"
else
    fail "Docker not found. Install: https://docs.docker.com/get-docker/"
    exit 1
fi

if docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version --short 2>/dev/null || echo "unknown")
    ok "Docker Compose: $COMPOSE_VERSION"
else
    fail "Docker Compose not found. Install: https://docs.docker.com/compose/install/"
    exit 1
fi

# ---- 2. Check/Create .env ----
echo ""
echo "--- Configuration ---"

if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        warn ".env created from .env.example - please edit it"
        echo "     At minimum, set OLLAMA_HOST to your Ollama server address."
    else
        fail ".env.example not found"
        exit 1
    fi
else
    ok ".env exists"
fi

# Load .env
set -a
source .env
set +a

OLLAMA_HOST="${OLLAMA_HOST:-localhost}"
ok "Ollama host: $OLLAMA_HOST"

# ---- 3. Check Ollama connectivity ----
echo ""
echo "--- Checking Ollama ---"

OLLAMA_URL="http://${OLLAMA_HOST}:11434"
if curl -s --connect-timeout 5 "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
    ok "Ollama reachable at ${OLLAMA_URL}"
    
    # List available models
    MODELS=$(curl -s "${OLLAMA_URL}/api/tags" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for m in data.get('models', []):
        print(f\"  - {m['name']}\")
except: pass
" 2>/dev/null)
    
    if [ -n "$MODELS" ]; then
        echo "  Available models:"
        echo "$MODELS"
    fi
else
    warn "Ollama not reachable at ${OLLAMA_URL}"
    echo "     Make sure Ollama is running and accessible."
    echo "     Set OLLAMA_HOST in .env to your Ollama server address."
fi

# ---- 4. Pull models ----
echo ""
echo "--- Models ---"

EVE_MODEL="${EVE_MODEL:-llama3:8b}"
ADAM_MODEL="${ADAM_MODEL:-forzer/GigaChat3-10B-A1.8B}"

check_model() {
    local model="$1"
    local label="$2"
    if curl -s "${OLLAMA_URL}/api/tags" 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
names = [m['name'] for m in data.get('models', [])]
sys.exit(0 if any('$model' in n for n in names) else 1)
" 2>/dev/null; then
        ok "$label model '$model' available"
    else
        warn "$label model '$model' not found"
        read -p "     Pull it now? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "     Pulling $model (this may take a while)..."
            curl -s "${OLLAMA_URL}/api/pull" -d "{\"name\": \"$model\"}" | while read -r line; do
                STATUS=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
                if [ -n "$STATUS" ]; then
                    echo -ne "     $STATUS\r"
                fi
            done
            echo ""
            ok "Model $model pulled"
        fi
    fi
}

check_model "$EVE_MODEL" "EVE"
check_model "$ADAM_MODEL" "ADAM"

# ---- 5. Create directories ----
echo ""
echo "--- Directories ---"

for dir in logs data data/qdrant workspace Legacy Legacy/Archive; do
    mkdir -p "$dir"
done
ok "Required directories created"

# ---- 6. Summary ----
echo ""
echo "========================================="
echo "  Setup complete!"
echo "========================================="
echo ""
echo "  Quick start:"
echo ""
echo "    # Launch EVE (standard world with pressure)"
echo "    docker compose --profile standard up -d"
echo ""
echo "    # Launch ADAM in Garden of Eden"
echo "    docker compose --profile eden up -d"
echo ""
echo "    # Launch everything"
echo "    docker compose --profile standard --profile eden up -d"
echo ""
echo "    # Open dashboard"
echo "    open http://localhost:8110"
echo ""
echo "    # Watch thoughts"
echo "    tail -f logs/inner_monologue.jsonl | jq -r .thought"
echo ""
