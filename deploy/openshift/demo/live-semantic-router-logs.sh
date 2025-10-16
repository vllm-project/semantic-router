#!/bin/bash
#
# Live Demo Log Viewer for Semantic Router
#
# This script tails the semantic-router logs and highlights interesting events:
# - Classification decisions (category detected)
# - Model routing (which model was selected)
# - Jailbreak detection (security events)
# - PII detection
# - Cache hits
# - Request content
#
# Usage: ./live-demo-logs.sh
#

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${CYAN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║       SEMANTIC ROUTER - LIVE DEMO LOG VIEWER                               ║${NC}"
echo -e "${BOLD}${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}📡 Watching semantic-router logs in real-time...${NC}"
echo -e "${CYAN}Press Ctrl+C to stop${NC}"
echo ""
echo -e "${BOLD}Legend:${NC}"
echo -e "  ${GREEN}🔍 CLASSIFICATION${NC} - Category detection"
echo -e "  ${BLUE}🎯 ROUTING${NC}        - Model selection"
echo -e "  ${MAGENTA}🛡️  SECURITY${NC}      - Jailbreak/PII detection"
echo -e "  ${CYAN}💾 CACHE${NC}          - Cache hit/miss"
echo -e "  ${YELLOW}📨 REQUEST${NC}       - User request content"
echo ""
echo -e "${BOLD}${CYAN}────────────────────────────────────────────────────────────────────────────${NC}"
echo ""

# Tail logs and highlight interesting events
oc logs -n vllm-semantic-router-system deployment/semantic-router --follow --tail=0 2>/dev/null | while read -r line; do
    # Get timestamp if available
    timestamp=$(echo "$line" | grep -o '"ts":"[^"]*"' | cut -d'"' -f4 | cut -d'T' -f2 | cut -d'.' -f1)

    # Extract message
    msg=$(echo "$line" | grep -o '"msg":"[^"]*"' | cut -d'"' -f4)

    # Highlight REQUEST CONTENT
    if echo "$line" | grep -q "Received request body"; then
        content=$(echo "$line" | grep -o '"content": "[^"]*"' | cut -d'"' -f4)
        model=$(echo "$line" | grep -o '"model": "[^"]*"' | cut -d'"' -f4)
        if [ -n "$content" ]; then
            echo -e "${YELLOW}📨 [${timestamp}] REQUEST:${NC} ${BOLD}${content}${NC} ${CYAN}(model: ${model})${NC}"
        fi
    fi

    # Highlight CLASSIFICATION
    if echo "$line" | grep -q "Classification result"; then
        category=$(echo "$line" | grep -o 'category:[^,}]*' | cut -d':' -f2 | tr -d ' "')
        confidence=$(echo "$line" | grep -o 'confidence:[^,}]*' | cut -d':' -f2 | tr -d ' "')
        if [ -n "$category" ]; then
            echo -e "${GREEN}🔍 [${timestamp}] CLASSIFICATION:${NC} ${BOLD}${category}${NC} ${CYAN}(confidence: ${confidence})${NC}"
        fi
    fi

    # Highlight JAILBREAK DETECTION
    if echo "$line" | grep -q "BENIGN.*benign.*confidence"; then
        confidence=$(echo "$line" | grep -o 'confidence: [0-9.]*' | cut -d' ' -f2)
        echo -e "${GREEN}🛡️  [${timestamp}] SECURITY:${NC} ${BOLD}BENIGN${NC} ${CYAN}(confidence: ${confidence})${NC}"
    elif echo "$line" | grep -q "Jailbreak classification result"; then
        # Parse the jailbreak result - {0 0.99999964} means class 0 (benign) with confidence
        result=$(echo "$line" | grep -o '{[0-9 .]*}' | tr -d '{}')
        class=$(echo "$result" | awk '{print $1}')
        conf=$(echo "$result" | awk '{print $2}')
        if [ "$class" = "0" ]; then
            echo -e "${GREEN}🛡️  [${timestamp}] JAILBREAK CHECK:${NC} ${BOLD}BENIGN${NC} ${CYAN}(confidence: ${conf})${NC}"
        else
            echo -e "${RED}🛡️  [${timestamp}] JAILBREAK CHECK:${NC} ${BOLD}${RED}THREAT DETECTED${NC} ${YELLOW}(class: ${class}, conf: ${conf})${NC}"
        fi
    fi

    # Highlight PII DETECTION
    if echo "$line" | grep -qi "PII policy check passed\|No PII"; then
        echo -e "${GREEN}🔒 [${timestamp}] PII:${NC} ${BOLD}No PII detected - Safe${NC}"
    elif echo "$line" | grep -qi "PII.*blocked\|PII.*rejected"; then
        echo -e "${RED}🔒 [${timestamp}] PII:${NC} ${BOLD}${RED}PII DETECTED & BLOCKED${NC}"
    fi
    # Skip generic PII messages that are just informational

    # Highlight MODEL ROUTING
    if echo "$msg" | grep -qi "Routing to model"; then
        routed_model=$(echo "$msg" | grep -o 'Model-[AB]')
        if [ -n "$routed_model" ]; then
            if [ "$routed_model" == "Model-A" ]; then
                echo -e "${BLUE}🎯 [${timestamp}] ROUTING:${NC} ${BOLD}${BLUE}${routed_model}${NC}"
            else
                echo -e "${BLUE}🎯 [${timestamp}] ROUTING:${NC} ${BOLD}${MAGENTA}${routed_model}${NC}"
            fi
        fi
    fi

    # Highlight SELECTED MODEL (with category)
    if echo "$msg" | grep -qi "Selected model"; then
        category=$(echo "$msg" | grep -o 'category [a-z ]*' | sed 's/category //' | tr '[:lower:]' '[:upper:]')
        selected_model=$(echo "$msg" | grep -o 'Model-[AB]')
        score=$(echo "$msg" | grep -o 'score [0-9.]*' | sed 's/score //')
        if [ -n "$selected_model" ]; then
            echo -e "${CYAN}🔍 [${timestamp}] CLASSIFIED:${NC} ${BOLD}${MAGENTA}${category}${NC} (score: ${score}) → ${CYAN}${selected_model}${NC}"
        fi
    fi

    # Highlight CACHE HITS
    if echo "$line" | grep -q "cache_hit"; then
        similarity=$(echo "$line" | grep -o '"similarity":[^,]*' | cut -d':' -f2)
        query=$(echo "$line" | grep -o '"query":"[^"]*"' | cut -d'"' -f4)
        if [ -n "$query" ]; then
            echo -e "${CYAN}💾 [${timestamp}] CACHE HIT:${NC} ${similarity} - ${query}"
        fi
    fi

    # Highlight REASONING MODE
    if echo "$line" | grep -qi "reasoning mode\|chain.of.thought"; then
        echo -e "${MAGENTA}🧠 [${timestamp}] REASONING:${NC} ${BOLD}Chain-of-thought enabled${NC}"
    fi

    # Highlight ERRORS
    if echo "$line" | grep -q '"level":"error"'; then
        error_msg=$(echo "$line" | grep -o '"msg":"[^"]*"' | cut -d'"' -f4)
        echo -e "${RED}❌ [${timestamp}] ERROR:${NC} ${BOLD}${RED}${error_msg}${NC}"
    fi

    # Show raw interesting structured logs
    if echo "$line" | grep -q "category\|routed_to\|confidence" | grep -q "event"; then
        echo -e "${CYAN}📊 [${timestamp}] METRICS:${NC} ${line}"
    fi
done
