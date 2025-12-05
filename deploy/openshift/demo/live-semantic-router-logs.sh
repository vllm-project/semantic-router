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

# Color definitions - Enhanced for better visibility
RED='\033[0;31m'
BRIGHT_RED='\033[1;31m'
GREEN='\033[0;32m'
BRIGHT_GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BRIGHT_BLUE='\033[1;34m'
MAGENTA='\033[0;35m'
BRIGHT_MAGENTA='\033[1;35m'
CYAN='\033[0;36m'
BRIGHT_CYAN='\033[1;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
# Background colors for emphasis
BG_RED='\033[41m'
BG_GREEN='\033[42m'
BG_YELLOW='\033[43m'
BG_BLUE='\033[44m'
BG_MAGENTA='\033[45m'
BG_CYAN='\033[46m'
NC='\033[0m' # No Color

echo -e "${BOLD}${CYAN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║       SEMANTIC ROUTER - LIVE DEMO LOG VIEWER                               ║${NC}"
echo -e "${BOLD}${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}📡 Watching semantic-router logs in real-time...${NC}"
echo -e "${CYAN}Press Ctrl+C to stop${NC}"
echo ""
echo -e "${BOLD}Legend (Enhanced Colors for Live Demo):${NC}"
echo -e "  ${BRIGHT_CYAN}🔍 CLASSIFIED${NC}    - Category ${BOLD}${YELLOW}NAME${NC} in bright yellow → model"
echo -e "  ${BRIGHT_BLUE}🎯 ROUTING${NC}       - ${BG_BLUE}${WHITE}Model-A${NC} or ${BG_MAGENTA}${WHITE}Model-B${NC} selection"
echo -e "  ${BRIGHT_GREEN}🛡️  SECURITY${NC}     - ${BOLD}${WHITE}BENIGN${NC} or ${BG_RED}${WHITE}THREAT${NC} detection"
echo -e "  ${BG_CYAN}${WHITE}💾 CACHE HIT${NC}     - Cache hits for faster responses"
echo -e "  ${BRIGHT_MAGENTA}🧠 REASONING${NC}     - Chain-of-thought mode enabled"
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

    # Highlight JAILBREAK DETECTION - Enhanced with bright colors
    if echo "$line" | grep -q "BENIGN.*benign.*confidence"; then
        confidence=$(echo "$line" | grep -o 'confidence: [0-9.]*' | cut -d' ' -f2)
        echo -e "${BRIGHT_GREEN}🛡️  [${timestamp}] SECURITY:${NC} ${BOLD}${WHITE}BENIGN${NC} ${CYAN}(confidence: ${confidence})${NC}"
    elif echo "$line" | grep -q "Jailbreak classification result"; then
        # Parse the jailbreak result - {0 0.99999964} means class 0 (benign) with confidence
        result=$(echo "$line" | grep -o '{[0-9 .]*}' | tr -d '{}')
        class=$(echo "$result" | awk '{print $1}')
        conf=$(echo "$result" | awk '{print $2}')
        if [ "$class" = "0" ]; then
            echo -e "${BRIGHT_GREEN}🛡️  [${timestamp}] JAILBREAK CHECK:${NC} ${BOLD}${WHITE}BENIGN${NC} ${CYAN}(confidence: ${conf})${NC}"
        else
            echo -e "${BG_RED}${WHITE}🛡️  [${timestamp}] JAILBREAK CHECK: THREAT DETECTED${NC} ${YELLOW}(class: ${class}, conf: ${conf})${NC}"
        fi
    fi

    # Highlight PII DETECTION - Enhanced
    if echo "$line" | grep -qi "PII policy check passed\|No PII"; then
        echo -e "${BRIGHT_GREEN}🔒 [${timestamp}] PII:${NC} ${BOLD}${WHITE}No PII detected - Safe${NC}"
    elif echo "$line" | grep -qi "PII.*blocked\|PII.*rejected"; then
        echo -e "${BG_RED}${WHITE}🔒 [${timestamp}] PII: PII DETECTED & BLOCKED${NC}"
    fi
    # Skip generic PII messages that are just informational

    # Highlight MODEL ROUTING - Enhanced with brighter colors
    if echo "$msg" | grep -qi "Routing to model"; then
        routed_model=$(echo "$msg" | grep -o 'Model-[AB]')
        if [ -n "$routed_model" ]; then
            if [ "$routed_model" == "Model-A" ]; then
                echo -e "${BRIGHT_BLUE}🎯 [${timestamp}] ROUTING:${NC} ${BG_BLUE}${WHITE}${routed_model}${NC}"
            else
                echo -e "${BRIGHT_MAGENTA}🎯 [${timestamp}] ROUTING:${NC} ${BG_MAGENTA}${WHITE}${routed_model}${NC}"
            fi
        fi
    fi

    # Highlight CLASSIFIED - Enhanced to show category in unique color, separate from score
    if echo "$msg" | grep -qi "Selected model"; then
        # Extract category name (stop before "with score")
        category=$(echo "$msg" | grep -o 'category [a-z ]*with' | sed 's/category //' | sed 's/ with$//' | tr '[:lower:]' '[:upper:]')
        selected_model=$(echo "$msg" | grep -o 'Model-[AB]')
        score=$(echo "$msg" | grep -o 'score [0-9.]*' | sed 's/score //')
        if [ -n "$selected_model" ]; then
            # Category in bright yellow (no background), score in cyan, model with background
            if [ "$selected_model" == "Model-A" ]; then
                echo -e "${BRIGHT_CYAN}🔍 [${timestamp}] CLASSIFIED:${NC} ${BOLD}${YELLOW}${category}${NC} ${CYAN}WITH SCORE${NC} (score: ${BOLD}${score}${NC}) → ${BG_BLUE}${WHITE}${selected_model}${NC}"
            else
                echo -e "${BRIGHT_CYAN}🔍 [${timestamp}] CLASSIFIED:${NC} ${BOLD}${YELLOW}${category}${NC} ${CYAN}WITH SCORE${NC} (score: ${BOLD}${score}${NC}) → ${BG_MAGENTA}${WHITE}${selected_model}${NC}"
            fi
        fi
    fi

    # Highlight CACHE HITS - Enhanced
    if echo "$line" | grep -q "cache_hit"; then
        similarity=$(echo "$line" | grep -o '"similarity":[^,]*' | cut -d':' -f2)
        query=$(echo "$line" | grep -o '"query":"[^"]*"' | cut -d'"' -f4)
        if [ -n "$query" ]; then
            echo -e "${BRIGHT_CYAN}💾 [${timestamp}]${NC} ${BG_CYAN}${WHITE}CACHE HIT${NC} ${BOLD}${similarity}${NC} - ${YELLOW}${query}${NC}"
        fi
    fi

    # Highlight REASONING MODE - Enhanced (distinguish enabled vs disabled)
    if echo "$line" | grep -qi "Applied reasoning mode.*enabled: true\|reasoning mode.*enabled"; then
        echo -e "${BRIGHT_MAGENTA}🧠 [${timestamp}] REASONING:${NC} ${BOLD}${WHITE}Chain-of-thought enabled${NC}"
    elif echo "$line" | grep -qi "Reasoning mode disabled"; then
        echo -e "${CYAN}🧠 [${timestamp}] REASONING:${NC} Chain-of-thought disabled"
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
