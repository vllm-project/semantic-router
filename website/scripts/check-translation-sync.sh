#!/bin/bash
# Check translation sync status between English source docs and translations
# Usage: ./scripts/check-translation-sync.sh [--locale LOCALE] [--help]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBSITE_DIR="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$WEBSITE_DIR/docs"
I18N_BASE="$WEBSITE_DIR/i18n"

LOCALE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -l|--locale)
            LOCALE="$2"
            shift 2
            ;;
        -h|--help)
            cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Check translation sync status between English source docs and translations.

Options:
  -l, --locale LOCALE   Check specific locale only (default: all)
  -h, --help            Show this help message
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Discover available locales
if [[ -n "$LOCALE" ]]; then
    LOCALES=("$LOCALE")
else
    LOCALES=()
    for dir in "$I18N_BASE"/*/docusaurus-plugin-content-docs/current; do
        if [[ -d "$dir" ]]; then
            # Extract locale from path: i18n/LOCALE/docusaurus-plugin-content-docs/current
            locale="${dir#"$I18N_BASE"/}"
            locale="${locale%%/*}"
            LOCALES+=("$locale")
        fi
    done
fi

if [[ ${#LOCALES[@]} -eq 0 ]]; then
    echo "No translation locales found in $I18N_BASE" >&2
    exit 2
fi

cd "$WEBSITE_DIR"

total_synced=0
total_outdated=0
total_missing=0

check_locale() {
    local locale="$1"
    local i18n_dir="i18n/$locale/docusaurus-plugin-content-docs/current"
    
    if [[ ! -d "$WEBSITE_DIR/$i18n_dir" ]]; then
        echo -e "${RED}Error: Translation directory not found: $i18n_dir${NC}" >&2
        return 1
    fi
    
    local outdated_count=0
    local missing_count=0
    local synced_count=0
    
    declare -a outdated_files
    declare -a missing_files
    
    while IFS= read -r -d '' source_file; do
        local rel_path="${source_file#"$DOCS_DIR"/}"
        
        [[ "$rel_path" == "OWNER" ]] && continue
        
        local i18n_rel_path="$i18n_dir/$rel_path"
        local i18n_file="$WEBSITE_DIR/$i18n_rel_path"
        
        if [[ ! -f "$i18n_file" ]]; then
            missing_files+=("$rel_path")
            ((missing_count++))
            continue
        fi
        
        local source_timestamp=$(git log -1 --format="%ct" -- "docs/$rel_path" 2>/dev/null || echo "0")
        local source_commit=$(git log -1 --format="%h" -- "docs/$rel_path" 2>/dev/null || echo "?")
        local source_date=$(git log -1 --format="%ci" -- "docs/$rel_path" 2>/dev/null | cut -d' ' -f1 || echo "?")
        
        local i18n_timestamp=$(git log -1 --format="%ct" -- "$i18n_rel_path" 2>/dev/null || echo "0")
        local i18n_commit=$(git log -1 --format="%h" -- "$i18n_rel_path" 2>/dev/null || echo "?")
        local i18n_date=$(git log -1 --format="%ci" -- "$i18n_rel_path" 2>/dev/null | cut -d' ' -f1 || echo "?")
        
        if [[ "$i18n_timestamp" == "0" ]]; then
            ((synced_count++))
            continue
        fi
        
        if [[ "$source_timestamp" -gt "$i18n_timestamp" ]]; then
            outdated_files+=("$rel_path|$i18n_commit|$i18n_date|$source_commit|$source_date")
            ((outdated_count++))
        else
            ((synced_count++))
        fi
        
    done < <(find "$DOCS_DIR" -name "*.md" -print0)
    
    echo -e "${CYAN}[$locale]${NC}"
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        echo -e "  ${RED}Missing:${NC}"
        for file in "${missing_files[@]}"; do
            echo -e "    ${RED}✗${NC} $file"
        done
    fi
    
    if [[ ${#outdated_files[@]} -gt 0 ]]; then
        echo -e "  ${YELLOW}Outdated:${NC}"
        for entry in "${outdated_files[@]}"; do
            IFS='|' read -r file i18n_commit i18n_date source_commit source_date <<< "$entry"
            echo -e "    ${YELLOW}↓${NC} $file"
            echo -e "      $i18n_commit ($i18n_date) -> $source_commit ($source_date)"
        done
    fi
    
    local total=$((synced_count + outdated_count + missing_count))
    local sync_rate=0
    [[ $total -gt 0 ]] && sync_rate=$((synced_count * 100 / total))
    
    echo -e "  ${GREEN}✓${NC} $synced_count  ${YELLOW}↓${NC} $outdated_count  ${RED}✗${NC} $missing_count  (${sync_rate}%)"
    echo ""
    
    ((total_synced += synced_count))
    ((total_outdated += outdated_count))
    ((total_missing += missing_count))
    
    [[ $outdated_count -gt 0 ]] || [[ $missing_count -gt 0 ]]
}

echo -e "${BLUE}=== Translation Sync Check ===${NC}"
echo ""

locale_has_issues=false
for locale in "${LOCALES[@]}"; do
    if check_locale "$locale"; then
        locale_has_issues=true
    fi
done

if [[ ${#LOCALES[@]} -gt 1 ]]; then
    echo -e "${BLUE}=== Total ===${NC}"
    total=$((total_synced + total_outdated + total_missing))
    sync_rate=0
    [[ $total -gt 0 ]] && sync_rate=$((total_synced * 100 / total))
    echo -e "${GREEN}✓ Synced: $total_synced${NC}  ${YELLOW}↓ Outdated: $total_outdated${NC}  ${RED}✗ Missing: $total_missing${NC}"
    echo -e "Sync rate: ${sync_rate}% ($total_synced / $total)"
fi

if $locale_has_issues; then
    exit 1
else
    exit 0
fi
