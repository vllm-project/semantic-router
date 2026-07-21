#!/bin/bash
# Check translation sync status between English source docs and translations.
# Usage: ./scripts/check-translation-sync.sh [--locale LOCALE] [--fix-status] [--help]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBSITE_DIR="$(dirname "$SCRIPT_DIR")"

find_git_bin() {
    local candidate
    local candidates=()

    [[ -x "/usr/bin/git" ]] && candidates+=("/usr/bin/git")
    if candidate="$(command -v git 2>/dev/null)"; then
        candidates+=("$candidate")
    fi

    for candidate in "${candidates[@]}"; do
        [[ -n "$candidate" ]] || continue
        if "$candidate" --version >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    echo "No usable git executable found" >&2
    return 1
}

GIT_BIN="$(find_git_bin)" || exit 2
REPO_ROOT="$("$GIT_BIN" -C "$WEBSITE_DIR" rev-parse --show-toplevel 2>/dev/null || { cd "$WEBSITE_DIR/.." && pwd; })"
DOCS_DIR="$REPO_ROOT/website/docs"
I18N_BASE="$REPO_ROOT/website/i18n"

LOCALE=""
FIX_STATUS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -l|--locale)
            LOCALE="$2"
            shift 2
            ;;
        --fix-status)
            FIX_STATUS=true
            shift
            ;;
        -h|--help)
            cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Check translation sync status between English source docs and translations.

The check uses each translated file's translation.source_commit metadata, not
file modification time, so translations remain stale until their source_commit is
advanced to the English source revision they were reviewed against.

Exit status is 0 when translations are fully synced, 1 when drift or metadata
issues are found, and 2 for usage or setup errors.

Options:
  -l, --locale LOCALE   Check specific locale only (default: all)
    --fix-status          Update translation.outdated to match computed drift
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
MAGENTA='\033[0;35m'
NC='\033[0m'

frontmatter_value() {
    local key="$1"
    local file="$2"

    grep -m1 -E "^[[:space:]]*$key:" "$file" 2>/dev/null \
        | sed -E "s/^[[:space:]]*$key:[[:space:]]*//; s/[[:space:]]+#.*$//; s/^['\"]//; s/['\"]$//"
}

source_update_count() {
    local source_commit="$1"
    local source_repo_path="$2"

    "$GIT_BIN" log --oneline "$source_commit"..HEAD -- "$source_repo_path" 2>/dev/null \
        | wc -l \
        | tr -d '[:space:]'
}

source_commit_is_usable() {
    local source_commit="$1"
    local source_repo_path="$2"

    "$GIT_BIN" log -1 --format="%h" "$source_commit" -- "$source_repo_path" >/dev/null 2>&1
}

set_outdated_status() {
    local file="$1"
    local expected="$2"
    local tmp_file

    tmp_file="$(mktemp)"
    awk -v expected="$expected" '
        /^[[:space:]]*outdated:[[:space:]]*(true|false)[[:space:]]*$/ && !done {
            sub(/outdated:[[:space:]]*(true|false)/, "outdated: " expected)
            done = 1
        }
        { print }
    ' "$file" > "$tmp_file"
    mv "$tmp_file" "$file"
}

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

cd "$REPO_ROOT"

total_synced=0
total_outdated=0
total_missing=0
total_metadata=0
total_status_mismatch=0
total_status_fixed=0

check_locale() {
    local locale="$1"
    local i18n_dir="website/i18n/$locale/docusaurus-plugin-content-docs/current"

    if [[ ! -d "$REPO_ROOT/$i18n_dir" ]]; then
        echo -e "${RED}Error: Translation directory not found: $i18n_dir${NC}" >&2
        return 1
    fi

    local outdated_count=0
    local missing_count=0
    local synced_count=0
    local metadata_count=0
    local status_mismatch_count=0
    local status_fixed_count=0

    declare -a outdated_files
    declare -a missing_files
    declare -a metadata_files
    declare -a status_mismatches
    declare -a status_fixed

    while IFS= read -r -d '' source_file; do
        local rel_path="${source_file#"$DOCS_DIR"/}"

        [[ "$rel_path" == "OWNER" ]] && continue

        local source_repo_path="website/docs/$rel_path"
        local expected_source_file="docs/$rel_path"
        local i18n_rel_path="$i18n_dir/$rel_path"
        local i18n_file="$REPO_ROOT/$i18n_rel_path"

        if [[ ! -f "$i18n_file" ]]; then
            missing_files+=("$rel_path")
            ((missing_count++))
            continue
        fi

        local source_commit
        local source_file_meta
        local outdated_meta
        source_commit="$(frontmatter_value "source_commit" "$i18n_file")"
        source_file_meta="$(frontmatter_value "source_file" "$i18n_file")"
        outdated_meta="$(frontmatter_value "outdated" "$i18n_file")"

        if [[ -z "$source_commit" ]]; then
            metadata_files+=("$rel_path|missing translation.source_commit")
            ((metadata_count++))
            continue
        fi

        if [[ -z "$source_file_meta" ]]; then
            metadata_files+=("$rel_path|missing translation.source_file")
            ((metadata_count++))
            continue
        fi

        if [[ -z "$outdated_meta" ]]; then
            metadata_files+=("$rel_path|missing translation.outdated")
            ((metadata_count++))
            continue
        fi

        if [[ "$source_file_meta" != "$expected_source_file" ]]; then
            metadata_files+=("$rel_path|source_file is $source_file_meta, expected $expected_source_file")
            ((metadata_count++))
            continue
        fi

        if ! source_commit_is_usable "$source_commit" "$source_repo_path"; then
            metadata_files+=("$rel_path|invalid source_commit $source_commit")
            ((metadata_count++))
            continue
        fi

        local latest_source_commit
        local updates
        latest_source_commit="$("$GIT_BIN" log -1 --format="%h" -- "$source_repo_path" 2>/dev/null || echo "?")"
        updates="$(source_update_count "$source_commit" "$source_repo_path")"
        [[ -n "$updates" ]] || updates=0

        if [[ "$updates" -gt 0 ]]; then
            outdated_files+=("$rel_path|$source_commit|$latest_source_commit|$updates")
            ((outdated_count++))

            if [[ "$outdated_meta" != "true" ]]; then
                if $FIX_STATUS; then
                    set_outdated_status "$i18n_file" "true"
                    status_fixed+=("$rel_path|outdated set to true")
                    ((status_fixed_count++))
                else
                    status_mismatches+=("$rel_path|outdated is $outdated_meta, expected true")
                    ((status_mismatch_count++))
                fi
            fi
        else
            ((synced_count++))

            if [[ "$outdated_meta" == "true" ]]; then
                if $FIX_STATUS; then
                    set_outdated_status "$i18n_file" "false"
                    status_fixed+=("$rel_path|outdated set to false")
                    ((status_fixed_count++))
                else
                    status_mismatches+=("$rel_path|outdated is true, expected false")
                    ((status_mismatch_count++))
                fi
            fi
        fi

    done < <(find "$DOCS_DIR" -name "*.md" -type f -print0)

    echo -e "${CYAN}[$locale]${NC}"

    if [[ ${#missing_files[@]} -gt 0 ]]; then
        echo -e "  ${RED}Missing translations:${NC}"
        for file in "${missing_files[@]}"; do
            echo -e "    ${RED}✗${NC} $file"
        done
    fi

    if [[ ${#metadata_files[@]} -gt 0 ]]; then
        echo -e "  ${MAGENTA}Metadata issues:${NC}"
        for entry in "${metadata_files[@]}"; do
            IFS='|' read -r file reason <<< "$entry"
            echo -e "    ${MAGENTA}!${NC} $file"
            echo -e "      $reason"
        done
    fi

    if [[ ${#outdated_files[@]} -gt 0 ]]; then
        echo -e "  ${YELLOW}Outdated translations:${NC}"
        for entry in "${outdated_files[@]}"; do
            IFS='|' read -r file source_commit latest_source_commit updates <<< "$entry"
            echo -e "    ${YELLOW}↓${NC} $file"
            echo -e "      source_commit $source_commit -> $latest_source_commit ($updates source commits)"
        done
    fi

    if [[ ${#status_mismatches[@]} -gt 0 ]]; then
        echo -e "  ${YELLOW}Status metadata mismatches:${NC}"
        for entry in "${status_mismatches[@]}"; do
            IFS='|' read -r file reason <<< "$entry"
            echo -e "    ${YELLOW}~${NC} $file"
            echo -e "      $reason"
        done
    fi

    if [[ ${#status_fixed[@]} -gt 0 ]]; then
        echo -e "  ${GREEN}Status metadata updated:${NC}"
        for entry in "${status_fixed[@]}"; do
            IFS='|' read -r file reason <<< "$entry"
            echo -e "    ${GREEN}+${NC} $file"
            echo -e "      $reason"
        done
    fi

    local total=$((synced_count + outdated_count + missing_count + metadata_count))
    local sync_rate=0
    [[ $total -gt 0 ]] && sync_rate=$((synced_count * 100 / total))

    echo -e "  ${GREEN}✓${NC} $synced_count  ${YELLOW}↓${NC} $outdated_count  ${RED}✗${NC} $missing_count  ${MAGENTA}!${NC} $metadata_count  ${YELLOW}~${NC} $status_mismatch_count  ${GREEN}+${NC} $status_fixed_count  (${sync_rate}%)"
    echo ""

    ((total_synced += synced_count))
    ((total_outdated += outdated_count))
    ((total_missing += missing_count))
    ((total_metadata += metadata_count))
    ((total_status_mismatch += status_mismatch_count))
    ((total_status_fixed += status_fixed_count))

    [[ $outdated_count -gt 0 ]] || [[ $missing_count -gt 0 ]] || [[ $metadata_count -gt 0 ]] || [[ $status_mismatch_count -gt 0 ]]
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
    total=$((total_synced + total_outdated + total_missing + total_metadata))
    sync_rate=0
    [[ $total -gt 0 ]] && sync_rate=$((total_synced * 100 / total))
    echo -e "${GREEN}✓ Synced: $total_synced${NC}  ${YELLOW}↓ Outdated: $total_outdated${NC}  ${RED}✗ Missing: $total_missing${NC}  ${MAGENTA}! Metadata: $total_metadata${NC}  ${YELLOW}~ Status: $total_status_mismatch${NC}  ${GREEN}+ Fixed: $total_status_fixed${NC}"
    echo -e "Sync rate: ${sync_rate}% ($total_synced / $total)"
fi

if $locale_has_issues; then
    exit 1
else
    exit 0
fi
