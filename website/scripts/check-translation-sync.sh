#!/bin/bash
# Check translation sync status between English source docs and translations.
# Usage: ./scripts/check-translation-sync.sh [--locale LOCALE] [--fix-status] [--help]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBSITE_DIR="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$WEBSITE_DIR/docs"
I18N_BASE="$WEBSITE_DIR/i18n"

LOCALE=""
FIX_STATUS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -l|--locale)
            if [[ -z "${2:-}" ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
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

Latest Git commit times are the primary drift signal. translation.source_commit
is a secondary metadata signal: when it is behind but the translated file is not
older, the file is reported for verification instead of definitely outdated.

Exit status is 0 when translations are fully synced, 1 when translation or
metadata work remains, and 2 for usage or setup errors.

Options:
  -l, --locale LOCALE   Check specific locale only (default: all)
    --fix-status          Update unambiguous translation.outdated flags
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

    awk -v key="$key" '
        NR == 1 {
            if ($0 != "---") exit
            in_frontmatter = 1
            next
        }
        in_frontmatter && $0 == "---" { exit }
        in_frontmatter {
            pattern = "^[[:space:]]*" key ":[[:space:]]*"
            if ($0 ~ pattern) {
                value = $0
                sub(pattern, "", value)
                print value
                exit
            }
        }
    ' "$file" | sed -E "s/[[:space:]]+#.*$//; s/^['\"]//; s/['\"]$//"
}

source_update_count() {
    local source_commit="$1"
    local source_path="$2"

    git log --oneline "$source_commit"..HEAD -- "$source_path" 2>/dev/null \
        | wc -l \
        | tr -d '[:space:]'
}

source_commit_is_usable() {
    local source_commit="$1"

    git cat-file -e "$source_commit^{commit}" >/dev/null 2>&1 \
        && git merge-base --is-ancestor "$source_commit" HEAD >/dev/null 2>&1
}

set_outdated_status() {
    local file="$1"
    local expected="$2"
    local tmp_file

    tmp_file="$(mktemp "${file}.tmp.XXXXXX")"
    awk -v expected="$expected" '
        NR == 1 && $0 == "---" {
            in_frontmatter = 1
            print
            next
        }
        in_frontmatter && $0 == "---" {
            in_frontmatter = 0
            print
            next
        }
        in_frontmatter && /^[[:space:]]*outdated:[[:space:]]*(true|false)[[:space:]]*$/ && !updated {
            sub(/outdated:[[:space:]]*(true|false)/, "outdated: " expected)
            updated = 1
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

cd "$WEBSITE_DIR"

total_likely_synced=0
total_outdated=0
total_missing=0
total_metadata=0
total_verification=0
total_status_mismatch=0
total_status_fixed=0

check_locale() {
    local locale="$1"
    local i18n_dir="i18n/$locale/docusaurus-plugin-content-docs/current"

    if [[ ! -d "$WEBSITE_DIR/$i18n_dir" ]]; then
        echo -e "${RED}Error: Translation directory not found: $i18n_dir${NC}" >&2
        return 1
    fi

    local outdated_count=0
    local missing_count=0
    local likely_synced_count=0
    local metadata_count=0
    local verification_count=0
    local status_mismatch_count=0
    local status_fixed_count=0

    declare -a outdated_files
    declare -a missing_files
    declare -a metadata_files
    declare -a verification_files
    declare -a status_mismatches
    declare -a status_fixed

    while IFS= read -r -d '' source_file; do
        local rel_path="${source_file#"$DOCS_DIR"/}"

        [[ "$rel_path" == "OWNER" ]] && continue

        local source_path="docs/$rel_path"
        local expected_source_file="docs/$rel_path"
        local i18n_rel_path="$i18n_dir/$rel_path"
        local i18n_file="$WEBSITE_DIR/$i18n_rel_path"

        if [[ ! -f "$i18n_file" ]]; then
            missing_files+=("$rel_path")
            missing_count=$((missing_count + 1))
            continue
        fi

        local source_timestamp
        local source_commit
        local source_date
        local i18n_timestamp
        local i18n_commit
        local i18n_date
        source_timestamp="$(git log -1 --format="%ct" -- "$source_path" 2>/dev/null || echo "0")"
        source_commit="$(git log -1 --format="%h" -- "$source_path" 2>/dev/null || echo "?")"
        source_date="$(git log -1 --format="%cs" -- "$source_path" 2>/dev/null || echo "?")"
        i18n_timestamp="$(git log -1 --format="%ct" -- "$i18n_rel_path" 2>/dev/null || echo "0")"
        i18n_commit="$(git log -1 --format="%h" -- "$i18n_rel_path" 2>/dev/null || echo "?")"
        i18n_date="$(git log -1 --format="%cs" -- "$i18n_rel_path" 2>/dev/null || echo "?")"
        [[ -n "$source_timestamp" ]] || source_timestamp=0
        [[ -n "$i18n_timestamp" ]] || i18n_timestamp=0

        local source_commit_meta
        local source_file_meta
        local outdated_meta
        local metadata_reason=""
        local metadata_state="unknown"
        local updates=0
        source_commit_meta="$(frontmatter_value "source_commit" "$i18n_file")"
        source_file_meta="$(frontmatter_value "source_file" "$i18n_file")"
        outdated_meta="$(frontmatter_value "outdated" "$i18n_file")"

        if [[ -z "$source_commit_meta" ]]; then
            metadata_reason="missing translation.source_commit"
        elif ! source_commit_is_usable "$source_commit_meta"; then
            metadata_reason="invalid translation.source_commit $source_commit_meta"
        fi

        if [[ -z "$source_file_meta" ]]; then
            metadata_reason="${metadata_reason:+$metadata_reason; }missing translation.source_file"
        elif [[ "$source_file_meta" != "$expected_source_file" ]]; then
            metadata_reason="${metadata_reason:+$metadata_reason; }source_file is $source_file_meta, expected $expected_source_file"
        fi

        if [[ "$outdated_meta" != "true" && "$outdated_meta" != "false" ]]; then
            metadata_reason="${metadata_reason:+$metadata_reason; }missing or invalid translation.outdated"
        fi

        if [[ -n "$metadata_reason" ]]; then
            metadata_files+=("$rel_path|$metadata_reason")
            metadata_count=$((metadata_count + 1))
        else
            updates="$(source_update_count "$source_commit_meta" "$source_path")"
            [[ -n "$updates" ]] || updates=0
            if [[ "$updates" -gt 0 ]]; then
                metadata_state="behind"
            else
                metadata_state="current"
            fi
        fi

        local expected_outdated=""
        if [[ "$i18n_timestamp" == "0" ]]; then
            if [[ -z "$metadata_reason" ]]; then
                metadata_files+=("$rel_path|translation file has no Git history")
                metadata_count=$((metadata_count + 1))
            fi
            likely_synced_count=$((likely_synced_count + 1))
        elif [[ "$source_timestamp" -gt "$i18n_timestamp" ]]; then
            outdated_files+=("$rel_path|$i18n_commit|$i18n_date|$source_commit|$source_date")
            outdated_count=$((outdated_count + 1))
            expected_outdated="true"
        else
            likely_synced_count=$((likely_synced_count + 1))
            if [[ "$metadata_state" == "behind" ]]; then
                verification_files+=("$rel_path|$source_commit_meta|$source_commit|$i18n_commit|$updates")
                verification_count=$((verification_count + 1))
            elif [[ "$metadata_state" == "current" ]]; then
                expected_outdated="false"
            fi
        fi

        if [[ -n "$expected_outdated" && ( "$outdated_meta" == "true" || "$outdated_meta" == "false" ) && "$outdated_meta" != "$expected_outdated" ]]; then
            if $FIX_STATUS; then
                set_outdated_status "$i18n_file" "$expected_outdated"
                status_fixed+=("$rel_path|outdated set to $expected_outdated")
                status_fixed_count=$((status_fixed_count + 1))
            else
                status_mismatches+=("$rel_path|outdated is $outdated_meta, expected $expected_outdated")
                status_mismatch_count=$((status_mismatch_count + 1))
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

    if [[ ${#outdated_files[@]} -gt 0 ]]; then
        echo -e "  ${YELLOW}Outdated translations (English commit is newer):${NC}"
        for entry in "${outdated_files[@]}"; do
            IFS='|' read -r file i18n_commit i18n_date source_commit source_date <<< "$entry"
            echo -e "    ${YELLOW}↓${NC} $file"
            echo -e "      $i18n_commit ($i18n_date) -> $source_commit ($source_date)"
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

    if [[ ${#verification_files[@]} -gt 0 ]]; then
        echo -e "  ${CYAN}Metadata needs verification (Chinese is not older):${NC}"
        for entry in "${verification_files[@]}"; do
            IFS='|' read -r file recorded_commit source_commit i18n_commit updates <<< "$entry"
            echo -e "    ${CYAN}?${NC} $file"
            echo -e "      source_commit $recorded_commit -> $source_commit ($updates source commits); Chinese latest $i18n_commit"
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

    local total=$((likely_synced_count + outdated_count + missing_count))
    local sync_rate=0
    [[ $total -gt 0 ]] && sync_rate=$((likely_synced_count * 100 / total))

    echo -e "  ${GREEN}✓${NC} $likely_synced_count likely synced  ${YELLOW}↓${NC} $outdated_count  ${RED}✗${NC} $missing_count  ${MAGENTA}!${NC} $metadata_count  ${CYAN}?${NC} $verification_count  ${YELLOW}~${NC} $status_mismatch_count  ${GREEN}+${NC} $status_fixed_count  (${sync_rate}%)"
    echo ""

    total_likely_synced=$((total_likely_synced + likely_synced_count))
    total_outdated=$((total_outdated + outdated_count))
    total_missing=$((total_missing + missing_count))
    total_metadata=$((total_metadata + metadata_count))
    total_verification=$((total_verification + verification_count))
    total_status_mismatch=$((total_status_mismatch + status_mismatch_count))
    total_status_fixed=$((total_status_fixed + status_fixed_count))

    [[ $outdated_count -gt 0 ]] || [[ $missing_count -gt 0 ]] || [[ $metadata_count -gt 0 ]] || [[ $verification_count -gt 0 ]] || [[ $status_mismatch_count -gt 0 ]]
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
    total=$((total_likely_synced + total_outdated + total_missing))
    sync_rate=0
    [[ $total -gt 0 ]] && sync_rate=$((total_likely_synced * 100 / total))
    echo -e "${GREEN}✓ Likely synced: $total_likely_synced${NC}  ${YELLOW}↓ Outdated: $total_outdated${NC}  ${RED}✗ Missing: $total_missing${NC}  ${MAGENTA}! Metadata: $total_metadata${NC}  ${CYAN}? Verify: $total_verification${NC}  ${YELLOW}~ Status: $total_status_mismatch${NC}  ${GREEN}+ Fixed: $total_status_fixed${NC}"
    echo -e "Likely sync rate: ${sync_rate}% ($total_likely_synced / $total)"
fi

if $locale_has_issues; then
    exit 1
else
    exit 0
fi
