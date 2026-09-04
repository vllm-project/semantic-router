#!/bin/bash
# Check translation sync status between English source docs and translations.
# Usage: ./scripts/check-translation-sync.sh [--locale LOCALE] [--coverage-only] [--fix-status] [--update-baseline] [--help]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBSITE_DIR="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$WEBSITE_DIR/docs"
I18N_BASE="$WEBSITE_DIR/i18n"
BASELINE_FILE="$I18N_BASE/translation-coverage-baseline.txt"

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

LOCALE=""
FIX_STATUS=false
UPDATE_BASELINE=false
COVERAGE_ONLY=false

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
        --update-baseline)
            UPDATE_BASELINE=true
            shift
            ;;
        --coverage-only)
            COVERAGE_ONLY=true
            shift
            ;;
        -h|--help)
            cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Check translation sync status between English source docs and translations.

Latest Git commit times are the primary drift signal. translation.source_commit
is a secondary metadata signal: when it is behind but the translated file is not
older, the file is reported for verification instead of definitely outdated.

Missing locale files use Docusaurus's current-English fallback and are reported
as coverage information. Exit status is 0 when every present translation is
synced, 1 when a present translation or its metadata needs work, and 2 for usage
or setup errors.

Options:
  -l, --locale LOCALE   Check specific locale only (default: all)
      --fix-status       Update unambiguous translation.outdated flags
      --update-baseline  Record the current locale override paths and exit
      --coverage-only    Check current-doc coverage without source drift metadata
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

if $FIX_STATUS && $UPDATE_BASELINE; then
    echo "--fix-status and --update-baseline cannot be used together" >&2
    exit 2
fi
if $UPDATE_BASELINE && [[ -n "$LOCALE" ]]; then
    echo "--locale and --update-baseline cannot be used together; the baseline covers every locale" >&2
    exit 2
fi
if $COVERAGE_ONLY && { $FIX_STATUS || $UPDATE_BASELINE; }; then
    echo "--coverage-only cannot be combined with a modifying option" >&2
    exit 2
fi

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

    "$GIT_BIN" log --oneline "$source_commit"..HEAD -- "$source_path" 2>/dev/null \
        | wc -l \
        | tr -d '[:space:]'
}

source_commit_is_usable() {
    local source_commit="$1"

    "$GIT_BIN" cat-file -e "$source_commit^{commit}" >/dev/null 2>&1 \
        && "$GIT_BIN" merge-base --is-ancestor "$source_commit" HEAD >/dev/null 2>&1
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

update_coverage_baseline() {
    local tmp_file
    tmp_file="$(mktemp "${BASELINE_FILE}.tmp.XXXXXX")"

    {
        echo "# Current documentation locale overrides."
        echo "# Regenerate with: make docs-update-translation-baseline"
        for locale in "${LOCALES[@]}"; do
            local locale_dir="$I18N_BASE/$locale/docusaurus-plugin-content-docs/current"
            while IFS= read -r -d '' translated_file; do
                local rel_path="${translated_file#"$locale_dir"/}"
                printf '%s\t%s\n' "$locale" "$rel_path"
            done < <(find "$locale_dir" -type f \( -name "*.md" -o -name "*.mdx" \) -print0)
        done
    } > "$tmp_file"

    {
        sed -n '1,2p' "$tmp_file"
        sed -n '3,$p' "$tmp_file" | LC_ALL=C sort
    } > "${tmp_file}.sorted"
    mv "${tmp_file}.sorted" "$BASELINE_FILE"
    rm -f "$tmp_file"
    echo "Updated $BASELINE_FILE"
}

if $UPDATE_BASELINE; then
    update_coverage_baseline
    exit 0
fi

if [[ ! -f "$BASELINE_FILE" ]]; then
    echo "Translation coverage baseline not found: $BASELINE_FILE" >&2
    echo "Run make docs-update-translation-baseline and review the generated paths." >&2
    exit 2
fi

cd "$WEBSITE_DIR"

total_likely_synced=0
total_outdated=0
total_fallback=0
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
    local fallback_count=0
    local likely_synced_count=0
    local metadata_count=0
    local verification_count=0
    local status_mismatch_count=0
    local status_fixed_count=0
    local coverage_regression_count=0
    local stale_baseline_count=0
    local unrecorded_coverage_count=0
    local orphaned_translation_count=0

    declare -a outdated_files
    declare -a metadata_files
    declare -a verification_files
    declare -a status_mismatches
    declare -a status_fixed
    declare -a fallback_sections
    declare -a coverage_regressions
    declare -a stale_baseline_entries
    declare -a unrecorded_coverage
    declare -a orphaned_translations

    local baseline_locale
    local baseline_path
    while IFS=$'\t' read -r baseline_locale baseline_path; do
        [[ -n "$baseline_locale" && "$baseline_locale" != \#* ]] || continue
        [[ "$baseline_locale" == "$locale" ]] || continue

        if [[ ! -f "$DOCS_DIR/$baseline_path" ]]; then
            stale_baseline_entries+=("$baseline_path")
            stale_baseline_count=$((stale_baseline_count + 1))
        elif [[ ! -f "$WEBSITE_DIR/$i18n_dir/$baseline_path" ]]; then
            coverage_regressions+=("$baseline_path")
            coverage_regression_count=$((coverage_regression_count + 1))
        fi
    done < "$BASELINE_FILE"

    while IFS= read -r -d '' translated_file; do
        local translated_rel_path="${translated_file#"$WEBSITE_DIR/$i18n_dir"/}"
        if [[ ! -f "$DOCS_DIR/$translated_rel_path" ]]; then
            orphaned_translations+=("$translated_rel_path")
            orphaned_translation_count=$((orphaned_translation_count + 1))
        elif ! grep -Fqx "$locale"$'\t'"$translated_rel_path" "$BASELINE_FILE"; then
            unrecorded_coverage+=("$translated_rel_path")
            unrecorded_coverage_count=$((unrecorded_coverage_count + 1))
        fi
    done < <(find "$WEBSITE_DIR/$i18n_dir" -type f \( -name "*.md" -o -name "*.mdx" \) -print0)

    while IFS= read -r -d '' source_file; do
        local rel_path="${source_file#"$DOCS_DIR"/}"

        [[ "$rel_path" == "OWNER" ]] && continue

        local source_path="docs/$rel_path"
        local expected_source_file="docs/$rel_path"
        local i18n_rel_path="$i18n_dir/$rel_path"
        local i18n_file="$WEBSITE_DIR/$i18n_rel_path"

        if [[ ! -f "$i18n_file" ]]; then
            fallback_count=$((fallback_count + 1))
            local section="${rel_path%%/*}"
            [[ "$section" != "$rel_path" ]] || section="(root)"
            fallback_sections+=("$section")
            continue
        fi

        if $COVERAGE_ONLY; then
            likely_synced_count=$((likely_synced_count + 1))
            continue
        fi

        local source_timestamp
        local source_commit
        local source_date
        local i18n_timestamp
        local i18n_commit
        local i18n_date
        source_timestamp="$("$GIT_BIN" log -1 --format="%ct" -- "$source_path" 2>/dev/null || echo "0")"
        source_commit="$("$GIT_BIN" log -1 --format="%h" -- "$source_path" 2>/dev/null || echo "?")"
        source_date="$("$GIT_BIN" log -1 --format="%cs" -- "$source_path" 2>/dev/null || echo "?")"
        i18n_timestamp="$("$GIT_BIN" log -1 --format="%ct" -- "$i18n_rel_path" 2>/dev/null || echo "0")"
        i18n_commit="$("$GIT_BIN" log -1 --format="%h" -- "$i18n_rel_path" 2>/dev/null || echo "?")"
        i18n_date="$("$GIT_BIN" log -1 --format="%cs" -- "$i18n_rel_path" 2>/dev/null || echo "?")"
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

    done < <(find "$DOCS_DIR" -type f \( -name "*.md" -o -name "*.mdx" \) -print0)

    echo -e "${CYAN}[$locale]${NC}"

    if [[ $fallback_count -gt 0 ]]; then
        echo -e "  ${CYAN}English fallback:${NC} $fallback_count page(s) have no locale override"
        printf '%s\n' "${fallback_sections[@]}" \
            | LC_ALL=C sort \
            | uniq -c \
            | while read -r count section; do
                echo "    $section: $count"
            done
    fi

    if [[ ${#coverage_regressions[@]} -gt 0 ]]; then
        echo -e "  ${RED}Coverage regressions (baseline override deleted):${NC}"
        printf '    %s\n' "${coverage_regressions[@]}"
    fi

    if [[ ${#stale_baseline_entries[@]} -gt 0 ]]; then
        echo -e "  ${RED}Stale baseline entries (English source retired or moved):${NC}"
        printf '    %s\n' "${stale_baseline_entries[@]}"
    fi

    if [[ ${#unrecorded_coverage[@]} -gt 0 ]]; then
        echo -e "  ${YELLOW}Unrecorded coverage additions:${NC}"
        printf '    %s\n' "${unrecorded_coverage[@]}"
        echo "    Run make docs-update-translation-baseline and review the baseline diff."
    fi

    if [[ ${#orphaned_translations[@]} -gt 0 ]]; then
        echo -e "  ${RED}Locale overrides without a current English source:${NC}"
        printf '    %s\n' "${orphaned_translations[@]}"
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

    local total=$((likely_synced_count + outdated_count + fallback_count))
    local sync_rate=0
    [[ $total -gt 0 ]] && sync_rate=$((likely_synced_count * 100 / total))

    echo -e "  ${GREEN}✓${NC} $likely_synced_count translated and likely synced  ${CYAN}↪${NC} $fallback_count English fallback  ${RED}−${NC} $coverage_regression_count coverage regressions  ${YELLOW}↓${NC} $outdated_count  ${MAGENTA}!${NC} $metadata_count  ${CYAN}?${NC} $verification_count  ${YELLOW}~${NC} $status_mismatch_count  ${GREEN}+${NC} $status_fixed_count  (${sync_rate}% translated coverage)"
    echo ""

    total_likely_synced=$((total_likely_synced + likely_synced_count))
    total_outdated=$((total_outdated + outdated_count))
    total_fallback=$((total_fallback + fallback_count))
    total_metadata=$((total_metadata + metadata_count))
    total_verification=$((total_verification + verification_count))
    total_status_mismatch=$((total_status_mismatch + status_mismatch_count))
    total_status_fixed=$((total_status_fixed + status_fixed_count))

    [[ $outdated_count -gt 0 ]] \
        || [[ $metadata_count -gt 0 ]] \
        || [[ $verification_count -gt 0 ]] \
        || [[ $status_mismatch_count -gt 0 ]] \
        || [[ $coverage_regression_count -gt 0 ]] \
        || [[ $stale_baseline_count -gt 0 ]] \
        || [[ $unrecorded_coverage_count -gt 0 ]] \
        || [[ $orphaned_translation_count -gt 0 ]]
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
    total=$((total_likely_synced + total_outdated + total_fallback))
    sync_rate=0
    [[ $total -gt 0 ]] && sync_rate=$((total_likely_synced * 100 / total))
    echo -e "${GREEN}✓ Translated and likely synced: $total_likely_synced${NC}  ${CYAN}↪ English fallback: $total_fallback${NC}  ${YELLOW}↓ Outdated: $total_outdated${NC}  ${MAGENTA}! Metadata: $total_metadata${NC}  ${CYAN}? Verify: $total_verification${NC}  ${YELLOW}~ Status: $total_status_mismatch${NC}  ${GREEN}+ Fixed: $total_status_fixed${NC}"
    echo -e "Translated coverage: ${sync_rate}% ($total_likely_synced / $total)"
fi

if $locale_has_issues; then
    exit 1
else
    exit 0
fi
