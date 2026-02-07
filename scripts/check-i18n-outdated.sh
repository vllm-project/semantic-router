#!/bin/bash
# Detect outdated translations by comparing source_commit with current EN state
# Fallback to timestamp comparison when source_commit is missing
#
# Output types:
#   [OUTDATED]     - EN changed since source_commit, translation needs update
#   [SYNC_ONLY]    - EN unchanged, but source_commit needs update to latest
#   [NO_TRACKING]  - Missing source_commit, using timestamp fallback
#
# Usage:
#   ./scripts/check-i18n-outdated.sh              # Check all docs
#   ./scripts/check-i18n-outdated.sh overview     # Check only overview dir
#   SHOW_DIFF=0 ./scripts/check-i18n-outdated.sh  # List only (no diff)

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

DOCS_BASE="website/docs"
I18N_BASE="website/i18n/zh-Hans/docusaurus-plugin-content-docs/current"

SUBPATH="${1:-}"
DOCS_DIR="$DOCS_BASE${SUBPATH:+/$SUBPATH}"
I18N_DIR="$I18N_BASE${SUBPATH:+/$SUBPATH}"

if [[ ! -d "$DOCS_DIR" ]]; then
    echo "Error: Directory does not exist $DOCS_DIR" >&2
    exit 1
fi

SHOW_DIFF=${SHOW_DIFF:-1}
outdated=0
sync_only=0
no_tracking=0

# Extract source_commit from frontmatter (POSIX compatible)
get_source_commit() {
    local file="$1"
    # Extract frontmatter and find source_commit value
    sed -n '/^---$/,/^---$/p' "$file" 2>/dev/null | \
        sed -n 's/.*source_commit:[[:space:]]*["\x27]*\([a-f0-9]\{7,\}\).*/\1/p' | head -1
}

while read -r en_file; do
    rel_path="${en_file#"$DOCS_DIR"/}"
    zh_file="$I18N_DIR/$rel_path"
    
    [[ ! -f "$zh_file" ]] && continue
    
    en_latest=$(git log -1 --format="%H" -- "$en_file" 2>/dev/null || echo "")
    [[ -z "$en_latest" ]] && continue
    
    source_commit=$(get_source_commit "$zh_file")
    
    if [[ -n "$source_commit" ]]; then
        # Expand short commit to full if needed
        full_source=$(git rev-parse "$source_commit" 2>/dev/null || echo "")
        
        if [[ -z "$full_source" ]]; then
            # source_commit doesn't exist (rebased away?)
            echo "[INVALID]    $rel_path: source_commit=$source_commit not found"
            ((outdated++)) || true
            continue
        fi
        
        # Check if EN has changes since source_commit
        if ! git diff --quiet "$full_source" "$en_latest" -- "$en_file" 2>/dev/null; then
            ((outdated++)) || true
            en_date=$(git log -1 --format="%cs" -- "$en_file")
            src_date=$(git log -1 --format="%cs" "$full_source" 2>/dev/null || echo "unknown")
            
            echo ">>> [OUTDATED]   $rel_path"
            echo "    source_commit: ${source_commit} ($src_date)"
            echo "    EN latest:     ${en_latest:0:7} ($en_date)"
            echo "    Update source_commit to: ${en_latest:0:7}"
            
            if [[ "$SHOW_DIFF" == "1" ]]; then
                echo ""
                git diff --color=always --stat "$full_source" "$en_latest" -- "$en_file" | head -3
                git diff --color=always -U0 "$full_source" "$en_latest" -- "$en_file" | grep -E '^(\x1b\[[0-9;]*m)?[\+\-]' | head -20
                echo ""
            fi
        elif [[ "${full_source:0:7}" != "${en_latest:0:7}" ]]; then
            # EN unchanged but source_commit points to older commit (e.g. typo fix in EN)
            ((sync_only++)) || true
            echo "[SYNC_ONLY]  $rel_path: source_commit=${source_commit} -> ${en_latest:0:7} (content unchanged)"
        fi
    else
        # No source_commit, fallback to timestamp
        en_ts=$(git log -1 --format="%ct" -- "$en_file" 2>/dev/null || echo "")
        zh_ts=$(git log -1 --format="%ct" -- "$zh_file" 2>/dev/null || echo "")
        
        if [[ -n "$en_ts" && -n "$zh_ts" ]] && (( en_ts > zh_ts )); then
            ((no_tracking++)) || true
            en_date=$(git log -1 --format="%cs" -- "$en_file")
            zh_date=$(git log -1 --format="%cs" -- "$zh_file")
            
            echo "[NO_TRACKING] $rel_path: EN=$en_date > ZH=$zh_date (add source_commit: ${en_latest:0:7})"
        fi
    fi
done < <(find "$DOCS_DIR" -type f -name "*.md" | sort)

echo ""
echo "=== Summary ==="
echo "  OUTDATED:     $outdated file(s) need translation update"
echo "  SYNC_ONLY:    $sync_only file(s) need source_commit update only"
echo "  NO_TRACKING:  $no_tracking file(s) missing source_commit (timestamp fallback)"
echo ""
echo "Use SHOW_DIFF=0 for list only."
