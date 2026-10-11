#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_SCRIPT="$SCRIPT_DIR/check-translation-sync.sh"
TEST_ROOT="$(mktemp -d)"

cleanup() {
    rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

commit_at() {
    local date="$1"
    local message="$2"

    GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" \
        git commit -q -am "$message"
}

assert_contains() {
    local text="$1"
    local expected="$2"

    [[ "$text" == *"$expected"* ]] || fail "expected output to contain: $expected"
}

mkdir -p \
    "$TEST_ROOT/website/docs/cases" \
    "$TEST_ROOT/website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/cases" \
    "$TEST_ROOT/website/scripts"
cp "$SOURCE_SCRIPT" "$TEST_ROOT/website/scripts/check-translation-sync.sh"

cd "$TEST_ROOT"
git init -q
git config user.name "Translation Sync Test"
git config user.email "translation-sync@example.com"

for name in definitely-outdated definitely-current verify-false verify-true fallback-only; do
    cat > "website/docs/cases/$name.md" <<EOF
# $name

Initial English content.
EOF
done
git add website/docs
commit_at "2026-01-01T00:00:00Z" "add English fixtures"
source_commit="$(git rev-parse --short HEAD)"

for name in definitely-outdated definitely-current verify-false verify-true; do
    outdated_value=false
    if [[ "$name" == definitely-current || "$name" == verify-true ]]; then
        outdated_value=true
    fi
    printf -- '---\ntranslation:\n  source_commit: "%s"\n  source_file: "docs/cases/%s.md"\n  outdated: %s\n---\n\n# %s\n\nInitial Chinese content.\n' \
        "$source_commit" "$name" "$outdated_value" "$name" \
        > "website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/cases/$name.md"
done
cat > website/i18n/translation-coverage-baseline.txt <<'EOF'
# Current documentation locale overrides.
# Regenerate with: make docs-update-translation-baseline
zh-Hans	cases/definitely-current.md
zh-Hans	cases/definitely-outdated.md
zh-Hans	cases/verify-false.md
zh-Hans	cases/verify-true.md
EOF
git add website/i18n
commit_at "2026-01-02T00:00:00Z" "add Chinese fixtures"

cat >> website/docs/cases/definitely-outdated.md <<'EOF'

New English content.
EOF
cat >> website/docs/cases/verify-false.md <<'EOF'

New English content.
EOF
cat >> website/docs/cases/verify-true.md <<'EOF'

New English content.
EOF
commit_at "2026-01-03T00:00:00Z" "update English fixtures"

cat >> website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/cases/verify-false.md <<'EOF'

Later Chinese-only edit without advancing source_commit.
EOF
cat >> website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/cases/verify-true.md <<'EOF'

Later Chinese-only edit without advancing source_commit.
EOF
commit_at "2026-01-04T00:00:00Z" "touch ambiguous Chinese fixture"

set +e
audit_output="$(website/scripts/check-translation-sync.sh --locale zh-Hans 2>&1)"
audit_status=$?
set -e

[[ $audit_status -eq 1 ]] || fail "audit should exit 1 when status work remains"
assert_contains "$audit_output" "Outdated translations (English commit is newer):"
assert_contains "$audit_output" "cases/definitely-outdated.md"
assert_contains "$audit_output" "Metadata needs verification (Chinese is not older):"
assert_contains "$audit_output" "cases/verify-false.md"
assert_contains "$audit_output" "cases/verify-true.md"
assert_contains "$audit_output" "cases/definitely-current.md"
assert_contains "$audit_output" "English fallback:"
assert_contains "$audit_output" "1 English fallback"
assert_contains "$audit_output" "cases: 1"

coverage_only_output="$(website/scripts/check-translation-sync.sh --coverage-only 2>&1)" \
    || fail "coverage-only audit should ignore source drift and metadata work"
assert_contains "$coverage_only_output" "0 coverage regressions"

deleted_translation="website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/cases/definitely-current.md"
rm "$deleted_translation"
set +e
coverage_output="$(website/scripts/check-translation-sync.sh --coverage-only 2>&1)"
coverage_status=$?
set -e
[[ $coverage_status -eq 1 ]] || fail "audit should fail when a baseline translation is deleted"
assert_contains "$coverage_output" "Coverage regressions (baseline override deleted):"
assert_contains "$coverage_output" "cases/definitely-current.md"
git checkout -- "$deleted_translation"

set +e
fix_output="$(website/scripts/check-translation-sync.sh --locale zh-Hans --fix-status 2>&1)"
fix_status=$?
set -e

[[ $fix_status -eq 1 ]] || fail "fix mode should exit 1 while translation work remains"
assert_contains "$fix_output" "outdated set to true"
assert_contains "$fix_output" "outdated set to false"

outdated_file="website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/cases/definitely-outdated.md"
current_file="website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/cases/definitely-current.md"
verify_false_file="website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/cases/verify-false.md"
verify_true_file="website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/cases/verify-true.md"

if ! grep -q '^  outdated: true$' "$outdated_file"; then
    printf '%s\n' "$fix_output" >&2
    sed -n '1,8p' "$outdated_file" >&2
    fail "definitely outdated file was not marked true"
fi
grep -q '^  outdated: false$' "$current_file" || fail "definitely current file was not marked false"
grep -q '^  outdated: false$' "$verify_false_file" || fail "ambiguous false status should remain unchanged"
grep -q '^  outdated: true$' "$verify_true_file" || fail "ambiguous true status should remain unchanged"

changed_files="$(git diff --name-only)"
assert_contains "$changed_files" "cases/definitely-outdated.md"
assert_contains "$changed_files" "cases/definitely-current.md"
[[ "$changed_files" != *"cases/verify-false.md"* ]] || fail "ambiguous false file was modified"
[[ "$changed_files" != *"cases/verify-true.md"* ]] || fail "ambiguous true file was modified"

unexpected_diff="$(git diff -U0 | grep -E '^[+-]' | grep -Ev '^---|^\+\+\+|^[+-]  outdated: (true|false)$' || true)"
[[ -z "$unexpected_diff" ]] || fail "fix mode changed content other than outdated flags"

before="$(git diff | cksum)"
website/scripts/check-translation-sync.sh --locale zh-Hans --fix-status >/dev/null || true
after="$(git diff | cksum)"
[[ "$before" == "$after" ]] || fail "fix mode is not idempotent"

mkdir -p website/i18n/fr/docusaurus-plugin-content-docs/current
set +e
fallback_output="$(website/scripts/check-translation-sync.sh --locale fr 2>&1)"
fallback_status=$?
set -e

[[ $fallback_status -eq 0 ]] || fail "English fallback without stale overrides should pass"
assert_contains "$fallback_output" "5 English fallback"

printf 'zh-Hans\tcases/retired.md\n' >> website/i18n/translation-coverage-baseline.txt
set +e
stale_baseline_output="$(website/scripts/check-translation-sync.sh --coverage-only 2>&1)"
stale_baseline_status=$?
set -e
[[ $stale_baseline_status -eq 1 ]] || fail "audit should fail when the baseline names a retired source"
assert_contains "$stale_baseline_output" "Stale baseline entries (English source retired or moved):"
assert_contains "$stale_baseline_output" "cases/retired.md"

website/scripts/check-translation-sync.sh --update-baseline >/dev/null
if grep -q 'cases/retired.md' website/i18n/translation-coverage-baseline.txt; then
    fail "baseline refresh retained a retired path"
fi
for name in definitely-current definitely-outdated verify-false verify-true; do
    grep -q $'^zh-Hans\tcases/'"$name"'.md$' website/i18n/translation-coverage-baseline.txt \
        || fail "baseline refresh omitted cases/$name.md"
done

echo "Translation sync behavior test passed."
