# Performance Testing CI Strategy

## The Problem You Identified

Running performance tests on **every PR** has significant costs:

- 💸 **Cost:** Burns 15-20 CI minutes per PR
- 🐌 **Speed:** Slows down developer workflow
- 📊 **Noise:** CI variance causes false positives
- 🔥 **Resources:** Downloads models, uses CPU intensively

**You're right to question this!**

---

## Current Setup (After Optimization)

The workflow now runs **only when needed**:

### ✅ Performance Tests Run When:

1. **PR has `performance` label** ← Developer explicitly requests it
2. **Manual trigger** ← Via GitHub Actions UI
3. ~~Every PR~~ ← **REMOVED to save costs**

### Usage:

```bash
# Developer workflow:
1. Open PR with code changes
2. Regular tests run (fast)
3. If touching performance-critical code:
   → Add "performance" label to PR
   → Performance tests run automatically
4. Review results in PR comment
```

---

## Alternative Strategies

Here are different approaches teams use, from most to least restrictive:

### Strategy 1: Label-Based (CURRENT - RECOMMENDED) 🏷️

**When it runs:**

- Only when PR has `performance` label
- Manual trigger via GitHub UI

**Pros:**

- ✅ Saves tons of CI time
- ✅ Developers control when tests run
- ✅ No noise on small PRs

**Cons:**

- ❌ Developers might forget to add label
- ❌ Regressions could slip through

**Best for:** Most teams, cost-conscious projects

---

### Strategy 2: Path-Based (Original Design) 📁

**When it runs:**

```yaml
on:
  pull_request:
    paths:
      - 'src/semantic-router/**'
      - 'candle-binding/**'
      - 'perf/**'
```

**Pros:**

- ✅ Automatic - no manual intervention
- ✅ Catches regressions early

**Cons:**

- ❌ Runs too often (most PRs touch these paths)
- ❌ High CI cost
- ❌ Slows down development

**Best for:** Critical production systems, unlimited CI budget

---

### Strategy 3: Scheduled + Manual Only ⏰

**When it runs:**

```yaml
on:
  schedule:
    - cron: "0 2 * * *"  # Daily at 2 AM
  workflow_dispatch:      # Manual only
```

**Pros:**

- ✅ Minimal CI cost
- ✅ No PR delays
- ✅ Nightly baseline still updates

**Cons:**

- ❌ Regressions found after merge (too late!)
- ❌ Developers must manually trigger

**Best for:** Early-stage projects, limited resources

---

### Strategy 4: Hybrid - Critical Paths Only 🎯

**When it runs:**

```yaml
on:
  pull_request:
    paths:
      - 'src/semantic-router/pkg/classification/**'  # Critical
      - 'src/semantic-router/pkg/cache/**'           # Critical
      - 'candle-binding/**'                          # Critical
      # NOT: docs, tests, configs, etc.
```

**Pros:**

- ✅ Automatic for critical code
- ✅ Reduced CI usage vs path-based
- ✅ Catches most important regressions

**Cons:**

- ❌ Still runs frequently
- ❌ Can miss indirect performance impacts

**Best for:** Mature projects with clear critical paths

---

### Strategy 5: PR Size Based 📏

**When it runs:**

```yaml
# Run only on large PRs (>500 lines changed)
if: github.event.pull_request.additions + github.event.pull_request.deletions > 500
```

**Pros:**

- ✅ Small PRs skip expensive tests
- ✅ Large risky changes get tested

**Cons:**

- ❌ Single-line change can cause regression
- ❌ Complex logic to maintain

**Best for:** Teams with predictable PR sizes

---

### Strategy 6: Pre-merge Only (Protected Branch) 🔒

**When it runs:**

```yaml
on:
  pull_request:
    types: [ready_for_review]  # Only when marked ready
  # OR
  push:
    branches: [main]  # Only after merge
```

**Pros:**

- ✅ Tests final code before/after merge
- ✅ Doesn't slow down draft PRs

**Cons:**

- ❌ Late feedback for developers
- ❌ Might catch issues post-merge

**Best for:** Fast-moving teams, trust-based workflows

---

## Recommended Setup by Project Stage

### 🌱 Early Stage Project

```yaml
Strategy: Scheduled + Manual
Performance Tests: Nightly only
Reason: Save CI budget, iterate fast
```

### 🌿 Growing Project

```yaml
Strategy: Label-Based (CURRENT)
Performance Tests: On 'performance' label
Reason: Balance cost vs safety
```

### 🌳 Mature Project

```yaml
Strategy: Hybrid Critical Paths
Performance Tests: Auto on critical code
Reason: High confidence, catch regressions
```

### 🏢 Enterprise Project

```yaml
Strategy: Every PR (Path-Based)
Performance Tests: Always
Reason: Zero tolerance for regressions
```

---

## How to Switch Strategies

### Switch to "Every PR" (Path-Based)

```yaml
# .github/workflows/performance-test.yml
on:
  pull_request:
    branches: [main]
    paths:
      - 'src/semantic-router/**'
      - 'candle-binding/**'

jobs:
  component-benchmarks:
    runs-on: ubuntu-latest
    # Remove the check-should-run job
    # Remove the needs/if conditions
```

### Switch to "Nightly Only"

```yaml
# .github/workflows/performance-test.yml
on:
  schedule:
    - cron: "0 3 * * *"
  workflow_dispatch:

# Disable PR trigger completely
```

### Keep Current (Label-Based)

No changes needed! Current setup is optimized.

---

## Cost Analysis

Assuming:

- 10 PRs per day
- 20 minutes per performance test
- $0.008 per minute (GitHub Actions pricing)

| Strategy | PRs Tested | CI Minutes/Day | Cost/Month |
|----------|------------|----------------|------------|
| Every PR | 10 | 200 min | $48/month |
| Label (25% use) | 2.5 | 50 min | $12/month |
| Critical Paths | 5 | 100 min | $24/month |
| Nightly Only | 0 | 0 min | $0/month |

**Current Label-Based:** Saves ~$36/month vs Every PR! 💰

---

## Best Practices

### For Developers

**When to add `performance` label:**

- ✅ Changing classification, cache, or decision engine
- ✅ Modifying CGO bindings
- ✅ Optimizing algorithms
- ✅ Changing batch processing logic
- ❌ Updating docs or tests
- ❌ Fixing typos
- ❌ Changing configs

### For Reviewers

**Check for performance label:**

```markdown
## Performance Checklist
- [ ] Does this PR touch classification/cache/decision code?
- [ ] Could this impact request latency?
- [ ] Should we add 'performance' label and run tests?
```

### For CI

**Monitor false negatives:**

- Track regressions found in nightly but missed in PRs
- If >5% slip through, consider tightening strategy

---

## FAQ

### Q: What if a regression slips through?

**A:** Nightly workflow will catch it and create an issue. You can:

1. Revert the problematic PR
2. Fix forward with a new PR
3. Update baseline if intentional

### Q: Can I force performance tests on a PR without label?

**A:** Yes! Two ways:

1. Add `performance` label to PR
2. Go to Actions tab → Performance Tests → Run workflow → Select your branch

### Q: What about main branch protection?

**A:** Performance tests are NOT required checks. They're:

- Advisory (warn but don't block)
- Opt-in (run when needed)
- Nightly will catch issues anyway

### Q: Should I run tests locally before PR?

**A:** Recommended for performance-critical changes:

```bash
make perf-bench-quick    # Takes 3-5 min
make perf-compare        # Compare vs baseline
```

---

## Summary

**Current Strategy: Label-Based ✅**

- Runs when PR has `performance` label
- Saves ~75% CI costs vs "every PR"
- Balances cost vs catching regressions
- Nightly workflow ensures baselines stay current

**To run performance tests on your PR:**

1. Add label: `performance`
2. Wait for tests to complete (~15 min)
3. Review results in PR comment

**Why nightly is still needed:**

- Updates baselines automatically
- Catches anything that slipped through
- Runs comprehensive 30s benchmarks
- Maintains performance history

**Best of both worlds:** Fast PRs + Accurate baselines! 🎯
