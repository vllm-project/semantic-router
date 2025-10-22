# Sample Output from Documentation Accuracy Checker

This document shows example outputs from running the documentation accuracy checker.

## Console Output

```
Starting Documentation Accuracy Checker
Epochs: 2
Repository: /home/runner/work/semantic-router/semantic-router
Documentation: /home/runner/work/semantic-router/semantic-router/website
Seed: 80

================================================================================
EPOCH 1/2
================================================================================

Step 1: Partitioning documents for epoch 0...
Selected 20 documents for this epoch:
  - website/docs/installation/docker-compose.md
  - website/docs/overview/architecture/envoy-extproc.md
  - website/docs/overview/architecture/router-implementation.md
  - website/docs/overview/architecture/system-architecture.md
  - website/docs/overview/categories/overview.md
  - website/docs/overview/categories/supported-categories.md
  - website/docs/overview/categories/technical-details.md
  - website/docs/overview/semantic-router-overview.md
  - website/docs/proposals/nvidia-dynamo-integration.md
  - website/docs/proposals/production-stack-integration.md
  ... and 10 more

Step 2: Building capability inventory...
Discovered 606 capabilities:
  - API: 424
  - config: 155
  - env: 27

Step 3: Comparing documentation to code...
Found 88 potential issues:
  - hallucination: 84
  - missing: 4

Step 4: Generating patches...
Generated 10 patch files

Step 5: Validating changes...
Running build command: make docs-build
Running linkcheck command: make markdown-lint-fix docs-lint-fix
✓ Build succeeded

✓ Epoch 1 complete. Results saved to /tmp/docs-accuracy-epoch-0

================================================================================
EPOCH 2/2
================================================================================

Step 1: Partitioning documents for epoch 1...
Selected 19 documents for this epoch:
  - website/docs/api/classification.md
  - website/docs/api/router.md
  - website/docs/installation/configuration.md
  - website/docs/installation/installation.md
  - website/docs/installation/kubernetes.md
  ...

Step 2: Building capability inventory...
Discovered 606 capabilities:
  - API: 424
  - config: 155
  - env: 27

Step 3: Comparing documentation to code...
Found 183 potential issues:
  - hallucination: 182
  - missing: 1

Step 4: Generating patches...
Generated 14 patch files

Step 5: Validating changes...
✓ Build succeeded

✓ Epoch 2 complete. Results saved to /tmp/docs-accuracy-epoch-1

================================================================================
FINAL REPORT
================================================================================

Total epochs: 2
Total docs checked: 39
Total capabilities discovered: 200
Total issues found: 100
Total claims checked: 390
Total claims fixed: 40

Final report saved to: /tmp/docs-accuracy-final-report.json

✓ Documentation accuracy check complete!
```

## JSON Report Examples

### Final Report (`docs-accuracy-final-report.json`)

```json
{
  "summary": {
    "total_epochs": 2,
    "total_docs_checked": 39,
    "total_capabilities_discovered": 200,
    "total_issues_found": 100,
    "total_claims_checked": 390,
    "total_claims_fixed": 40
  },
  "epochs": [
    {
      "epoch": 1,
      "docs_checked": 20,
      "capabilities_found": 100,
      "issues_found": 50,
      "build_success": true,
      "claims_checked": 200,
      "claims_fixed": 20
    },
    {
      "epoch": 2,
      "docs_checked": 19,
      "capabilities_found": 100,
      "issues_found": 50,
      "build_success": true,
      "claims_checked": 190,
      "claims_fixed": 20
    }
  ]
}
```

### Capability Inventory (`capabilities.json`)

```json
[
  {
    "name": "bert_model",
    "type": "config",
    "default": null,
    "valid_values": null,
    "version": null,
    "feature_gate": null,
    "source_paths": [
      "config/config.e2e.yaml:1"
    ],
    "description": null
  },
  {
    "name": "semantic_cache",
    "type": "config",
    "default": null,
    "valid_values": null,
    "version": null,
    "feature_gate": null,
    "source_paths": [
      "config/config.e2e.yaml:5"
    ],
    "description": null
  },
  {
    "name": "ClassifyRequest",
    "type": "API",
    "default": null,
    "valid_values": null,
    "version": null,
    "feature_gate": null,
    "source_paths": [
      "src/training/dual_classifier/dual_classifier.py:45"
    ],
    "description": null
  },
  {
    "name": "HUGGINGFACE_TOKEN",
    "type": "env",
    "default": null,
    "valid_values": null,
    "version": null,
    "feature_gate": null,
    "source_paths": [
      "scripts/download_models.sh:12"
    ],
    "description": null
  }
]
```

### Issues Report (`issues.json`)

```json
[
  {
    "doc_path": "website/docs/installation/docker-compose.md",
    "line_number": 37,
    "issue_type": "hallucination",
    "current_text": "- Docker Compose v2 (`docker compose` command, not the legacy `docker-compose`)",
    "proposed_fix": "VERIFY: Check if this configuration/API exists in codebase",
    "justification": "'docker-compose' not found in capability inventory",
    "evidence_citations": [
      "Capability inventory scan"
    ],
    "confidence": "medium"
  },
  {
    "doc_path": "website/docs/api/router.md",
    "line_number": 125,
    "issue_type": "outdated",
    "current_text": "Default timeout is 30 seconds",
    "proposed_fix": "Update to reflect current default of 60 seconds",
    "justification": "Config shows default timeout as 60s",
    "evidence_citations": [
      "config/config.yaml:45"
    ],
    "confidence": "high"
  },
  {
    "doc_path": "",
    "line_number": null,
    "issue_type": "missing",
    "current_text": "",
    "proposed_fix": "Add documentation for config 'tracing_enabled'",
    "justification": "Capability exists in code but not documented",
    "evidence_citations": [
      "config/config.tracing.yaml:12"
    ],
    "confidence": "medium"
  }
]
```

### Validation Report (`validation.json`)

```json
{
  "epoch": 1,
  "build_success": true,
  "build_output": "npm run build\n> build\n> docusaurus build\n\n[SUCCESS] Generated static files in build/",
  "linkcheck_output": "markdownlint checking complete\nNo broken links found",
  "claims_checked": 200,
  "claims_fixed": 20,
  "claims_remaining": 30,
  "unverified_count": 5,
  "broken_links_before": 0,
  "broken_links_after": 0,
  "pages_touched": 20,
  "confidence_ratings": {
    "website/docs/api/router.md": "High",
    "website/docs/installation/configuration.md": "Medium"
  }
}
```

## Patch Output Example

```markdown
# Patch for website/docs/api/router.md
# Epoch 0
# Issues found: 3

## OUTDATED
Line: 125
Current: Default timeout is 30 seconds...
Proposed: Update to reflect current default of 60 seconds
Evidence: config/config.yaml:45

## HALLUCINATION
Line: 156
Current: The 'legacy_mode' flag enables backward compatibility...
Proposed: VERIFY: Check if this configuration/API exists in codebase
Evidence: Capability inventory scan

## MISSING
Line: N/A
Current: 
Proposed: Add documentation for config 'tracing_enabled'
Evidence: config/config.tracing.yaml:12
```

## Directory Structure After Run

```
/tmp/
├── docs-accuracy-epoch-0/
│   ├── capabilities.json      # Discovered capabilities
│   ├── issues.json            # Documentation issues
│   └── validation.json        # Build and validation results
├── docs-accuracy-epoch-1/
│   ├── capabilities.json
│   ├── issues.json
│   └── validation.json
└── docs-accuracy-final-report.json  # Summary across all epochs
```

## Interpreting Results

### Issue Types

1. **Hallucination**: Documentation mentions features/configs that don't exist in code
   - **Action**: Remove or verify the claim with SMEs
   - **Example**: Documented config key not found in any YAML file

2. **Outdated**: Documentation doesn't match current implementation
   - **Action**: Update documentation to match code
   - **Example**: Default value changed but docs not updated

3. **Missing**: Code features not documented
   - **Action**: Add documentation for the feature
   - **Example**: New config option added to code but not in docs

### Confidence Levels

- **High**: Strong evidence from code, likely accurate issue
- **Medium**: Moderate evidence, should be reviewed
- **Low**: Weak evidence, may be false positive

### Next Steps

1. Review issues by confidence level (high → medium → low)
2. For each high-confidence issue:
   - Verify the evidence by checking source files
   - Update documentation or code as needed
   - Re-run checker to confirm fix
3. For medium/low confidence:
   - Manually inspect the claim
   - Determine if it's a real issue
   - Update checker heuristics if needed

## Integration with CI/CD

When integrated with GitHub Actions, the checker produces:

1. **Workflow artifacts** with all JSON reports
2. **PR comments** with summary statistics
3. **Step summaries** in the Actions UI
4. **Build status** indicators

This helps maintainers:
- Track documentation quality over time
- Catch doc-code drift early
- Ensure new features are documented
- Prevent hallucinated documentation
