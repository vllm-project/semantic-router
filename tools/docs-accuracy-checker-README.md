# Documentation Accuracy Checker (Epochic Loop)

## Overview

The Documentation Accuracy Checker is an automated system that iteratively improves project documentation by grounding every claim in the source code and configuration files. It runs for a fixed number of epochs and shows measurable accuracy gains after each iteration.

## Features

- **Deterministic Document Partitioning**: Distributes documentation files across epochs using stable hashing
- **Capability Inventory**: Automatically discovers APIs, configs, environment variables, and features from the codebase
- **Doc-Code Comparison**: Identifies outdated claims, missing features, and hallucinated content
- **Evidence-Based Fixes**: Every proposed change is backed by citations to source code
- **Validation Reports**: Includes build status, link checks, and accuracy metrics
- **Machine-Readable Output**: Generates JSON reports for automated processing

## Usage

### Basic Usage

Run with default settings (20 epochs):

```bash
make docs-accuracy-check
```

Or run directly:

```bash
python3 tools/docs-accuracy-checker.py
```

### Quick Test

Run a quick test with only 5 epochs:

```bash
make docs-accuracy-check-quick
```

### Advanced Usage

Customize the checker behavior:

```bash
python3 tools/docs-accuracy-checker.py \
  --epochs 10 \
  --repo-root . \
  --docs-root website \
  --seed 42 \
  --build-cmd "make docs-build" \
  --linkcheck-cmd "make markdown-lint-fix docs-lint-fix"
```

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 20 | Number of epochs to run |
| `--repo-root` | `.` | Repository root path |
| `--docs-root` | `website` | Documentation root path |
| `--docs-globs` | `website/docs/**/*.md` `website/docs/**/*.mdx` | Documentation file patterns |
| `--exclude-globs` | `**/node_modules/**` `**/.cache/**` `**/build/**` | Patterns to exclude |
| `--primary-branch` | `main` | Primary branch name |
| `--seed` | 80 | Random seed for deterministic partitioning |
| `--build-cmd` | `make docs-build` | Command to build documentation |
| `--linkcheck-cmd` | `make markdown-lint-fix docs-lint-fix` | Command to check links and lint |

## Output

The tool generates the following outputs:

### Per-Epoch Outputs

For each epoch `N`, files are saved to `/tmp/docs-accuracy-epoch-N/`:

- `capabilities.json`: Discovered capabilities from the codebase
- `issues.json`: Documentation issues found (outdated, missing, hallucinated)
- `validation.json`: Build status and metrics

### Final Report

A comprehensive report is saved to `/tmp/docs-accuracy-final-report.json` containing:

- Summary across all epochs
- Total documents checked
- Total capabilities discovered
- Total issues found
- Total claims checked and fixed

## How It Works

### Epoch Loop

For each epoch, the system:

1. **Partition Documents**: Selects a deterministic subset of documentation files
2. **Build Capability Inventory**: Scans codebase for APIs, configs, flags, and environment variables
3. **Compare Docs to Code**: Identifies mismatches between documentation and implementation
4. **Generate Patches**: Creates proposed fixes with evidence citations
5. **Validate Changes**: Runs build and link check commands
6. **Report Metrics**: Generates JSON reports with accuracy metrics

### Capability Discovery

The system discovers capabilities from:

- **Config Files**: YAML configuration keys and defaults
- **Python Source**: Classes, functions, and environment variables
- **Go Source**: Exported functions and types
- **Rust Source**: Public APIs (if applicable)

### Issue Detection

The system identifies three types of issues:

1. **Outdated Claims**: Documentation doesn't match current implementation
2. **Missing Features**: Code capabilities not documented
3. **Hallucinations**: Documented features that don't exist in code

### Evidence Requirements

Every proposed change includes:

- **Current Text**: Quote from documentation
- **Proposed Fix**: Specific correction or addition
- **Justification**: Explanation of the issue
- **Evidence Citations**: File paths and line numbers from source code
- **Confidence Level**: Low, medium, or high

## Integration with CI/CD

### Pre-Commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: docs-accuracy-check
      name: Documentation Accuracy Check
      entry: python3 tools/docs-accuracy-checker.py --epochs 5
      language: system
      pass_filenames: false
```

### GitHub Actions

Add to `.github/workflows/docs-check.yml`:

```yaml
name: Documentation Accuracy Check

on:
  pull_request:
    paths:
      - 'website/docs/**'
      - 'config/**'
      - 'src/**'

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Run documentation accuracy check
        run: |
          python3 tools/docs-accuracy-checker.py --epochs 5
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: docs-accuracy-results
          path: /tmp/docs-accuracy-*.json
```

## Grounding Rules

The system follows strict grounding rules:

1. **Evidence Required**: Every change must be backed by code/config evidence
2. **Citation Format**: Use `file:line` or `file:line-range` format
3. **Version Awareness**: Document behavior differences across versions
4. **Feature Gates**: Note when features are behind flags
5. **Ambiguity Handling**: Document ambiguities rather than inventing behavior
6. **No Hallucinations**: Never invent features; mark unverified items as `UNVERIFIED`

## Deterministic Partitioning

Documents are partitioned across epochs using a stable hash function:

```python
hash = SHA1(file_path + seed)
epoch = hash % total_epochs
```

This ensures:

- **Reproducibility**: Same seed produces same partitions
- **Coverage**: Each document assigned to exactly one epoch
- **Balance**: Approximately equal distribution across epochs

## Example Output

### Capability Inventory

```json
{
  "name": "router_mode",
  "type": "config",
  "default": "semantic",
  "source_paths": ["config/config.yaml:15"],
  "description": "Router operation mode"
}
```

### Documentation Issue

```json
{
  "doc_path": "website/docs/api/router.md",
  "line_number": 42,
  "issue_type": "outdated",
  "current_text": "Default mode is `simple`",
  "proposed_fix": "Default mode is `semantic`",
  "justification": "Config shows semantic as default",
  "evidence_citations": ["config/config.yaml:15"],
  "confidence": "high"
}
```

### Validation Report

```json
{
  "epoch": 1,
  "build_success": true,
  "claims_checked": 150,
  "claims_fixed": 12,
  "claims_remaining": 8,
  "unverified_count": 3,
  "pages_touched": 15
}
```

## Troubleshooting

### Build Failures

If documentation build fails:

1. Check `build_output` in validation report
2. Ensure dependencies are installed: `make docs-install`
3. Test build manually: `make docs-build`

### No Capabilities Found

If capability discovery returns empty results:

1. Verify `--repo-root` points to correct directory
2. Check that source code exists in `src/` directory
3. Ensure config files exist in `config/` directory

### Partitioning Issues

If documents not distributed properly:

1. Try different `--seed` values
2. Check `--docs-globs` patterns match your files
3. Verify `--exclude-globs` aren't too broad

## Contributing

To extend the checker:

1. **Add Capability Types**: Extend `discover_capabilities()` for new languages
2. **Add Issue Detectors**: Extend `compare_docs_to_code()` for new checks
3. **Add Validators**: Extend `validate_changes()` for additional checks

## License

Apache 2.0 - See LICENSE file for details
