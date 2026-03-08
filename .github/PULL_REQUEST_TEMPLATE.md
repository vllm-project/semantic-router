FILL IN THE PR DESCRIPTION HERE

FIX #xxxx (*link existing issues this PR will resolve*)

## Change Summary

- Task type:
- Primary skill:
- Impacted surfaces:
- Conditional surfaces intentionally skipped:
- Touched subsystems:
- Environment used for local validation: `cpu-local` / `amd-local`
- Behavior-visible change: `yes` / `no`
- E2E added or updated:
- Debt register:

## Validation

- Fast gate:
- Feature gate:
- Local smoke:
- Local E2E profiles:
- CI expectations:

**BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE**

---

- [ ] Make sure the code changes pass the [pre-commit](https://github.com/vllm-project/semantic-router/blob/main/CONTRIBUTING.md) checks.
- [ ] Resolve and record the project-level primary skill and impacted surfaces.
- [ ] If shared agent rules changed, update the relevant source-of-truth docs under `docs/agent/` and run `make agent-validate`.
- [ ] If the desired architecture still diverges from the implementation after this PR, add or update the relevant item in `docs/agent/tech-debt-register.md`.
- [ ] Run the agent fast gate for changed files.
- [ ] Run the agent feature gate for behavior-visible changes.
- [ ] Sign-off your commit by using <code>-s</code> when doing <code>git commit</code>
- [ ] Try to classify PRs for easy understanding of the type of changes, such as `[Bugfix]`, `[Feat]`, and `[CI]`.

<details>
<!-- inside this <details> section, markdown rendering does not work, so we use raw html here. -->
<summary><b> Detailed Checklist (Click to Expand) </b></summary>

<p>Thank you for your contribution to semantic-router! Before submitting the pull request, please ensure the PR meets the following criteria. This helps us maintain the code quality and improve the efficiency of the review process.</p>

<h3>PR Title and Classification</h3>
<p>Please try to classify PRs for easy understanding of the type of changes. The PR title is prefixed appropriately to indicate the type of change. Please use one of the following:</p>
<ul>
    <li><code>[Bugfix]</code> for bug fixes.</li>
    <li><code>[CI/Build]</code> for build or continuous integration improvements.</li>
    <li><code>[CLI]</code> for changes to the command-line interface tools.</li>
    <li><code>[Dashboard]</code> for changes to the dashboard or web UI.</li>
    <li><code>[Doc]</code> for documentation fixes and improvements.</li>
    <li><code>[Feat]</code> for new features in the cluster (e.g., autoscaling, disaggregated prefill, etc.).</li>
    <li><code>[Router]</code> for changes to the <code>vllm_router</code> (e.g., routing algorithm, router observability, etc.).</li>
    <li><code>[Misc]</code> for PRs that do not fit the above categories. Please use this sparingly.</li>
</ul>
<p><strong>Note:</strong> If the PR spans more than one category, please include all relevant prefixes.</p>

<h3>Code Quality</h3>

<p>The PR need to meet the following code quality standards:</p>

<ul>
    <li>Pass all linter checks. Please use <code>pre-commit</code> to format your code. See <code>README.md</code> for installation.</li>
    <li>Pass the changed-file structure gate: file length, function length, nesting, and interface size.</li>
    <li>The code need to be well-documented to ensure future contributors can easily understand the code.</li>
    <li> Please include sufficient tests to ensure the change is stay correct and robust. This includes both unit tests and integration tests.</li>
    <li>For user-visible behavior changes, include or update at least one relevant E2E test.</li>
</ul>

<h3>DCO and Signed-off-by</h3>
<p>When contributing changes to this project, you must agree to the <a href="https://github.com/vllm-project/vllm/blob/main/DCO">DCO</a>. Commits must include a <code>Signed-off-by:</code> header which certifies agreement with the terms of the DCO.</p>
<p>Using <code>-s</code> with <code>git commit</code> will automatically add this header.</p>

<h3>What to Expect for the Reviews</h3>
