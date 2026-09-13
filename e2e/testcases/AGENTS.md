# E2E testcases

- Each testcase must assert an externally visible contract with an explicit
  pass/fail condition.
- A benchmark or report-only probe does not replace acceptance coverage.
- Accuracy, rate, and latency assertions need a stated non-zero threshold;
  “one request succeeded” is not a routing or classifier acceptance bar.
- Shared helpers may hide mechanics, never acceptance semantics.
