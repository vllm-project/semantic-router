// e2e/pkg/verification/ciexclusions.go

package verification

// CIExclusion bounds one selected-but-CI-omitted testcase to a disposition:
// either an issue that owns deciding its re-inclusion (Issue) or a
// documented reason the omission is deliberate today (Rationale). Exactly
// one of the two must be set.
type CIExclusion struct {
	Issue     int
	Rationale string
}

// KnownCIExclusions is, per profile with an explicit CI override, the exact
// set of testcases the profile selects at runtime that the override omits
// from CI execution on main.
//
// Like KnownUnreachableDebt, this table is a ratchet, not an allowlist: Gate
// D requires the actual excluded set to equal these keys in both directions.
// A testcase newly selected by the profile but omitted from the CI list
// fails the gate until it is added to the list or bounded here (the
// silent-omission scenario); an entry whose testcase joins the CI list or
// leaves the profile selection also fails the gate until it is deleted. The
// table can only shrink intentionally, and a profile whose override is
// removed from the workflow must have its whole table deleted with it.
var KnownCIExclusions = map[string]map[string]CIExclusion{
	// The envoy-ai-gateway override currently runs 17 of the profile's 34
	// selected testcases in CI. #2379 (the coverage epic) owns the
	// disposition of the issue-bounded entries: re-inclusion in the
	// explicit list, or hand-off to a coverage-family child issue.
	"envoy-ai-gateway": {
		"anthropic-messages-protocol-headers": {Issue: 2379},
		"anthropic-messages-request":          {Issue: 2379},
		"anthropic-messages-response-shape":   {Issue: 2379},
		"anthropic-messages-streaming":        {Issue: 2379},
		"apiserver-classification-endpoints":  {Issue: 2379},
		"chat-completions-structured-output":  {Issue: 2379},
		"entrypoint-recipe-routing":           {Issue: 2379},
		"llm-classifier-distribution-routing": {Issue: 2379},
		"looper-latency-token-headers":        {Issue: 2379},
		"protocol-codec-openai-regression":    {Issue: 2379},
		"retention-directive":                 {Issue: 2379},
		"sequence-classifier-routing":         {Issue: 2379},
		"session-pricing-chat-completions":    {Issue: 2379},
		"session-pricing-response-api":        {Issue: 2379},
		"session-telemetry-metrics":           {Issue: 2379},

		// The workflow's own comment above the override deliberately skips
		// stress / pressure coverage "until the suite is stable again";
		// re-inclusion is a CI-owner decision tracked by that comment, not
		// a coverage gap this gate should force.
		"chat-completions-progressive-stress": {
			Rationale: "the workflow's ENVOY_AI_GATEWAY_CI_TESTS override deliberately skips stress / pressure coverage until the suite is stable again (see the comment above the list in integration-test-k8s.yml); re-inclusion is a CI-owner decision",
		},
		"chat-completions-stress-request": {
			Rationale: "the workflow's ENVOY_AI_GATEWAY_CI_TESTS override deliberately skips stress / pressure coverage until the suite is stable again (see the comment above the list in integration-test-k8s.yml); re-inclusion is a CI-owner decision",
		},
	},
}
