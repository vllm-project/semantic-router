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
//
// The table is empty because integration-test-k8s.yml currently declares no
// explicit per-profile testcase list: the former ENVOY_AI_GATEWAY_CI_TESTS
// override was replaced by the E2E_BASELINE_SUITE selector, which is not a
// testcase list, so there is no omission for this table to bound. A future
// explicit override must record its omissions here before Gate D passes.
var KnownCIExclusions = map[string]map[string]CIExclusion{}
