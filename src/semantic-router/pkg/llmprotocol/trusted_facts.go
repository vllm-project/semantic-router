package llmprotocol

// Trusted tool and MCP capability facts (issue #3476).
//
// These types are intentionally dependency-free: the config package owns the
// YAML surface (pkg/config.TrustedFactsConfig) while this package owns the
// request-path eligibility vocabulary. The decision engine threads the
// config through this gate before relevance or learned ranking so capability,
// authorization, availability, and stage role stay one permission plane and
// never become a parallel MCP policy.

type TrustedEnforcement string

const (
	TrustedDisabled      TrustedEnforcement = "disabled"
	TrustedAdvisory      TrustedEnforcement = "advisory"
	TrustedAuthoritative TrustedEnforcement = "authoritative"
)

type TrustedStage string

const (
	TrustedStageCandidate TrustedStage = "candidate"
	TrustedStageVerifier  TrustedStage = "verifier"
	TrustedStageAdvisor   TrustedStage = "advisor"
	TrustedStageFinal     TrustedStage = "final"
)

// TrustedFacts carries the effective request-path inputs. TrustSources must
// already be validated against the operator/gateway/runtime allowlist;
// Stage is the Looper role requesting tools. Fresh indicates whether
// runtime availability evidence is within the configured freshness bound.
type TrustedFacts struct {
	Enforcement  TrustedEnforcement
	TrustSources []string
	Stage        TrustedStage
	Fresh        bool
}

// TrustedOutcome is the gate result. It never carries raw prompts,
// arguments, results, or credentials — callers record only the outcome,
// enforcement, and stage.
type TrustedOutcome string

const (
	TrustedAllow   TrustedOutcome = "allow"
	TrustedNarrow  TrustedOutcome = "narrow"
	TrustedDeny    TrustedOutcome = "deny"
	TrustedObserve TrustedOutcome = "observe"
)

// EvaluateTrustedFacts applies the eligibility gate. Disabled always allows
// (existing behavior unchanged). Advisory never denies — it observes.
// Authoritative allows only when at least one trust source is present, the
// stage is known, and freshness holds; otherwise it narrows (stale) or
// denies (missing source or unknown stage). It never widens privileges and
// never executes tools.
//
// Contract-only in this change (issue #3476): the helper and its unit tests
// define the gate vocabulary. Wiring into the decision engine before
// relevance/ranking is a tracked follow-up so this change stays additive
// and cannot alter existing selection when trusted_facts is disabled.
func EvaluateTrustedFacts(f TrustedFacts) TrustedOutcome {
	switch f.Enforcement {
	case TrustedDisabled:
		return TrustedAllow
	case TrustedAdvisory, "":
		return TrustedObserve
	case TrustedAuthoritative:
		if len(f.TrustSources) == 0 {
			return TrustedDeny
		}
		switch f.Stage {
		case TrustedStageCandidate, TrustedStageVerifier, TrustedStageAdvisor, TrustedStageFinal:
		default:
			return TrustedDeny
		}
		if !f.Fresh {
			return TrustedNarrow
		}
		return TrustedAllow
	default:
		return TrustedDeny
	}
}
