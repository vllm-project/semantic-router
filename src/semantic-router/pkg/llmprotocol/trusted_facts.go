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

// TrustedSource names a declared trust-source value. The declarations are
// validated against this allowlist at config load
// (pkg/config.TrustedFactsConfig.Validate); the gate itself only consumes the
// resolved Authorized/Available facts below so availability evidence can never
// be mistaken for authorization.
type TrustedSource string

const (
	TrustedSourceOperatorPolicy  TrustedSource = "operator-policy"
	TrustedSourceGatewayAttested TrustedSource = "gateway-attested"
	TrustedSourceRuntimeFresh    TrustedSource = "runtime-fresh"
)

// Authorizes reports whether the declared source alone can authorize tool
// use. Only operator-owned recipe policy authorizes by declaration: gateway
// attestation must be verified per request (see TrustedFacts.Authorized) and
// runtime availability evidence can only narrow, never authorize.
func (s TrustedSource) Authorizes() bool {
	return s == TrustedSourceOperatorPolicy
}

// TrustedFacts carries the effective request-path inputs as distinct facts so
// capability, authorization, availability, and stage role stay one permission
// plane. Callers must resolve each plane from its own authoritative input:
// capability from the request/model capability gate, authorization from
// operator policy or verified gateway-attested context, availability from
// bounded fresh runtime evidence, and stage from the requesting Looper role
// checked against the decision's configured AllowedStages.
type TrustedFacts struct {
	Enforcement TrustedEnforcement
	// Capable reports that the request/model capability requirement for tools
	// is met.
	Capable bool
	// Authorized reports that operator policy or verified gateway-attested
	// context authorizes tool use. Runtime evidence must never set this.
	Authorized bool
	// Available reports that bounded fresh runtime availability evidence
	// holds. Stale or missing availability only narrows, never denies or
	// authorizes.
	Available bool
	// Stage is the Looper role requesting tools.
	Stage TrustedStage
	// AllowedStages lists the decision's configured stage roles. An empty
	// list denies: a policy that authorizes no role authorizes nothing.
	AllowedStages []TrustedStage
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
// Authoritative denies when capability is missing, when neither operator
// policy nor verified gateway attestation authorizes, or when the requesting
// stage is not one of the configured allowed roles; it narrows when only
// availability evidence is stale or missing, and allows otherwise. Runtime
// evidence can only narrow: it is never consulted for authorization. The gate
// never widens privileges and never executes tools.
func EvaluateTrustedFacts(f TrustedFacts) TrustedOutcome {
	switch f.Enforcement {
	case TrustedDisabled:
		return TrustedAllow
	case TrustedAdvisory, "":
		return TrustedObserve
	case TrustedAuthoritative:
		if !f.Capable {
			return TrustedDeny
		}
		if !f.Authorized {
			return TrustedDeny
		}
		if !trustedStageAllowed(f.Stage, f.AllowedStages) {
			return TrustedDeny
		}
		if !f.Available {
			return TrustedNarrow
		}
		return TrustedAllow
	default:
		return TrustedDeny
	}
}

// trustedStageAllowed reports whether the requesting stage is one of the
// configured allowed roles. An empty allow-list denies: a closed default for
// an authorization gate.
func trustedStageAllowed(stage TrustedStage, allowed []TrustedStage) bool {
	for _, s := range allowed {
		if s == stage {
			return true
		}
	}
	return false
}
