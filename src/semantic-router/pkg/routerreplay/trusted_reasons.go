package routerreplay

// Trusted tool capability reason (issue #3476).
//
// Content-minimized by construction: it records only the enforcement mode,
// stage role, trust sources, and outcome. It never stores raw prompts,
// tool arguments/results, credentials, or hidden reasoning.

type TrustedFactsReason struct {
	Enforcement  string   `json:"enforcement"`
	Stage        string   `json:"stage"`
	TrustSources []string `json:"trust_sources"`
	Outcome      string   `json:"outcome"`
}

// NewTrustedFactsReason builds a content-minimized reason. Callers pass only
// the already-validated enforcement/stage/sources/outcome strings; any raw
// content passed by mistake is dropped (only counts are kept).
func NewTrustedFactsReason(enforcement, stage string, sources []string, outcome string) TrustedFactsReason {
	cp := make([]string, 0, len(sources))
	cp = append(cp, sources...)
	return TrustedFactsReason{
		Enforcement:  enforcement,
		Stage:        stage,
		TrustSources: cp,
		Outcome:      outcome,
	}
}
