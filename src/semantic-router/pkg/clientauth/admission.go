package clientauth

import (
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Decision is the outcome of admitting one request.
type Decision int

const (
	// Admitted lets the request proceed.
	Admitted Decision = iota
	// Unauthorized refuses a request for a protected name that presents no
	// key the name accepts.
	Unauthorized
	// Superseded refuses a request whose name the configuration activated
	// while the request was arriving protects differently from the one that
	// serves it. A retry is served by the active configuration.
	Superseded
	// TargetNotAllowed refuses a name not declared in this API namespace.
	TargetNotAllowed
)

// Admit decides a request for model, in the api namespace, that presents the
// Authorization header authorization. served is the policy of the
// configuration that serves the request and active the policy of the one
// active once the request's target is known: a connection or stream leased
// before then grants nothing, so a request is admitted only when both protect
// its name alike, and then by its key. The name is trimmed as entrypoint
// resolution trims it.
func Admit(served, active *Policy, api config.InferenceAPI, model, authorization string) Decision {
	model = strings.TrimSpace(model)
	allowed := served.allowsTarget(api, model)
	if allowed != active.allowsTarget(api, model) || !served.protectsAlike(active, api, model) {
		return Superseded
	}
	if !allowed {
		return TargetNotAllowed
	}
	if !served.Authorize(api, model, BearerToken(authorization)) {
		return Unauthorized
	}
	return Admitted
}

// BearerToken returns the token of an Authorization header that presents a
// Bearer credential, or "" for any other header. A Bearer token is the only
// client credential: api-key and x-api-key headers carry none.
func BearerToken(authorization string) string {
	scheme, token, found := strings.Cut(strings.TrimSpace(authorization), " ")
	if !found || !strings.EqualFold(scheme, "Bearer") {
		return ""
	}
	return strings.TrimSpace(token)
}

// protectsAlike reports whether p and q accept the same keys for model in
// the api namespace, protecting it or not alike. The order the keys were
// declared in does not matter.
func (p *Policy) protectsAlike(q *Policy, api config.InferenceAPI, model string) bool {
	if p == q {
		return true
	}
	pKeys, pProtected := p.keys(api, model)
	qKeys, qProtected := q.keys(api, model)
	if pProtected != qProtected || len(pKeys) != len(qKeys) {
		return false
	}
	// Compile keeps each digest once, so equal lengths and containment make
	// equal sets.
	for _, key := range pKeys {
		if !slices.Contains(qKeys, key) {
			return false
		}
	}
	return true
}
