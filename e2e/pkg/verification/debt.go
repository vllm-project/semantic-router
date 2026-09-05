// e2e/pkg/verification/debt.go

package verification

// UnreachableDebt bounds one registered-but-unreachable testcase to a
// disposition: either a child issue that owns resolving it (Issue) or a
// documented reason it is manual-only today (Rationale). Exactly one of the
// two must be set.
//
// A child issue owning an entry means the family owns the disposition of the
// legacy testcase; it does not promise the child issue must preserve that
// exact testcase. Removal, rename, or replacement remain valid resolutions —
// each of which must also delete the entry here.
type UnreachableDebt struct {
	Issue     int
	Rationale string
}

// KnownUnreachableDebt is the exact set of registered-but-unreachable
// testcases on main, recorded as audit findings for issue #2379.
//
// This table is a debt ratchet, not an allowlist: the gate requires the
// actual unreachable set to equal these keys in both directions. A newly
// unreachable testcase fails the gate until it gets a profile mapping or a
// bounded disposition here; a resolved testcase also fails the gate until
// its stale entry is deleted. The table can only shrink intentionally.
var KnownUnreachableDebt = map[string]UnreachableDebt{
	// #3179 — selector algorithm coverage owns disposition of the legacy
	// core-selection cases (several reference algorithms that no longer
	// exist in the runtime decision-algorithm catalog).
	"core-selection-static":       {Issue: 3179},
	"core-selection-elo":          {Issue: 3179},
	"core-selection-elo-feedback": {Issue: 3179},
	"core-selection-routerdc":     {Issue: 3179},
	"core-selection-automix":      {Issue: 3179},
	"core-selection-hybrid":       {Issue: 3179},
	"core-selection-thompson":     {Issue: 3179},
	"core-selection-gmtrouter":    {Issue: 3179},
	"core-selection-router-r1":    {Issue: 3179},

	// #3178 — signal / classification-family coverage owns disposition of
	// the MCP classification cases (no registered profile deploys an MCP
	// classifier stack) and the superseded authz signal case.
	"mcp-http-classification":      {Issue: 3178},
	"mcp-stdio-classification":     {Issue: 3178},
	"mcp-model-reasoning":          {Issue: 3178},
	"mcp-fallback-behavior":        {Issue: 3178},
	"mcp-probability-distribution": {Issue: 3178},
	"authz-rbac-routing":           {Issue: 3178},

	// #3180 — decision plugin coverage owns disposition of the legacy
	// OpenAI-backed RAG case.
	"rag-openai": {Issue: 3180},

	// Documented manual-only rationale: mirrors the authz-rbac profile
	// contract, where direct client identity headers are stripped by the
	// production anti-spoofing path.
	"ratelimit-limitor": {
		Rationale: "direct x-authz-* client identity headers are stripped by the production anti-spoofing path, so this testcase cannot run under the authz-rbac profile until identity is supplied by a trusted post-auth source / owning profile",
	},
}
