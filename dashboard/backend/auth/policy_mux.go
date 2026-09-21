package auth

import (
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"sync"
)

// PolicyMux is the only registration seam for Dashboard HTTP handlers. A
// handler in a protected namespace cannot be installed without the complete
// route contract that authorizes it, and the same registry that dispatches a
// request also answers the authentication middleware's policy lookup.
type PolicyMux struct {
	mux       *http.ServeMux
	mu        sync.RWMutex
	contracts map[string]RouteContract
	sealed    bool
}

// NewPolicyMux returns an empty registry.
func NewPolicyMux() *PolicyMux {
	return &PolicyMux{
		mux:       http.NewServeMux(),
		contracts: map[string]RouteContract{},
	}
}

// Handle binds one validated contract to a handler.
func (m *PolicyMux) Handle(contract RouteContract, handler http.Handler) {
	m.HandleGroup([]RouteContract{contract}, handler)
}

// HandleFunc binds one validated contract to a handler function.
func (m *PolicyMux) HandleFunc(contract RouteContract, handler http.HandlerFunc) {
	m.Handle(contract, handler)
}

// HandleGroup binds several contracts to one dispatching handler. Every
// contract is validated before any route is installed, so a dispatcher cannot
// register its handler separately from part of its authorization metadata.
func (m *PolicyMux) HandleGroup(contracts []RouteContract, handler http.Handler) {
	if handler == nil {
		panic("route handler is required")
	}
	if len(contracts) == 0 {
		panic("at least one route contract is required")
	}

	validated := make([]RouteContract, 0, len(contracts))
	seen := make(map[string]struct{}, len(contracts))
	for _, contract := range contracts {
		contract = mustValidateRouteContract(contract)
		if _, exists := seen[contract.Pattern]; exists {
			panic(fmt.Sprintf("route contract already declared for %q in handler group", contract.Pattern))
		}
		seen[contract.Pattern] = struct{}{}
		validated = append(validated, contract)
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if m.sealed {
		panic("route registry is sealed")
	}
	for _, contract := range validated {
		if _, exists := m.contracts[contract.Pattern]; exists {
			panic(fmt.Sprintf("route contract already registered for %q", contract.Pattern))
		}
	}
	for _, contract := range validated {
		m.mux.Handle(contract.Pattern, handler)
		m.contracts[contract.Pattern] = contract
	}
}

// HandleFallback registers a handler outside the protected namespaces, such as
// the static frontend or a loopback-only internal endpoint. Protected
// namespaces must use Handle, HandleFunc, or HandleGroup.
func (m *PolicyMux) HandleFallback(pattern string, handler http.Handler) {
	if handler == nil {
		panic("fallback handler is required")
	}
	pattern = normalizeRoutePattern(pattern)
	if isProtectedNamespace(pattern) {
		panic(fmt.Sprintf("protected fallback %q requires a route contract", pattern))
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if m.sealed {
		panic("route registry is sealed")
	}
	m.mux.Handle(pattern, handler)
}

// Seal closes registration. Setup seals the registry before publishing the
// server so a late registration cannot bypass the validated inventory.
func (m *PolicyMux) Seal() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sealed = true
}

// ServeHTTP dispatches to the registered handler.
func (m *PolicyMux) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	m.mux.ServeHTTP(w, r)
}

// LookupRoutePolicy resolves the request the same way ServeHTTP will dispatch
// it and returns the policy for its method. OPTIONS on a registered route is
// public so handlers can answer CORS preflight.
func (m *PolicyMux) LookupRoutePolicy(method, path string) (RoutePolicy, RouteLookup) {
	method = strings.ToUpper(strings.TrimSpace(method))
	request := &http.Request{Method: method, URL: &url.URL{Path: normalizePolicyPath(path)}}
	_, pattern := m.mux.Handler(request)

	m.mu.RLock()
	contract, found := m.contracts[pattern]
	m.mu.RUnlock()
	if !found {
		return RoutePolicy{}, RouteNotFound
	}
	for _, policy := range contract.Policies {
		if policy.Method == method {
			return policy, RouteFound
		}
	}
	if method == http.MethodOptions {
		return optionsPolicy(), RouteFound
	}
	return RoutePolicy{}, RouteMethodNotAllowed
}

// Contracts returns a sorted copy of every registered contract.
func (m *PolicyMux) Contracts() []RouteContract {
	m.mu.RLock()
	defer m.mu.RUnlock()

	contracts := make([]RouteContract, 0, len(m.contracts))
	for _, contract := range m.contracts {
		copied := contract
		copied.Policies = make([]RoutePolicy, len(contract.Policies))
		for index, policy := range contract.Policies {
			policy.Permissions = append([]string(nil), policy.Permissions...)
			copied.Policies[index] = policy
		}
		contracts = append(contracts, copied)
	}
	sort.Slice(contracts, func(i, j int) bool {
		return contracts[i].Pattern < contracts[j].Pattern
	})
	return contracts
}
