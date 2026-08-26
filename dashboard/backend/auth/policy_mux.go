package auth

import (
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"sync"
)

// PolicyMux is the only registration seam for dashboard HTTP handlers. A
// protected handler cannot be installed without its complete route contract.
type PolicyMux struct {
	mux       *http.ServeMux
	mu        sync.RWMutex
	contracts map[string]RouteContract
	sealed    bool
}

func NewPolicyMux() *PolicyMux {
	return &PolicyMux{
		mux:       http.NewServeMux(),
		contracts: map[string]RouteContract{},
	}
}

func (m *PolicyMux) Handle(contract RouteContract, handler http.Handler) {
	m.HandleGroup([]RouteContract{contract}, handler)
}

// HandleGroup binds multiple concrete route contracts to one handler. All
// contracts are validated before any route is installed, so proxy dispatchers
// cannot register a handler separately from their authorization metadata.
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

func (m *PolicyMux) HandleFunc(contract RouteContract, handler http.HandlerFunc) {
	m.Handle(contract, handler)
}

// HandleFallback registers a non-API fallback such as the static frontend.
// API and embedded routes must use Handle, HandleFunc, or HandleGroup.
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

// Seal closes startup registration. Setup calls it before publishing the
// server so late route additions cannot bypass the validated inventory.
func (m *PolicyMux) Seal() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sealed = true
}

func (m *PolicyMux) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	m.mux.ServeHTTP(w, r)
}

func (m *PolicyMux) LookupRoutePolicy(method, path string) (RoutePolicy, RouteLookup) {
	method = strings.ToUpper(strings.TrimSpace(method))
	path = normalizePolicyPath(path)

	request := &http.Request{Method: method, URL: &url.URL{Path: path}}
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
		return RoutePolicy{
			Method:        http.MethodOptions,
			AuditMode:     AuditNone,
			Sensitivity:   SensitivityPublic,
			ResourceOwner: ResourceOwnerPublic,
			Public:        true,
		}, RouteFound
	}
	return RoutePolicy{}, RouteMethodNotAllowed
}

func (m *PolicyMux) Contracts() []RouteContract {
	m.mu.RLock()
	defer m.mu.RUnlock()

	contracts := make([]RouteContract, 0, len(m.contracts))
	for _, contract := range m.contracts {
		copyContract := contract
		copyContract.Policies = append([]RoutePolicy(nil), contract.Policies...)
		contracts = append(contracts, copyContract)
	}
	sort.Slice(contracts, func(i, j int) bool {
		return contracts[i].Pattern < contracts[j].Pattern
	})
	return contracts
}
