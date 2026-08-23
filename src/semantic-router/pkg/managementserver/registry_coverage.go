package managementserver

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

var registryPathParameter = regexp.MustCompile(`\{([A-Za-z][A-Za-z0-9]*)\}`)

// ValidateRegisteredOperations proves that the production registrar accepts
// every advertised Management operation. Coverage probes the combined mux and
// each registrar in isolation so a broad route from one domain cannot hide a
// missing owner in another. It does not execute domain handlers: a registered
// handler may legitimately return 404 when its probe resource does not exist.
func ValidateRegisteredOperations(registrar RouteRegistrar, disabledOperationIDs ...string) error {
	if nilRouteRegistrar(registrar) {
		return fmt.Errorf("management route registrar is required")
	}
	operations := managementapi.Operations()
	knownOperations := make(map[string]struct{}, len(operations))
	for _, operation := range operations {
		knownOperations[operation.OperationID] = struct{}{}
	}
	disabled := make(map[string]struct{}, len(disabledOperationIDs))
	for _, operationID := range disabledOperationIDs {
		_, known := knownOperations[operationID]
		if _, duplicate := disabled[operationID]; duplicate || operationID == "" || !known {
			return fmt.Errorf("invalid disabled Management operation %q", operationID)
		}
		disabled[operationID] = struct{}{}
	}
	registrars := []RouteRegistrar{registrar}
	if server, ok := registrar.(*Server); ok {
		if server == nil || len(server.routes) == 0 {
			return fmt.Errorf("management route registrar is required")
		}
		registrars = append([]RouteRegistrar(nil), server.routes...)
	}
	combined, err := registeredOperationMux(registrars)
	if err != nil {
		return err
	}
	owners := make([]*http.ServeMux, len(registrars))
	for index, candidate := range registrars {
		owners[index], err = registeredOperationMux([]RouteRegistrar{candidate})
		if err != nil {
			return err
		}
	}
	problems := make([]string, 0)
	for _, operation := range operations {
		pattern := registeredOperationPattern(combined, operation)
		_, excluded := disabled[operation.OperationID]
		if excluded {
			if pattern != "" {
				problems = append(problems, fmt.Sprintf(
					"disabled Management operation %s is registered by %q", operation.OperationID, pattern,
				))
			}
			continue
		}
		if pattern == "" {
			problems = append(problems, fmt.Sprintf(
				"Management operation %s (%s %s) is not registered", operation.OperationID, operation.Method, operation.Path,
			))
			continue
		}
		if !patternOwnsOperation(pattern, operation) {
			problems = append(problems, fmt.Sprintf(
				"Management operation %s (%s %s) is matched only by over-broad pattern %q",
				operation.OperationID, operation.Method, operation.Path, pattern,
			))
			continue
		}
		ownerIndexes := make([]int, 0, 1)
		for index, owner := range owners {
			if registeredOperationPattern(owner, operation) == pattern {
				ownerIndexes = append(ownerIndexes, index)
			}
		}
		if len(ownerIndexes) != 1 {
			problems = append(problems, fmt.Sprintf(
				"Management operation %s (%s %s) has %d route owners for pattern %q",
				operation.OperationID, operation.Method, operation.Path, len(ownerIndexes), pattern,
			))
		}
	}
	if len(problems) != 0 {
		return fmt.Errorf("%s", strings.Join(problems, "; "))
	}
	return nil
}

func registeredOperationMux(registrars []RouteRegistrar) (mux *http.ServeMux, err error) {
	mux = http.NewServeMux()
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("register Management routes: %v", recovered)
			mux = nil
		}
	}()
	for index, registrar := range registrars {
		if nilRouteRegistrar(registrar) {
			return nil, fmt.Errorf("management route registrar %d is nil", index)
		}
		registrar.Register(mux)
	}
	return mux, nil
}

func registeredOperationPattern(mux *http.ServeMux, operation managementapi.OperationContract) string {
	path := concreteRegistryPath(operation.Path)
	request := httptest.NewRequest(string(operation.Method), "https://management.local"+path, nil)
	_, pattern := mux.Handler(request)
	return pattern
}

func patternOwnsOperation(pattern string, operation managementapi.OperationContract) bool {
	method, path, found := strings.Cut(pattern, " ")
	if !found || method != string(operation.Method) {
		return false
	}
	if path == operation.Path {
		return true
	}
	if strings.Contains(operation.Path, "}:") && path == concreteRegistryPath(operation.Path) {
		return true
	}
	parameter := strings.Index(operation.Path, "/{")
	if parameter < 0 {
		return false
	}
	return path == operation.Path[:parameter+1]
}

func concreteRegistryPath(path string) string {
	return registryPathParameter.ReplaceAllStringFunc(path, func(match string) string {
		name := strings.TrimSuffix(strings.TrimPrefix(match, "{"), "}")
		switch name {
		case "providerId":
			return "openai"
		case "routingRevision":
			return "1"
		default:
			return "10000000-0000-4000-8000-000000000001"
		}
	})
}
