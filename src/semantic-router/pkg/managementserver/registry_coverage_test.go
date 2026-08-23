package managementserver

import (
	"math/rand"
	"net/http"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

type operationCoverageRegistrar struct {
	omit   map[string]bool
	status int
}

type operationCoverageAggregate []RouteRegistrar

type overbroadOperationCoverageRegistrar struct{}

func (registrars operationCoverageAggregate) Register(mux *http.ServeMux) {
	for _, registrar := range registrars {
		registrar.Register(mux)
	}
}

func (overbroadOperationCoverageRegistrar) Register(mux *http.ServeMux) {
	for _, method := range []managementapi.HTTPMethod{
		managementapi.MethodGET,
		managementapi.MethodPOST,
		managementapi.MethodPUT,
		managementapi.MethodPATCH,
		managementapi.MethodDELETE,
	} {
		mux.HandleFunc(string(method)+" "+managementapi.BasePath+"/", func(http.ResponseWriter, *http.Request) {})
	}
}

func (registrar operationCoverageRegistrar) Register(mux *http.ServeMux) {
	for _, operation := range managementapi.Operations() {
		if registrar.omit[operation.OperationID] {
			continue
		}
		path := operation.Path
		if strings.Contains(path, "}:") {
			path = concreteRegistryPath(path)
		}
		mux.HandleFunc(string(operation.Method)+" "+path, func(response http.ResponseWriter, _ *http.Request) {
			status := registrar.status
			if status == 0 {
				status = http.StatusUnauthorized
			}
			response.WriteHeader(status)
		})
	}
}

func TestValidateRegisteredOperationsRequiresEveryRegistryRoute(t *testing.T) {
	if err := ValidateRegisteredOperations(operationCoverageRegistrar{omit: map[string]bool{}}); err != nil {
		t.Fatal(err)
	}
	const missing = "postAuthBootstrap"
	if err := ValidateRegisteredOperations(operationCoverageRegistrar{omit: map[string]bool{missing: true}}); err == nil {
		t.Fatalf("coverage accepted missing %s", missing)
	}
	if err := ValidateRegisteredOperations(operationCoverageRegistrar{omit: map[string]bool{missing: true}}, missing); err != nil {
		t.Fatalf("explicitly disabled operation was not accepted: %v", err)
	}
}

func TestValidateRegisteredOperationsAcceptsDomainNotFoundResponses(t *testing.T) {
	registrar := operationCoverageRegistrar{omit: map[string]bool{}, status: http.StatusNotFound}
	if err := ValidateRegisteredOperations(registrar); err != nil {
		t.Fatalf("coverage treated a handler response as a missing route: %v", err)
	}
}

func TestValidateRegisteredOperationsRejectsUnregisteredExclusion(t *testing.T) {
	if err := ValidateRegisteredOperations(operationCoverageRegistrar{omit: map[string]bool{}}, "notAnOperation"); err == nil {
		t.Fatal("coverage accepted an unknown disabled operation")
	}
}

func TestValidateRegisteredOperationsRejectsOverbroadCoverage(t *testing.T) {
	err := ValidateRegisteredOperations(overbroadOperationCoverageRegistrar{})
	if err == nil || !strings.Contains(err.Error(), "over-broad pattern") {
		t.Fatalf("over-broad registrar error = %v", err)
	}
}

func TestValidateRegisteredOperationsRejectsDuplicateOwners(t *testing.T) {
	registrar := operationCoverageRegistrar{omit: map[string]bool{}}
	server, err := NewServer(&catalogRuntimeStub{}, registrar, registrar)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateRegisteredOperations(server); err == nil || !strings.Contains(err.Error(), "conflicts with pattern") {
		t.Fatalf("duplicate registrar error = %v", err)
	}
}

func TestRouteCoverageRejectsTypedNilRegistrar(t *testing.T) {
	var registrar *operationCoverageRegistrar
	if _, err := NewServer(&catalogRuntimeStub{}, registrar); err == nil {
		t.Fatal("NewServer accepted a typed-nil route registrar")
	}
	if err := ValidateRegisteredOperations(registrar); err == nil {
		t.Fatal("coverage accepted a typed-nil route registrar")
	}
}

func TestValidateRegisteredOperationsChecksServerDomainMux(t *testing.T) {
	const disabled = "postAuthRecovery"
	server, err := NewServer(
		&catalogRuntimeStub{},
		operationCoverageRegistrar{omit: map[string]bool{disabled: true}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateRegisteredOperations(server, disabled); err != nil {
		t.Fatalf("disabled domain operation was obscured by the transport mount: %v", err)
	}
}

func TestRegisteredOperationPatternMatchesIdentityResourcePrefix(t *testing.T) {
	mux := http.NewServeMux()
	(&IdentityResourceRoutes{}).Register(mux)
	operation, found := managementapi.LookupOperation(
		managementapi.MethodDELETE,
		managementapi.BasePath+"/management-roles/{roleId}",
	)
	if !found {
		t.Fatal("DELETE Management Role operation is absent from the registry")
	}
	if pattern := registeredOperationPattern(mux, operation); pattern != "DELETE "+rolePath+"/" {
		t.Fatalf("registered pattern = %q, want %q", pattern, "DELETE "+rolePath+"/")
	}
}

func TestProductionRouteShapeCoversRegistryAcrossRegistrarPermutations(t *testing.T) {
	random := rand.New(rand.NewSource(1))
	for iteration := 0; iteration < 128; iteration++ {
		identityRoutes := []RouteRegistrar{
			&IdentityAuthRoutes{},
			&IdentityResourceRoutes{},
			&IdentityLifecycleRoutes{},
			&WorkloadIdentityRoutes{},
		}
		random.Shuffle(len(identityRoutes), func(left, right int) {
			identityRoutes[left], identityRoutes[right] = identityRoutes[right], identityRoutes[left]
		})
		registrars := productionOperationRegistrars(operationCoverageAggregate(identityRoutes))
		random.Shuffle(len(registrars), func(left, right int) {
			registrars[left], registrars[right] = registrars[right], registrars[left]
		})
		server, testProductionRouteShapeCoversRegistryAcrossRegistrarPermutationsErr := NewServer(&catalogRuntimeStub{}, registrars...)
		if testProductionRouteShapeCoversRegistryAcrossRegistrarPermutationsErr != nil {
			t.Fatalf("iteration %d: %v", iteration, testProductionRouteShapeCoversRegistryAcrossRegistrarPermutationsErr)
		}
		if err := ValidateRegisteredOperations(server, "postAuthRecovery"); err != nil {
			t.Fatalf("iteration %d: %v", iteration, err)
		}
		mux, testProductionRouteShapeCoversRegistryAcrossRegistrarPermutationsErr := registeredOperationMux(server.routes)
		if testProductionRouteShapeCoversRegistryAcrossRegistrarPermutationsErr != nil {
			t.Fatalf("iteration %d: %v", iteration, testProductionRouteShapeCoversRegistryAcrossRegistrarPermutationsErr)
		}
		for _, operation := range managementapi.Operations() {
			pattern := registeredOperationPattern(mux, operation)
			if operation.OperationID == "postAuthRecovery" {
				if pattern != "" {
					t.Fatalf("iteration %d: disabled recovery matched %q", iteration, pattern)
				}
				continue
			}
			if !patternOwnsOperation(pattern, operation) {
				t.Fatalf("iteration %d: %s %s matched %q", iteration, operation.Method, operation.Path, pattern)
			}
		}
	}
}

func productionOperationRegistrars(identity RouteRegistrar) []RouteRegistrar {
	return []RouteRegistrar{
		&ProviderRoutes{},
		&ProviderCatalogAdministrationRoutes{},
		identity,
		&NamespaceRoutes{},
		&SubjectRoutes{},
		&APIKeyRoutes{},
		&DelegationRoutes{},
		&InvitationRoutes{},
		&ProviderCredentialRoutes{},
		&RoutingRoutes{},
		&PolicyRoutes{},
		&AccessReadRoutes{},
		&OperationRoutes{},
		&UnknownUsageRoutes{},
		&ObservabilityRoutes{},
		&StatisticsRoutes{},
		&RuntimeDiagnosticsRoutes{},
		&AgentRoutes{},
	}
}
