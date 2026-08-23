package managementserver

import (
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

// This list is intentionally independent of AgentRoutes.Register and
// agentHTTPContracts. The Management API package pins the detailed transport
// metadata; this test pins the server-side dispatch surface.
var expectedAgentHandlerRoutes = []agentHTTPContract{
	{managementapi.MethodGET, managementapi.BasePath + "/agent-profiles"},
	{managementapi.MethodPOST, managementapi.BasePath + "/agent-profiles"},
	{managementapi.MethodGET, managementapi.BasePath + "/agent-profiles/{profile}"},
	{managementapi.MethodPATCH, managementapi.BasePath + "/agent-profiles/{profile}"},
	{managementapi.MethodDELETE, managementapi.BasePath + "/agent-profiles/{profile}"},

	{managementapi.MethodGET, managementapi.BasePath + "/agent-skills"},
	{managementapi.MethodPOST, managementapi.BasePath + "/agent-skills"},
	{managementapi.MethodGET, managementapi.BasePath + "/agent-skills/{skill}"},
	{managementapi.MethodPATCH, managementapi.BasePath + "/agent-skills/{skill}"},
	{managementapi.MethodDELETE, managementapi.BasePath + "/agent-skills/{skill}"},

	{managementapi.MethodGET, managementapi.BasePath + "/agent-tools"},

	{managementapi.MethodGET, managementapi.BasePath + "/agent-tool-credentials"},
	{managementapi.MethodPOST, managementapi.BasePath + "/agent-tool-credentials"},
	{managementapi.MethodGET, managementapi.BasePath + "/agent-tool-credentials/{credential}"},
	{managementapi.MethodPATCH, managementapi.BasePath + "/agent-tool-credentials/{credential}"},
	{managementapi.MethodDELETE, managementapi.BasePath + "/agent-tool-credentials/{credential}"},
	{managementapi.MethodPOST, managementapi.BasePath + "/agent-tool-credentials/{credential}:rotate"},

	{managementapi.MethodGET, managementapi.BasePath + "/agent-tool-sources"},
	{managementapi.MethodPOST, managementapi.BasePath + "/agent-tool-sources"},
	{managementapi.MethodGET, managementapi.BasePath + "/agent-tool-sources/{source}"},
	{managementapi.MethodPATCH, managementapi.BasePath + "/agent-tool-sources/{source}"},
	{managementapi.MethodDELETE, managementapi.BasePath + "/agent-tool-sources/{source}"},
	{managementapi.MethodPOST, managementapi.BasePath + "/agent-tool-sources/{source}:test"},
	{managementapi.MethodPOST, managementapi.BasePath + "/agent-tool-sources/{source}:approve"},

	{managementapi.MethodGET, managementapi.BasePath + "/agent-sessions"},
	{managementapi.MethodPOST, managementapi.BasePath + "/agent-sessions"},
	{managementapi.MethodGET, managementapi.BasePath + "/agent-sessions/{session}"},
	{managementapi.MethodPATCH, managementapi.BasePath + "/agent-sessions/{session}"},
	{managementapi.MethodDELETE, managementapi.BasePath + "/agent-sessions/{session}"},
	{managementapi.MethodPOST, managementapi.BasePath + "/agent-sessions/{session}/turns"},
	{managementapi.MethodGET, managementapi.BasePath + "/agent-sessions/{session}/turns"},
	{managementapi.MethodGET, managementapi.BasePath + "/agent-sessions/{session}/events"},
	{managementapi.MethodPOST, managementapi.BasePath + "/agent-sessions/{session}/turns/{turn}:cancel"},

	{managementapi.MethodGET, managementapi.BasePath + "/agent-artifacts/{artifact}"},
	{managementapi.MethodGET, managementapi.BasePath + "/agent-artifacts/{artifact}/content"},

	{managementapi.MethodPOST, managementapi.BasePath + "/publication-plans/{plan}:commit"},
}

func TestAgentHandlerContractTableIsCompleteAndMounted(t *testing.T) {
	if got, want := len(expectedAgentHandlerRoutes), 36; got != want {
		t.Fatalf("expected Agent handler table has %d rows, want %d", got, want)
	}

	expected := agentHandlerRouteSet(t, expectedAgentHandlerRoutes)
	declared := agentHandlerRouteSet(t, agentHTTPContracts())
	assertSameAgentHandlerRoutes(t, "agentHTTPContracts", declared, expected)

	registry := make(map[string]struct{})
	for _, operation := range managementapi.Operations() {
		if strings.HasPrefix(operation.Path, managementapi.BasePath+"/agent-") ||
			operation.Path == managementapi.BasePath+"/publication-plans/{plan}:commit" {
			registry[agentHandlerRouteKey(operation.Method, operation.Path)] = struct{}{}
		}
	}
	assertSameAgentHandlerRoutes(t, "Management operation registry", registry, expected)

	mux := http.NewServeMux()
	(&AgentRoutes{}).Register(mux)
	for _, route := range expectedAgentHandlerRoutes {
		key := agentHandlerRouteKey(route.method, route.path)
		request := httptest.NewRequest(string(route.method), "https://management.local"+concreteRegistryPath(route.path), nil)
		_, pattern := mux.Handler(request)
		if pattern == "" {
			t.Errorf("AgentRoutes.Register does not mount %s", key)
		}
	}
}

func agentHandlerRouteSet(t *testing.T, routes []agentHTTPContract) map[string]struct{} {
	t.Helper()
	result := make(map[string]struct{}, len(routes))
	for _, route := range routes {
		key := agentHandlerRouteKey(route.method, route.path)
		if _, duplicate := result[key]; duplicate {
			t.Fatalf("duplicate Agent handler route %s", key)
		}
		result[key] = struct{}{}
	}
	return result
}

func agentHandlerRouteKey(method managementapi.HTTPMethod, path string) string {
	return string(method) + " " + path
}

func assertSameAgentHandlerRoutes(t *testing.T, source string, got, want map[string]struct{}) {
	t.Helper()
	var missing, extra []string
	for key := range want {
		if _, found := got[key]; !found {
			missing = append(missing, key)
		}
	}
	for key := range got {
		if _, found := want[key]; !found {
			extra = append(extra, key)
		}
	}
	sort.Strings(missing)
	sort.Strings(extra)
	if len(missing) != 0 || len(extra) != 0 {
		t.Errorf("%s route drift: missing %v, extra %v", source, missing, extra)
	}
}
