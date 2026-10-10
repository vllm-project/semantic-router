//go:build !windows

package apiserver

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

func TestSystemOneHTTPUsesPublishedDeploymentAndNativeTypes(t *testing.T) {
	fake := runtimetest.New(runtimetest.Model{ID: "served"})
	var forwarded string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/systemone" {
			body, _ := io.ReadAll(r.Body)
			forwarded = string(body)
			w.Header().Set("Server-Timing", "total;dur=2")
			_, _ = io.WriteString(w, `{"model":"served","answers":{"choice":{"type":"choice","choice":"coding","probabilities":{"coding":0.9,"other":0.1},"confidence":0.9},"score":{"type":"score","score":0.75,"probabilities":{"0":0.25,"1":0.75},"confidence":0.75,"legend":{"0":"easy","1":"hard"}},"noul":{"type":"noul","noul":0.9}},"sets":{"set":{"selected":["code"],"probabilities":{"code":0.9}}},"spans":{"span":[{"start":0,"end":2,"text":"张三","label":"person","probability":0.9}]},"usage":{"input_tokens":12,"output_tokens":0}}`)
			return
		}
		fake.Handler().ServeHTTP(w, r)
	}))
	defer upstream.Close()
	manager := modelservice.NewManager()
	defer func() { _ = manager.Shutdown(context.Background()) }()
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"selected": {Provider: config.ModelRuntimeProvider, Endpoint: upstream.URL, ServedName: "served"}}
	cfg.DecisionRules = []config.DecisionSignalRule{{Name: "probe", Deployment: "selected", Question: config.DecisionQuestion{Type: "noul", Instructions: "test?"}}}
	if err := manager.Reconcile(cfg); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if _, err := manager.Published().Card(ctx, "selected"); err != nil {
		t.Fatal(err)
	}
	previous := modelservice.DefaultManager()
	modelservice.SetDefault(manager)
	defer modelservice.SetDefault(previous)
	server := httptest.NewServer((&ClassificationAPIServer{config: cfg}).setupRoutes())
	defer server.Close()
	response, err := server.Client().Get(server.URL + systemOneDiagnosticPath)
	if err != nil {
		t.Fatal(err)
	}
	var capabilities SystemOneCapabilities
	_ = json.NewDecoder(response.Body).Decode(&capabilities)
	response.Body.Close()
	if response.StatusCode != 200 || len(capabilities.Deployments) != 1 || capabilities.Deployments[0].ID != "selected" || len(capabilities.Deployments[0].QuestionTypes) != 3 {
		t.Fatalf("capabilities=%+v status=%d", capabilities, response.StatusCode)
	}
	body := `{"deployment":"selected","request":{"model":"unselected","state":"张三 needs code","questions":{"choice":{"type":"choice","instructions":"Task?","criteria":{"coding":"write code","other":"other"}},"score":{"type":"score","instructions":"Difficulty?","criteria":["easy","hard"]},"noul":{"type":"noul","instructions":"Needs facts?"},"set":{"type":"set","instructions":"Needs?","criteria":{"code":"code"}},"span":{"type":"span","instructions":"People?","criteria":{"person":"person"}}},"options":{"return_meta":true}}}`
	status, data := diagnosticRequest(t, server, "systemone", body)
	if status != 200 || !strings.Contains(string(data), `"spans"`) || !strings.Contains(string(data), `"sets"`) || !strings.Contains(forwarded, `"model":"served"`) || strings.Contains(forwarded, "unselected") {
		t.Fatalf("status=%d body=%s forwarded=%s", status, data, forwarded)
	}
	for _, tc := range []struct {
		body   string
		status int
	}{
		{`{"deployment":"unknown","request":{}}`, 404},
		{`{"deployment":"selected","expected_artifact":"different/new-model","request":{}}`, 409},
		{`{"deployment":"selected","request":[],"url":"http://untrusted"}`, 400},
		{`{"deployment":"selected","request":[]}`, 400},
		{`{"deployment":"selected","request":` + strings.Repeat(" ", systemOneRequestLimit) + `{}}`, 413},
	} {
		status, data := diagnosticRequest(t, server, "systemone", tc.body)
		if status != tc.status {
			t.Fatalf("status=%d want=%d body=%s", status, tc.status, data)
		}
	}
}

func TestSystemOneHTTPRequiresDistinctReadAndInvokePermissions(t *testing.T) {
	t.Setenv("SYSTEMONE_INSPECTOR_TOKEN", "test-inspector")
	server := testManagementAPIServer(t, config.ManagementAPIConfig{Auth: config.ManagementAPIAuthConfig{Mode: config.ManagementAuthModeBearer, Tokens: []config.ManagementAPITokenRef{{Env: "SYSTEMONE_INSPECTOR_TOKEN", Role: "inspector"}}, Roles: map[string][]string{"inspector": {"config.read"}}}})
	mux := server.setupRoutes()
	for _, tc := range []struct {
		method, token string
		status        int
	}{
		{http.MethodGet, "", 401},
		{http.MethodPost, "", 401},
		{http.MethodGet, "test-inspector", 200},
		{http.MethodPost, "test-inspector", 403},
	} {
		request := httptest.NewRequest(tc.method, systemOneDiagnosticPath, strings.NewReader(`{"deployment":"selected","request":{}}`))
		if tc.token != "" {
			request.Header.Set("Authorization", "Bearer "+tc.token)
		}
		response := httptest.NewRecorder()
		mux.ServeHTTP(response, request)
		if response.Code != tc.status {
			t.Fatalf("%s token=%t: got %d want %d: %s", tc.method, tc.token != "", response.Code, tc.status, response.Body.String())
		}
	}
}
