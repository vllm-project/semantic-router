package modelservice

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector/connectortest"
)

func lastBody(t *testing.T, gateway *connectortest.Gateway, path string) map[string]json.RawMessage {
	t.Helper()
	calls := gateway.Calls()
	for i := len(calls) - 1; i >= 0; i-- {
		if calls[i].Path == path {
			var body map[string]json.RawMessage
			if err := json.Unmarshal(calls[i].Body, &body); err != nil {
				t.Fatal(err)
			}
			return body
		}
	}
	t.Fatalf("no call reached %s: %+v", path, calls)
	return nil
}

func TestSystemOneDeploymentConnectsWithTLSAndAKeyFile(t *testing.T) {
	gateway := connectortest.New(t, map[string]http.HandlerFunc{
		"/systemone": func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(routeAnswer))
		},
	})
	deployment := gateway.Deployment(config.SystemOneProvider, "vllm-sr/d3-ultra")
	deployment.Endpoint += "/systemone"
	deployment.Schema = "vllm-sr/4.0"
	deployment.Capabilities = &config.DeploymentCapabilities{QuestionTypes: []string{"choice"}}
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"remote-d3": deployment}
	remote := newSystemOne(t, cfg)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	response, err := remote.Decide(ctx, "remote-d3", Request{State: "Our app has returned 500 errors since the deploy.", Questions: []Question{routeQuestion}})
	if err != nil {
		t.Fatal(err)
	}
	if answer := response.Answers["route"]; answer.Choice != "engineering" {
		t.Fatalf("answer: %+v", answer)
	}
	if model := string(lastBody(t, gateway, connectortest.Prefix+"/systemone")["model"]); model != `"vllm-sr/d3-ultra"` {
		t.Fatalf("asked as %s", model)
	}

	// A native alias forwards its request as the API's own model.
	result, err := remote.Forward(ctx, "remote-d3", json.RawMessage(`{"model":"native-d3","state":"hello","questions":[{"id":"route","type":"choice","instructions":"Which team?","choices":[{"key":"billing"},{"key":"engineering"}]}]}`))
	if err != nil || result.Status != http.StatusOK {
		t.Fatalf("forward: %+v %v", result, err)
	}
	if model := string(lastBody(t, gateway, connectortest.Prefix+"/systemone")["model"]); model != `"vllm-sr/d3-ultra"` {
		t.Fatalf("forwarded as %s", model)
	}
}

func TestAttachedRuntimeConnectsThroughItsConnectionBlock(t *testing.T) {
	gateway := connectortest.New(t, map[string]http.HandlerFunc{
		"/health": func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"status":"ok"}`))
		},
	})
	deployment := gateway.Deployment(config.ModelRuntimeProvider, "decision-kai")
	deployment.TimeoutMs = 2000
	client, err := NewAttachedClient(deployment.Endpoint, deployment)
	if err != nil {
		t.Fatal(err)
	}
	if ready, status, readyErr := client.Ready(context.Background()); readyErr != nil || !ready {
		t.Fatalf("ready=%v status=%q err=%v", ready, status, readyErr)
	}
	if calls := gateway.Calls(); len(calls) != 1 || calls[0].Path != connectortest.Prefix+"/health" {
		t.Fatalf("calls: %+v", calls)
	}

	// Without its credential the gateway refuses the runtime.
	bare, err := NewAttachedClient(deployment.Endpoint, config.ModelDeployment{TLS: deployment.TLS})
	if err != nil {
		t.Fatal(err)
	}
	if ready, _, _ := bare.Ready(context.Background()); ready {
		t.Fatal("a runtime behind the gateway answered without its credential")
	}
}

func TestAttachedPlansKeepTheirConnection(t *testing.T) {
	prefix := ""
	auth := &config.DeploymentAuth{Header: "X-Runtime-Key", Prefix: &prefix, APIKeyEnv: "RUNTIME_KEY"}
	deployments := map[string]config.ModelDeployment{
		"one": {Provider: config.ModelRuntimeProvider, Endpoint: "https://runtime.example.com:8100", Auth: auth, TimeoutMs: 1500},
		"two": {Provider: config.ModelRuntimeProvider, Endpoint: "https://runtime.example.com:8100"},
	}
	plans := planProcesses(deployments, []string{"vllm-srun"}, "", 4, "cpu")
	keys := map[string]string{}
	for _, plan := range plans {
		keys[plan.logical] = plan.attachedClientKey()
		if plan.logical == "one" && (plan.connection.Auth != auth || plan.connection.TimeoutMs != 1500) {
			t.Fatalf("plan lost its connection: %+v", plan.connection)
		}
	}
	if keys["one"] == keys["two"] {
		t.Fatal("deployments with different credentials share one client")
	}
}
