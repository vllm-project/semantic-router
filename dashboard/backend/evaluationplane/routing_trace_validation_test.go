package evaluationplane

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

func TestValidateRoutingTraceArtifactAcceptsStrictBoundedCaseJoinedRows(t *testing.T) {
	runDir := t.TempDir()
	writeRoutingTraceRows(t, runDir, routingTraceRow("case-1"))
	if err := validateRoutingTraceArtifact(runDir, map[string]struct{}{"case-1": {}}); err != nil {
		t.Fatalf("validateRoutingTraceArtifact: %v", err)
	}
}

func TestValidateRoutingTraceArtifactEnforcesGlobalNodeAndByteBudgets(t *testing.T) {
	t.Run("maximum global node budget", func(t *testing.T) {
		runDir := t.TempDir()
		row := routingTraceRowWithNodes("case-1", maxRoutingTraceNodes)
		writeRoutingTraceRows(t, runDir, row)
		if err := validateRoutingTraceArtifact(runDir, map[string]struct{}{"case-1": {}}); err != nil {
			t.Fatalf("maximum bounded worker trace rejected: %v", err)
		}
	})

	t.Run("truncated flag cannot bypass global node budget", func(t *testing.T) {
		runDir := t.TempDir()
		row := routingTraceRowWithNodes("case-1", maxRoutingTraceNodes+1)
		row["truncated"] = true
		writeRoutingTraceRows(t, runDir, row)
		err := validateRoutingTraceArtifact(runDir, map[string]struct{}{"case-1": {}})
		if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "global node budget") {
			t.Fatalf("oversized trace error=%v, want global-node ErrInvalid", err)
		}
	})

	t.Run("serialized line byte budget", func(t *testing.T) {
		runDir := t.TempDir()
		row := routingTraceRow("case-1")
		row["recipe"] = strings.Repeat("a", maxRoutingTraceLineBytes)
		writeRoutingTraceRows(t, runDir, row)
		if err := validateRoutingTraceArtifact(runDir, map[string]struct{}{"case-1": {}}); !errors.Is(err, ErrInvalid) {
			t.Fatalf("oversized serialized line error=%v, want ErrInvalid", err)
		}
	})
}

func assertAuthenticatedRoutingPublication(
	t *testing.T, service *Service, root string, run Run, routerCredential string,
	authenticatedRequests *atomic.Int64, diagnostics *bytes.Buffer,
) {
	t.Helper()
	if got := authenticatedRequests.Load(); got != int64(run.SampleLimit) {
		t.Fatalf("authenticated Router requests = %d, want %d", got, run.SampleLimit)
	}
	traceBytes, err := os.ReadFile(filepath.Join(root, "runs", run.ID, "routing-traces.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(traceBytes), `"truncated":false`) {
		t.Fatalf("real worker trace omitted its explicit truncation status: %s", traceBytes)
	}
	reportBytes, err := service.ReportJSONAs(SystemActor(), run.ID)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(reportBytes, []byte("routing-traces.jsonl")) {
		t.Fatal("public report exposed its private routing trace artifact")
	}
	for label, payload := range map[string][]byte{
		"worker diagnostics": diagnostics.Bytes(),
		"routing traces":     traceBytes,
		"public report":      reportBytes,
	} {
		if bytes.Contains(payload, []byte(routerCredential)) {
			t.Fatalf("%s leaked the Router evaluation credential", label)
		}
	}
}

func TestRealWorkerAuthenticatedRoutingUsesServerBrokeredCredential(t *testing.T) {
	python := os.Getenv("VLLM_SR_EVALUATION_TEST_PYTHON")
	if python == "" {
		t.Skip("set VLLM_SR_EVALUATION_TEST_PYTHON to run the real Python worker")
	}
	pythonRoot, err := filepath.Abs("../../../src/vllm-sr")
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("PYTHONPATH", pythonRoot)
	t.Setenv("TMPDIR", "/tmp")
	const routerAccessEnv = "ROUTER_EVAL_TOKEN"
	const routerAccessValue = "server-owned-router-evaluation-secret"
	t.Setenv(routerAccessEnv, routerAccessValue)
	var authenticatedRouterRequests atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		switch request.URL.Path {
		case "/v1/models":
			_, _ = writer.Write([]byte(`{"data":[{"id":"entrypoint-a","routing":{"resolution":"virtual","selectable":true,"default_route":true,"recipe":"default"}}]}`))
		case "/api/v1/eval":
			if request.Header.Get("Authorization") == "Bearer "+routerAccessValue {
				authenticatedRouterRequests.Add(1)
			}
			_, _ = writer.Write([]byte(`{"recipe":"default","decision_result":{"decision_name":"route","algorithm":"static","plugins":["audit"]},"recommended_models":["Org/Fast Model"],"selected_model":"Org/Fast Model","selection_status":"selected","selection_method":"static","eval_trace":[{"decision_name":"route","state":"matched","matched":true,"confidence":0.9,"on_unknown":"no_match","root_trace":{"node_type":"leaf","state":"matched","matched":true,"confidence":0.9,"children":[]}}],"signal_confidences":{"domain:reasoning":0.9},"applied_unknown_policies":{"domain:reasoning":"no_match"}}`))
		default:
			http.NotFound(writer, request)
		}
	}))
	t.Cleanup(server.Close)

	root := filepath.Join(t.TempDir(), "evaluation")
	if mkdirErr := os.Mkdir(root, 0o700); mkdirErr != nil {
		t.Fatal(mkdirErr)
	}
	configPath := filepath.Join(root, "config.yaml")
	if writeErr := os.WriteFile(configPath, []byte(modelArmTestYAML), 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	service, err := NewService(Options{
		DataDir: root, PythonPath: python, ConfigPath: configPath,
		RouterAPIURL: server.URL, EnvoyURL: server.URL,
		RouterAPIKeyEnv: routerAccessEnv,
		CredentialProvider: staticCredentialProvider{
			token: "distinct-dashboard-management-secret",
		},
		CodeRevision: testSourceRevision, MaxConcurrent: 1,
	})
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}
	var diagnostics bytes.Buffer
	service.process.(*CommandProcess).diagnosticSink = &diagnostics
	t.Cleanup(func() { _ = service.Close() })
	run, err := service.CreateRunAs(context.Background(), SystemActor(), CreateRunRequest{
		ClientRequestID: newTestClientRequestID(),
		Name:            "real routing trace", SuiteIDs: []string{"live-mom-core"},
		TrackIDs: []TrackID{"routing"}, Mode: ModeLive, TargetID: mixtureTargetID("default"),
		ChangeProfile: "recipe", SampleLimit: 4, Concurrency: 1, Seed: 17,
	})
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	manifest, manifestBytes, err := readRunManifestStrict(filepath.Join(root, "runs", run.ID, manifestFileName))
	if err != nil {
		t.Fatal(err)
	}
	if manifest.Target.RouterAPIKey == nil || manifest.Target.RouterAPIKey.SchemaVersion != SchemaVersion ||
		manifest.Target.RouterAPIKey.Env != routerAccessEnv || bytes.Contains(manifestBytes, []byte(routerAccessValue)) {
		t.Fatalf("authenticated manifest omitted its SecretRef or leaked the credential: %s", manifestBytes)
	}
	if _, startErr := service.StartRunAs(context.Background(), SystemActor(), run.ID); startErr != nil {
		t.Fatalf("StartRun: %v", startErr)
	}
	completed := waitForRunStatus(t, service, run.ID, StatusCompleted)
	if completed.Error != "" {
		t.Fatalf("live routing lifecycle failed: %s; diagnostics=%s", completed.Error, diagnostics.String())
	}
	assertAuthenticatedRoutingPublication(
		t, service, root, run, routerAccessValue, &authenticatedRouterRequests, &diagnostics,
	)
}

func TestValidateRoutingTraceArtifactRejectsUntrustedOrUnboundedRows(t *testing.T) {
	tests := []struct {
		name  string
		rows  []map[string]any
		cases map[string]struct{}
		match string
	}{
		{
			name: "unknown prompt field",
			rows: []map[string]any{func() map[string]any {
				row := routingTraceRow("case-1")
				row["prompt"] = "must remain private"
				return row
			}()},
			cases: map[string]struct{}{"case-1": {}}, match: "unknown field",
		},
		{
			name:  "case is not joined",
			rows:  []map[string]any{routingTraceRow("other-case")},
			cases: map[string]struct{}{"case-1": {}}, match: "absent from the validated case set",
		},
		{
			name:  "duplicate case",
			rows:  []map[string]any{routingTraceRow("case-1"), routingTraceRow("case-1")},
			cases: map[string]struct{}{"case-1": {}, "case-2": {}}, match: "duplicate case_id",
		},
		{
			name: "null collection",
			rows: []map[string]any{func() map[string]any {
				row := routingTraceRow("case-1")
				row["signals"] = nil
				return row
			}()},
			cases: map[string]struct{}{"case-1": {}}, match: "collections cannot be null",
		},
		{
			name: "token collection exceeds bound",
			rows: []map[string]any{func() map[string]any {
				row := routingTraceRow("case-1")
				row["plugins"] = make([]string, maxRoutingTraceTokens+1)
				for index := range row["plugins"].([]string) {
					row["plugins"].([]string)[index] = "plugin"
				}
				return row
			}()},
			cases: map[string]struct{}{"case-1": {}}, match: "cardinality limit",
		},
		{
			name: "malformed applied unknown policy",
			rows: []map[string]any{func() map[string]any {
				row := routingTraceRow("case-1")
				row["applied_unknown_policies"] = []any{[]string{"signal-only"}}
				return row
			}()},
			cases: map[string]struct{}{"case-1": {}}, match: "must contain a key and policy",
		},
		{
			name:  "trace tree exceeds depth",
			rows:  []map[string]any{routingTraceRowWithDepth("case-1", maxRoutingTraceDepth+1)},
			cases: map[string]struct{}{"case-1": {}}, match: "depth limit",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			runDir := t.TempDir()
			writeRoutingTraceRows(t, runDir, test.rows...)
			if err := validateRoutingTraceArtifact(runDir, test.cases); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("validateRoutingTraceArtifact error=%v, want ErrInvalid containing %q", err, test.match)
			}
		})
	}
}

func routingTraceRow(caseID string) map[string]any {
	return map[string]any{
		"schema_version":           SchemaVersion,
		"case_id":                  caseID,
		"truncated":                false,
		"recipe":                   "default recipe",
		"plugins":                  []string{},
		"recommended_models":       []string{},
		"traces":                   []any{},
		"signals":                  []any{},
		"applied_unknown_policies": []any{},
	}
}

func routingTraceRowWithNodes(caseID string, count int) map[string]any {
	row := routingTraceRow(caseID)
	root := map[string]any{
		"node_type": "signal", "matched": true, "confidence_scored": false, "children": []any{},
	}
	queue := []map[string]any{root}
	remaining := count - 1
	for len(queue) > 0 && remaining > 0 {
		parent := queue[0]
		queue = queue[1:]
		childCount := maxRoutingTraceChildren
		if remaining < childCount {
			childCount = remaining
		}
		children := make([]any, 0, childCount)
		for range childCount {
			child := map[string]any{
				"node_type": "signal", "matched": true, "confidence_scored": false, "children": []any{},
			}
			children = append(children, child)
			queue = append(queue, child)
		}
		parent["children"] = children
		remaining -= childCount
	}
	row["traces"] = []any{map[string]any{
		"decision_name": "route", "matched": true, "root_trace": root,
	}}
	return row
}

func routingTraceRowWithDepth(caseID string, depth int) map[string]any {
	row := routingTraceRow(caseID)
	var node map[string]any
	for index := 0; index < depth; index++ {
		children := []any{}
		current := map[string]any{
			"node_type": "signal", "matched": true, "confidence_scored": false, "children": children,
		}
		if node != nil {
			current["children"] = []any{node}
		}
		node = current
	}
	row["traces"] = []any{map[string]any{
		"decision_name": "route", "matched": true, "root_trace": node,
	}}
	return row
}

func writeRoutingTraceRows(t *testing.T, runDir string, rows ...map[string]any) {
	t.Helper()
	var encoded strings.Builder
	for _, row := range rows {
		data, err := json.Marshal(row)
		if err != nil {
			t.Fatalf("marshal routing trace: %v", err)
		}
		encoded.Write(data)
		encoded.WriteByte('\n')
	}
	if err := os.WriteFile(filepath.Join(runDir, "routing-traces.jsonl"), []byte(encoded.String()), 0o600); err != nil {
		t.Fatalf("write routing traces: %v", err)
	}
}
