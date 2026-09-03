package evaluationplane

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func newLedgerContractBroker(t *testing.T) (*workerHTTPBroker, *atomic.Int64) {
	t.Helper()
	t.Setenv("TEST_AGENT_TASK_LEDGER_KEY", "agent-task-secret")
	t.Setenv("TEST_FAULT_RECOVERY_LEDGER_KEY", "fault-recovery-secret")
	t.Setenv("TEST_HARD_POLICY_LEDGER_KEY", "hard-policy-secret")
	t.Setenv("TEST_PRODUCTION_LEDGER_KEY", "production-secret")
	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		calls.Add(1)
		body, err := io.ReadAll(request.Body)
		if err != nil || request.Method != http.MethodGet || len(body) != 0 {
			t.Errorf("ledger request method=%s body=%q error=%v", request.Method, body, err)
		}
		writer.Header().Set("Content-Type", "application/json")
		switch request.URL.RequestURI() {
		case "/sealed-agent-tasks":
			if authorization := request.Header.Get("Authorization"); authorization != "Bearer agent-task-secret" {
				t.Errorf("agent-task authorization = %q", authorization)
			}
			_, _ = writer.Write([]byte(`{"attempts":[]}`))
		case "/sealed-fault-recovery":
			if authorization := request.Header.Get("Authorization"); authorization != "Bearer fault-recovery-secret" {
				t.Errorf("fault-recovery authorization = %q", authorization)
			}
			_, _ = writer.Write([]byte(`{"pairs":[]}`))
		case "/sealed-hard-policy":
			if authorization := request.Header.Get("Authorization"); authorization != "Bearer hard-policy-secret" {
				t.Errorf("hard-policy authorization = %q", authorization)
			}
			_, _ = writer.Write([]byte(`{"observations":[]}`))
		case "/sealed-production-experiment":
			if authorization := request.Header.Get("Authorization"); authorization != "Bearer production-secret" {
				t.Errorf("production authorization = %q", authorization)
			}
			_, _ = writer.Write([]byte(`{"assignments":[],"preference_outcomes":[]}`))
		default:
			http.NotFound(writer, request)
		}
	}))
	t.Cleanup(server.Close)
	manifest := RunManifest{Concurrency: 1, SampleLimit: 2, Target: ManifestTarget{
		AgentTaskLedger: &ServiceEndpoint{
			SchemaVersion: SchemaVersion, URL: server.URL + "/sealed-agent-tasks",
			APIKey: &SecretRef{SchemaVersion: SchemaVersion, Env: "TEST_AGENT_TASK_LEDGER_KEY"}, TimeoutSeconds: 2,
		},
		FaultRecoveryLedger: &ServiceEndpoint{
			SchemaVersion: SchemaVersion, URL: server.URL + "/sealed-fault-recovery",
			APIKey: &SecretRef{SchemaVersion: SchemaVersion, Env: "TEST_FAULT_RECOVERY_LEDGER_KEY"}, TimeoutSeconds: 2,
		},
		HardPolicyLedger: &ServiceEndpoint{
			SchemaVersion: SchemaVersion, URL: server.URL + "/sealed-hard-policy",
			APIKey: &SecretRef{SchemaVersion: SchemaVersion, Env: "TEST_HARD_POLICY_LEDGER_KEY"}, TimeoutSeconds: 2,
		},
		ProductionExperimentLedger: &ServiceEndpoint{
			SchemaVersion: SchemaVersion, URL: server.URL + "/sealed-production-experiment",
			APIKey: &SecretRef{SchemaVersion: SchemaVersion, Env: "TEST_PRODUCTION_LEDGER_KEY"}, TimeoutSeconds: 2,
		},
	}}
	credentials, err := (&CommandProcess{}).brokerCredentials(manifest)
	if err != nil {
		t.Fatalf("resolve server-owned ledger credentials: %v", err)
	}
	broker := newWorkerHTTPBroker(manifest, credentials)
	assertLedgerOperation := func(name string, operation workerBrokerOperation, endpoint *ServiceEndpoint, credential string) {
		t.Helper()
		if operation.method != http.MethodGet || operation.url != endpoint.URL ||
			operation.credential != credential || operation.maxTimeoutMS != 2_000 {
			t.Fatalf("%s broker contract = %+v", name, operation)
		}
	}
	assertLedgerOperation("agent-task", broker.operations[workerBrokerAgentTaskLedger], manifest.Target.AgentTaskLedger, "agent-task-secret")
	assertLedgerOperation("fault-recovery", broker.operations[workerBrokerFaultRecoveryLedger], manifest.Target.FaultRecoveryLedger, "fault-recovery-secret")
	assertLedgerOperation("hard-policy", broker.operations[workerBrokerHardPolicyLedger], manifest.Target.HardPolicyLedger, "hard-policy-secret")
	assertLedgerOperation("production", broker.operations[workerBrokerProductionExperimentLedger], manifest.Target.ProductionExperimentLedger, "production-secret")
	return broker, &calls
}

func TestWorkerHTTPBrokerLedgerEndpointsAreExactServerOwnedContracts(t *testing.T) {
	broker, calls := newLedgerContractBroker(t)

	requests := []workerBrokerRequest{
		{ID: 1, Operation: workerBrokerAgentTaskLedger, TrackID: "agentic", CaseID: "agent-task-ledger", AttemptID: "ledger-fetch", Payload: json.RawMessage("null"), TimeoutMS: 2_000},
		{ID: 2, Operation: workerBrokerFaultRecoveryLedger, TrackID: "agentic", CaseID: "fault-recovery-ledger", AttemptID: "ledger-fetch", Payload: json.RawMessage("null"), TimeoutMS: 2_000},
		{ID: 3, Operation: workerBrokerHardPolicyLedger, TrackID: "safety", CaseID: "hard-policy-ledger", AttemptID: "ledger-fetch", Payload: json.RawMessage("null"), TimeoutMS: 2_000},
		{ID: 4, Operation: workerBrokerProductionExperimentLedger, TrackID: "preference", CaseID: "production-ledger", AttemptID: "ledger-fetch", Payload: json.RawMessage("null"), TimeoutMS: 2_000},
	}
	var previousRequestID uint64
	for _, request := range requests {
		encoded, marshalErr := json.Marshal(request)
		if marshalErr != nil {
			t.Fatal(marshalErr)
		}
		validated, decodeErr := decodeWorkerBrokerRequest(encoded, previousRequestID, broker.operations)
		if decodeErr != nil {
			t.Fatalf("decode ledger request %d: %v", request.ID, decodeErr)
		}
		response := broker.execute(context.Background(), validated)
		if !response.Success || !digestPattern.MatchString(response.BrokerReceipt) {
			t.Fatalf("ledger response %d = %+v", request.ID, response)
		}
		previousRequestID = request.ID
	}
	if calls.Load() != 4 {
		t.Fatalf("ledger calls = %d, want 4", calls.Load())
	}
	if broker.entries[1].responsePayload["attempts"] == nil || broker.entries[2].responsePayload["pairs"] == nil || broker.entries[3].responsePayload["observations"] == nil || broker.entries[4].responsePayload["assignments"] == nil {
		t.Fatalf("sealed ledger payloads were not retained in the broker journal: %+v", broker.entries)
	}

	for name, invalid := range map[string]workerBrokerRequest{
		"agent task wrong track": {
			ID: 1, Operation: workerBrokerAgentTaskLedger, TrackID: "safety", CaseID: "case-1", AttemptID: "attempt-1", Payload: json.RawMessage("null"), TimeoutMS: 1_000,
		},
		"agent task wrong case": {
			ID: 1, Operation: workerBrokerAgentTaskLedger, TrackID: "agentic", CaseID: "case-1", AttemptID: "ledger-fetch", Payload: json.RawMessage("null"), TimeoutMS: 1_000,
		},
		"fault recovery wrong track": {
			ID: 1, Operation: workerBrokerFaultRecoveryLedger, TrackID: "safety", CaseID: "case-1", AttemptID: "attempt-1", Payload: json.RawMessage("null"), TimeoutMS: 1_000,
		},
		"fault recovery wrong attempt": {
			ID: 1, Operation: workerBrokerFaultRecoveryLedger, TrackID: "agentic", CaseID: "fault-recovery-ledger", AttemptID: "attempt-1", Payload: json.RawMessage("null"), TimeoutMS: 1_000,
		},
		"hard policy wrong track": {
			ID: 1, Operation: workerBrokerHardPolicyLedger, TrackID: "preference", CaseID: "case-1", AttemptID: "attempt-1", Payload: json.RawMessage("null"), TimeoutMS: 1_000,
		},
		"production payload": {
			ID: 1, Operation: workerBrokerProductionExperimentLedger, TrackID: "preference", CaseID: "case-1", AttemptID: "attempt-1", Payload: json.RawMessage(`{}`), TimeoutMS: 1_000,
		},
		"endpoint timeout": {
			ID: 1, Operation: workerBrokerHardPolicyLedger, TrackID: "safety", CaseID: "case-1", AttemptID: "attempt-1", Payload: json.RawMessage("null"), TimeoutMS: 2_001,
		},
	} {
		encoded, marshalErr := json.Marshal(invalid)
		if marshalErr != nil {
			t.Fatal(marshalErr)
		}
		if _, decodeErr := decodeWorkerBrokerRequest(encoded, 0, broker.operations); decodeErr == nil {
			t.Fatalf("%s ledger request was accepted", name)
		}
	}

	withoutLedgers := newWorkerHTTPBroker(RunManifest{Concurrency: 1}, workerBrokerCredentials{})
	if _, present := withoutLedgers.operations[workerBrokerAgentTaskLedger]; present {
		t.Fatal("default runtime advertised an agent-task ledger operation")
	}
	if _, present := withoutLedgers.operations[workerBrokerFaultRecoveryLedger]; present {
		t.Fatal("default runtime advertised a fault-recovery ledger operation")
	}
	if _, present := withoutLedgers.operations[workerBrokerHardPolicyLedger]; present {
		t.Fatal("default runtime advertised a hard-policy ledger operation")
	}
	if _, present := withoutLedgers.operations[workerBrokerProductionExperimentLedger]; present {
		t.Fatal("default runtime advertised a production experiment ledger operation")
	}
}

func TestWorkerHTTPBrokerRejectsDuplicateRequestKeys(t *testing.T) {
	broker := newWorkerHTTPBroker(RunManifest{
		Concurrency: 1,
		Target:      ManifestTarget{RouterAPIURL: "http://router.invalid"},
	}, workerBrokerCredentials{})
	frame := []byte(`{"id":1,"id":1,"operation":"router.evaluate","track_id":"routing","case_id":"case-1","attempt_id":"attempt-1","payload":{},"timeout_ms":1000}`)
	if _, err := decodeWorkerBrokerRequest(frame, 0, broker.operations); err == nil {
		t.Fatal("broker request with duplicate keys was accepted")
	}
}

func TestWorkerHTTPBrokerRejectsDuplicateOperationPayloadKeys(t *testing.T) {
	broker := newWorkerHTTPBroker(RunManifest{Concurrency: 1}, workerBrokerCredentials{})
	_, err := broker.validatedPayload(
		workerBrokerRouterEvaluate,
		json.RawMessage(`{"model":"entrypoint","model":"other","messages":[],"evaluate_all_signals":true}`),
	)
	if err == nil {
		t.Fatal("broker operation payload with duplicate keys was accepted")
	}
}

func TestWorkerHTTPBrokerAllowsOnlyManifestOperationsAndKeepsCredentialServerSide(t *testing.T) {
	var authorized atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.Header.Get("Authorization") == "Bearer server-secret" {
			authorized.Store(true)
		}
		writer.Header().Set("Content-Type", "application/json")
		writer.Header().Set("x-vsr-selected-model", "logical-fast")
		writer.Header().Set("x-private-provider", "must-not-cross")
		_, _ = writer.Write([]byte(`{"data":[{"id":"virtual-entrypoint","routing":{"resolution":"virtual","selectable":true,"recipe":"recipe-a"}},{"id":"virtual-alias","routing":{"resolution":"virtual","selectable":true,"recipe":"recipe-a"}}]}`))
	}))
	t.Cleanup(server.Close)

	manifest := RunManifest{Concurrency: 2, Target: ManifestTarget{EnvoyURL: server.URL, Mixture: brokerTestMixture()}}
	broker := newWorkerHTTPBroker(manifest, workerBrokerCredentials{envoy: "server-secret"})
	response := broker.execute(context.Background(), workerBrokerRequest{
		ID: 1, Operation: workerBrokerListModels,
		Payload: json.RawMessage("null"), TimeoutMS: 1_000,
	})
	if !response.Success || response.StatusCode == nil || *response.StatusCode != http.StatusOK || !authorized.Load() {
		t.Fatalf("broker response=%+v authorized=%v", response, authorized.Load())
	}
	if response.Headers["x-vsr-selected-model"] != "logical-fast" || len(response.Headers) != 1 {
		t.Fatalf("broker leaked or omitted response headers: %#v", response.Headers)
	}
	if !digestPattern.MatchString(response.BrokerReceipt) {
		t.Fatalf("broker response omitted its server receipt: %+v", response)
	}

	for _, request := range []workerBrokerRequest{
		{ID: 1, Operation: "private.read", Payload: json.RawMessage("null"), TimeoutMS: 1_000},
		{ID: 1, Operation: workerBrokerListModels, Payload: json.RawMessage(`{}`), TimeoutMS: 1_000},
	} {
		encoded, err := json.Marshal(request)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := decodeWorkerBrokerRequest(encoded, 0, broker.operations); err == nil {
			t.Fatalf("unapproved broker request was accepted: %+v", request)
		}
	}
}

func TestWorkerHTTPBrokerNeverFollowsRedirects(t *testing.T) {
	var redirected atomic.Bool
	destination := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		redirected.Store(true)
	}))
	t.Cleanup(destination.Close)
	source := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Location", destination.URL)
		writer.WriteHeader(http.StatusFound)
	}))
	t.Cleanup(source.Close)

	broker := newWorkerHTTPBroker(
		RunManifest{Concurrency: 1, Target: ManifestTarget{EnvoyURL: source.URL}},
		workerBrokerCredentials{},
	)
	response := broker.execute(context.Background(), workerBrokerRequest{
		ID: 1, Operation: workerBrokerListModels,
		Payload: json.RawMessage("null"), TimeoutMS: 1_000,
	})
	if response.StatusCode == nil || *response.StatusCode != http.StatusFound || response.Success || redirected.Load() {
		t.Fatalf("redirect response=%+v destination_called=%v", response, redirected.Load())
	}
}

func TestWorkerHTTPBrokerRejectsDuplicateUpstreamResponseKeys(t *testing.T) {
	broker := newWorkerHTTPBroker(RunManifest{Concurrency: 1}, workerBrokerCredentials{})
	response, _ := broker.readUpstreamResponse(
		workerBrokerRouterEvaluate,
		&http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body: io.NopCloser(strings.NewReader(
				`{"selected_model":"arm-a","selected_model":"arm-b"}`,
			)),
		},
		workerBrokerResponse{Headers: map[string]string{}},
	)
	if response.Success || response.Error == nil || *response.Error != "response_error" {
		t.Fatalf("ambiguous upstream response was accepted: %+v", response)
	}
}

func TestWorkerHTTPBrokerAttestsEveryFrozenVirtualAliasAndRecipe(t *testing.T) {
	tests := map[string]struct {
		payload string
		valid   bool
	}{
		"complete with foreign virtual": {
			payload: `{"data":[{"id":"virtual-alias","routing":{"resolution":"virtual","selectable":true,"recipe":"recipe-a"}},{"id":"foreign","routing":{"resolution":"virtual","selectable":true,"recipe":"other"}},{"id":"virtual-entrypoint","routing":{"resolution":"virtual","selectable":true,"recipe":"recipe-a"}}]}`,
			valid:   true,
		},
		"missing alias": {
			payload: `{"data":[{"id":"virtual-entrypoint","routing":{"resolution":"virtual","selectable":true,"recipe":"recipe-a"}}]}`,
		},
		"wrong recipe": {
			payload: `{"data":[{"id":"virtual-alias","routing":{"resolution":"virtual","selectable":true,"recipe":"other"}},{"id":"virtual-entrypoint","routing":{"resolution":"virtual","selectable":true,"recipe":"recipe-a"}}]}`,
		},
		"duplicate frozen alias": {
			payload: `{"data":[{"id":"virtual-alias","routing":{"resolution":"virtual","selectable":true,"recipe":"recipe-a"}},{"id":"virtual-alias","routing":{"resolution":"virtual","selectable":true,"recipe":"recipe-a"}},{"id":"virtual-entrypoint","routing":{"resolution":"virtual","selectable":true,"recipe":"recipe-a"}}]}`,
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			broker := newWorkerHTTPBroker(
				RunManifest{Concurrency: 1, Target: ManifestTarget{Mixture: brokerTestMixture()}},
				workerBrokerCredentials{},
			)
			var payload map[string]any
			if err := json.Unmarshal([]byte(test.payload), &payload); err != nil {
				t.Fatal(err)
			}
			broker.captureSelectableModels(payload)
			if got := broker.frozenEntrypointDiscovered(); got != test.valid {
				t.Fatalf("frozen mixture discovery = %v, want %v", got, test.valid)
			}
			if test.valid {
				if err := broker.validateRoutedModel("virtual-entrypoint"); err != nil {
					t.Fatalf("frozen entrypoint rejected: %v", err)
				}
				if err := broker.validateRoutedModel("virtual-alias"); err == nil {
					t.Fatal("non-canonical frozen alias was accepted as the evaluation entrypoint")
				}
			}
		})
	}
}

func TestBrokerMixtureBindingRejectsOutOfDecisionArm(t *testing.T) {
	mixture := brokerTestMixture()
	mixture.Decisions[0].ArmIDs = []string{"arm-fast"}
	entrypoint := mixture.EntrypointModel
	recipe := mixture.RecipeName
	selected := "model-strong"
	armID := "arm-strong"
	algorithm := "static"
	decision := "quality"
	entry := executionAttestationEntry{
		Operation: workerBrokerRouterEvaluate, TrackID: "routing", Success: true,
		RequestedModel: &entrypoint, SelectedModel: &selected, ArmID: &armID,
		Recipe: &recipe, Algorithm: &algorithm, DecisionName: &decision, Headers: map[string]string{},
	}
	if err := validateBrokerMixtureBinding(mixture, entry); err == nil {
		t.Fatal("decision selected an arm outside its frozen boundary")
	}
	entry.DecisionName = nil
	if err := validateBrokerMixtureBinding(mixture, entry); err == nil {
		t.Fatal("algorithm selected an arm outside every matching frozen decision")
	}
	mixture.FallbackArmID = armID
	algorithm = "default"
	entry.Algorithm = &algorithm
	if err := validateBrokerMixtureBinding(mixture, entry); err != nil {
		t.Fatalf("explicit default fallback selection rejected: %v", err)
	}
}

func TestRoutingBrokerAttestationBindsRealizedSelectionMethod(t *testing.T) {
	tests := map[string]struct {
		selectionMethod string
		wantValid       bool
	}{
		"authorized realized method":   {selectionMethod: "static", wantValid: true},
		"unauthorized realized method": {selectionMethod: "unapproved"},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				writer.Header().Set("Content-Type", "application/json")
				_, _ = writer.Write([]byte(`{
					"recipe":"recipe-a",
					"selected_model":"model-fast",
					"selection_status":"selected",
					"selection_method":"` + test.selectionMethod + `",
					"decision_result":{"decision_name":"quality","algorithm":"static"}
				}`))
			}))
			t.Cleanup(server.Close)

			mixture := brokerTestMixture()
			broker := newWorkerHTTPBroker(RunManifest{
				Mode:        ModeLive,
				Concurrency: 1,
				TrackIDs:    []TrackID{"routing"},
				Target: ManifestTarget{
					RouterAPIURL: server.URL,
					Mixture:      mixture,
				},
			}, workerBrokerCredentials{})
			broker.captureSelectableModels(map[string]any{"data": []any{
				map[string]any{
					"id": "virtual-entrypoint",
					"routing": map[string]any{
						"resolution": "virtual", "selectable": true, "recipe": "recipe-a",
					},
				},
				map[string]any{
					"id": "virtual-alias",
					"routing": map[string]any{
						"resolution": "virtual", "selectable": true, "recipe": "recipe-a",
					},
				},
			}})
			response := broker.execute(context.Background(), workerBrokerRequest{
				ID: 1, Operation: workerBrokerRouterEvaluate, TrackID: "routing",
				CaseID: "case-1", AttemptID: "attempt-1",
				Payload:   json.RawMessage(`{"model":"virtual-entrypoint","messages":[{"role":"user","content":"hello"}],"evaluate_all_signals":true}`),
				TimeoutMS: 1_000,
			})
			if !response.Success {
				t.Fatalf("routing broker response = %+v", response)
			}
			entry := broker.entries[1]
			if entry.Algorithm == nil || *entry.Algorithm != test.selectionMethod ||
				entry.SelectionMethod == nil || *entry.SelectionMethod != test.selectionMethod {
				t.Fatalf("routing execution projection = %+v", entry)
			}
			if entry.RoutingRecipeDecision == nil ||
				entry.RoutingRecipeDecision.DecisionID != routingRecipeBrokerDecisionID(1) ||
				entry.RoutingRecipeDecision.CaseID != "case-1" ||
				entry.RoutingRecipeDecision.SelectionStatus != "selected" ||
				entry.RoutingRecipeDecision.SelectedArmID != "arm-fast" ||
				len(entry.RoutingRecipeDecision.RankedArmIDs) != 1 ||
				entry.RoutingRecipeDecision.RankedArmIDs[0] != "arm-fast" ||
				entry.FetchedAt == nil ||
				!entry.RoutingRecipeDecision.ObservedAt.Equal(entry.FetchedAt.UTC()) {
				t.Fatalf("broker-owned routing decision snapshot = %+v", entry.RoutingRecipeDecision)
			}
			if err := validateBrokerRoutingRecipeDecision(mixture, entry); err != nil {
				t.Fatalf("broker-owned routing decision rejected: %v", err)
			}
			if entry.responsePayload != nil {
				t.Fatal("ordinary routing payload was retained in the broker transcript")
			}
			if configured := response.Payload["decision_result"].(map[string]any)["algorithm"]; configured != "static" {
				t.Fatalf("configured routing diagnostic algorithm = %v, want static", configured)
			}
			if err := validateStoredExecutionAttestationEntry(entry, 1); err != nil {
				t.Fatalf("stored routing attestation rejected: %v", err)
			}
			err := validateBrokerMixtureBinding(mixture, entry)
			if test.wantValid && err != nil {
				t.Fatalf("authorized realized selector rejected: %v", err)
			}
			if !test.wantValid && err == nil {
				t.Fatal("unauthorized realized selector crossed the frozen decision boundary")
			}
		})
	}
}

func newTypedChatBroker(t *testing.T, chatCalls *atomic.Int64) *workerHTTPBroker {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		switch request.URL.Path {
		case "/v1/models":
			_, _ = writer.Write([]byte(`{"data":[{"id":"foreign-virtual","routing":{"resolution":"virtual","selectable":true,"recipe":"foreign"}},{"id":"virtual-alias","routing":{"resolution":"virtual","selectable":true,"recipe":"recipe-a"}},{"id":"virtual-entrypoint","routing":{"resolution":"virtual","selectable":true,"recipe":"recipe-a"}}]}`))
		case "/v1/chat/completions":
			chatCalls.Add(1)
			var body struct {
				Model            string  `json:"model"`
				Temperature      float64 `json:"temperature"`
				TopP             float64 `json:"top_p"`
				PresencePenalty  float64 `json:"presence_penalty"`
				FrequencyPenalty float64 `json:"frequency_penalty"`
				Seed             int64   `json:"seed"`
				Stream           bool    `json:"stream"`
				MaxTokens        int     `json:"max_tokens"`
			}
			if err := json.NewDecoder(request.Body).Decode(&body); err != nil ||
				(body.Model != "virtual-entrypoint" && body.Model != "model-fast") ||
				body.Temperature != 0 || body.TopP != 1 || body.PresencePenalty != 0 ||
				body.FrequencyPenalty != 0 || body.Seed != 73 || body.Stream ||
				body.MaxTokens != workerBrokerMaxOutputTokens {
				t.Fatalf("published chat request = %+v, error=%v", body, err)
			}
			writer.Header().Set("x-vsr-selected-model", "arm-fast")
			writer.Header().Set("x-vsr-selected-algorithm", "static")
			writer.Header().Set("x-vsr-selected-decision", "quality")
			_, _ = writer.Write([]byte(`{"choices":[{"message":{"content":"  exact   answer "}}],"usage":{"prompt_tokens":3,"completion_tokens":2}}`))
		default:
			http.NotFound(writer, request)
		}
	}))
	t.Cleanup(server.Close)
	broker := newWorkerHTTPBroker(
		RunManifest{SampleLimit: 4, Concurrency: 2, Seed: 73, SuiteIDs: []string{"suite-a", "suite-b"}, TrackIDs: []TrackID{"model_pool", "joint"}, Target: ManifestTarget{
			EnvoyURL: server.URL, Mixture: brokerTestMixture(),
		}},
		workerBrokerCredentials{envoy: "server-secret"},
	)
	if broker.requestMax != 112 {
		t.Fatalf("mixture broker request budget = %d, want 112", broker.requestMax)
	}
	return broker
}

func assertTypedChatBrokerRejectsUnsafeRequests(
	t *testing.T, broker *workerHTTPBroker, chatCalls *atomic.Int64,
	valid, directPayload json.RawMessage,
) {
	t.Helper()
	for _, malicious := range []json.RawMessage{
		json.RawMessage(`{"model":"virtual-entrypoint","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"http://169.254.169.254/latest/meta-data"}}]}],"temperature":0,"stream":false}`),
		json.RawMessage(`{"model":"provider-model","messages":[{"role":"user","content":"hello"}],"temperature":0,"stream":false}`),
		json.RawMessage(`{"model":"virtual-entrypoint","messages":[{"role":"user","content":"hello"}],"temperature":0,"stream":false,"max_tokens":1000000}`),
	} {
		response := broker.execute(context.Background(), workerBrokerRequest{
			ID: 4, Operation: workerBrokerRoutedChatCompletion, TrackID: "capacity",
			CaseID: "case-1", AttemptID: "attempt-unsafe", Payload: malicious, TimeoutMS: 1_000,
		})
		if response.Success || chatCalls.Load() != 2 {
			t.Fatalf("unsafe payload reached upstream: response=%+v calls=%d", response, chatCalls.Load())
		}
	}
	for _, invalid := range []workerBrokerRequest{
		{ID: 5, Operation: workerBrokerArmChatCompletion, TrackID: "model_pool", CaseID: "case-1", AttemptID: "virtual-arm", Payload: valid, TimeoutMS: 1_000},
		{ID: 6, Operation: workerBrokerArmChatCompletion, TrackID: "model_pool", CaseID: "case-1", AttemptID: "alias-arm", Payload: json.RawMessage(`{"model":"virtual-alias","messages":[{"role":"user","content":"hello"}],"temperature":0,"stream":false}`), TimeoutMS: 1_000},
		{ID: 7, Operation: workerBrokerArmChatCompletion, TrackID: "model_pool", CaseID: "case-1", AttemptID: "foreign-arm", Payload: json.RawMessage(`{"model":"foreign-model","messages":[{"role":"user","content":"hello"}],"temperature":0,"stream":false}`), TimeoutMS: 1_000},
		{ID: 8, Operation: workerBrokerRoutedChatCompletion, TrackID: "joint", CaseID: "case-1", AttemptID: "direct-route", Payload: directPayload, TimeoutMS: 1_000},
	} {
		response := broker.execute(context.Background(), invalid)
		if response.Success || chatCalls.Load() != 2 {
			t.Fatalf("invalid frozen-mixture request reached upstream: response=%+v calls=%d", response, chatCalls.Load())
		}
	}
}

func TestWorkerHTTPBrokerPublishesOnlyBoundedTypedChatRequests(t *testing.T) {
	var chatCalls atomic.Int64
	broker := newTypedChatBroker(t, &chatCalls)
	models := broker.execute(context.Background(), workerBrokerRequest{
		ID: 1, Operation: workerBrokerListModels, Payload: json.RawMessage("null"), TimeoutMS: 1_000,
	})
	if !models.Success {
		t.Fatalf("model discovery failed: %+v", models)
	}
	valid := json.RawMessage(`{"model":"virtual-entrypoint","messages":[{"role":"user","content":"hello"}],"temperature":0,"stream":false}`)
	chat := broker.execute(context.Background(), workerBrokerRequest{
		ID: 2, Operation: workerBrokerRoutedChatCompletion, TrackID: "joint",
		CaseID: "case-1", AttemptID: "attempt-1", Payload: valid, TimeoutMS: 1_000,
	})
	if !chat.Success || chatCalls.Load() != 1 {
		t.Fatalf("chat response=%+v calls=%d", chat, chatCalls.Load())
	}
	routedEntry := broker.entries[2]
	if routedEntry.RequestedModel == nil || *routedEntry.RequestedModel != "virtual-entrypoint" ||
		routedEntry.ArmID == nil || *routedEntry.ArmID != "arm-fast" ||
		routedEntry.ResponseContentDigest == nil || *routedEntry.ResponseContentDigest != digestString("exact answer") ||
		routedEntry.Algorithm == nil || *routedEntry.Algorithm != "static" ||
		routedEntry.Recipe == nil || *routedEntry.Recipe != "recipe-a" {
		t.Fatalf("routed attestation = %+v", routedEntry)
	}
	if routedEntry.SelectionStatus == nil || *routedEntry.SelectionStatus != "selected" ||
		routedEntry.SelectionMethod == nil || *routedEntry.SelectionMethod != "static" {
		t.Fatalf("header-only routed selection projection = %+v", routedEntry)
	}
	if err := validateBrokerMixtureBinding(broker.manifest.Target.Mixture, routedEntry); err != nil {
		t.Fatalf("header-only routed mixture binding rejected: %v", err)
	}
	expectedAnswer := "exact answer"
	quality := 1.0
	latencyMS := float64(routedEntry.LatencyMicroseconds) / 1000
	runtimeCost := serverRuntimeCost(routedEntry, broker.manifest.Target.Mixture.ModelArms)
	record := executionRecordEvidence{
		TrackID: "joint", CaseID: "case-1", AttemptID: "attempt-1", Status: "succeeded",
		SelectedArmID: routedEntry.ArmID, SelectionStatus: routedEntry.SelectionStatus,
		SelectionMethod: routedEntry.SelectionMethod, Recipe: routedEntry.Recipe,
		DecisionName: routedEntry.DecisionName, Algorithm: routedEntry.Algorithm,
		Success: &routedEntry.Success, Quality: &quality, LatencyMS: &latencyMS,
		InputTokens: routedEntry.InputTokens, OutputTokens: routedEntry.OutputTokens, RuntimeCost: runtimeCost,
	}
	messageDigest, err := canonicalMessageListDigest([]brokerMessage{{
		Role: "user", Content: json.RawMessage(`"hello"`),
	}})
	if err != nil {
		t.Fatalf("digest sealed visible case: %v", err)
	}
	if err := validateBrokerRecord(
		routedEntry, record, visibleCaseSet{MessageDigests: map[string]string{"case-1": messageDigest}},
		gradingCaseEvidence{ExpectedAnswer: &expectedAnswer},
		broker.manifest.Target.Mixture.ModelArms, nil, broker.manifest.Seed,
	); err != nil {
		t.Fatalf("header-only joint record rejected: %v", err)
	}
	if err := validateBrokerRecord(
		routedEntry, record, visibleCaseSet{MessageDigests: map[string]string{"case-1": messageDigest}},
		gradingCaseEvidence{ExpectedAnswer: &expectedAnswer},
		broker.manifest.Target.Mixture.ModelArms, nil, broker.manifest.Seed+1,
	); err == nil {
		t.Fatal("request attestation accepted a generation seed outside the frozen manifest")
	}
	directPayload := json.RawMessage(`{"model":"model-fast","messages":[{"role":"user","content":"hello"}],"temperature":0,"stream":false}`)
	direct := broker.execute(context.Background(), workerBrokerRequest{
		ID: 3, Operation: workerBrokerArmChatCompletion, TrackID: "model_pool",
		CaseID: "case-1", AttemptID: "attempt-arm-fast", Payload: directPayload, TimeoutMS: 1_000,
	})
	if !direct.Success || chatCalls.Load() != 2 {
		t.Fatalf("direct response=%+v calls=%d", direct, chatCalls.Load())
	}
	directEntry := broker.entries[3]
	if directEntry.RequestedModel == nil || *directEntry.RequestedModel != "model-fast" ||
		directEntry.ArmID == nil || *directEntry.ArmID != "arm-fast" {
		t.Fatalf("direct attestation = %+v", directEntry)
	}
	assertTypedChatBrokerRejectsUnsafeRequests(t, broker, &chatCalls, valid, directPayload)
}

func TestWorkerHTTPBrokerRequiresExactEvidenceIdentityForEveryPOST(t *testing.T) {
	operations := map[string]workerBrokerOperation{
		workerBrokerRouterEvaluate:       {method: http.MethodPost, url: "http://router/api/v1/eval"},
		workerBrokerRoutedChatCompletion: {method: http.MethodPost, url: "http://envoy/v1/chat/completions"},
		workerBrokerArmChatCompletion:    {method: http.MethodPost, url: "http://envoy/v1/chat/completions"},
	}
	payload := json.RawMessage(`{"model":"entrypoint","messages":[{"role":"user","content":"hello"}],"evaluate_all_signals":true}`)
	valid := workerBrokerRequest{
		ID: 1, Operation: workerBrokerRouterEvaluate, TrackID: "routing",
		CaseID: "case-1", AttemptID: "attempt-1", Payload: payload, TimeoutMS: 1_000,
	}
	encoded, _ := json.Marshal(valid)
	if _, err := decodeWorkerBrokerRequest(encoded, 0, operations); err != nil {
		t.Fatalf("valid evidence envelope rejected: %v", err)
	}
	for name, mutate := range map[string]func(*workerBrokerRequest){
		"missing case":    func(request *workerBrokerRequest) { request.CaseID = "" },
		"missing attempt": func(request *workerBrokerRequest) { request.AttemptID = "" },
		"wrong track":     func(request *workerBrokerRequest) { request.TrackID = "capacity" },
		"unknown field":   func(request *workerBrokerRequest) {},
	} {
		request := valid
		mutate(&request)
		encoded, _ := json.Marshal(request)
		if name == "unknown field" {
			var value map[string]any
			_ = json.Unmarshal(encoded, &value)
			value["url"] = "http://attacker"
			encoded, _ = json.Marshal(value)
		}
		if _, err := decodeWorkerBrokerRequest(encoded, 0, operations); err == nil {
			t.Fatalf("%s evidence envelope was accepted", name)
		}
	}
	chatPayload := json.RawMessage(`{"model":"entrypoint","messages":[{"role":"user","content":"hello"}],"temperature":0,"stream":false}`)
	for name, request := range map[string]workerBrokerRequest{
		"routed joint": {ID: 1, Operation: workerBrokerRoutedChatCompletion, TrackID: "joint", CaseID: "case-1", AttemptID: "attempt-1", Payload: chatPayload, TimeoutMS: 1_000},
		"direct pool":  {ID: 1, Operation: workerBrokerArmChatCompletion, TrackID: "model_pool", CaseID: "case-1", AttemptID: "attempt-1", Payload: chatPayload, TimeoutMS: 1_000},
	} {
		encoded, _ := json.Marshal(request)
		if _, err := decodeWorkerBrokerRequest(encoded, 0, operations); err != nil {
			t.Fatalf("valid %s evidence envelope rejected: %v", name, err)
		}
	}
	for name, request := range map[string]workerBrokerRequest{
		"routed on pool":  {ID: 1, Operation: workerBrokerRoutedChatCompletion, TrackID: "model_pool", CaseID: "case-1", AttemptID: "attempt-1", Payload: chatPayload, TimeoutMS: 1_000},
		"direct on joint": {ID: 1, Operation: workerBrokerArmChatCompletion, TrackID: "joint", CaseID: "case-1", AttemptID: "attempt-1", Payload: chatPayload, TimeoutMS: 1_000},
	} {
		encoded, _ := json.Marshal(request)
		if _, err := decodeWorkerBrokerRequest(encoded, 0, operations); err == nil {
			t.Fatalf("invalid %s evidence envelope was accepted", name)
		}
	}
}

func brokerTestMixture() *ManifestMixture {
	recipeName := "recipe-a"
	aliases := []string{"virtual-entrypoint", "virtual-alias"}
	arms := []ModelArm{
		{ID: "arm-fast", Model: "model-fast", ProviderModelIDDigest: digestString("provider-fast"), InputCostPerMillionTokensUSD: 1, OutputCostPerMillionTokensUSD: 2},
		{ID: "arm-strong", Model: "model-strong", ProviderModelIDDigest: digestString("provider-strong"), InputCostPerMillionTokensUSD: 3, OutputCostPerMillionTokensUSD: 4},
	}
	id := "mom-" + digestString(recipeName)[len("sha256:"):]
	recipeDigest := digestString("recipe")
	poolDigest := modelPoolSnapshotDigest(arms)
	selectorPolicyDigest := digestString("selector-policy")
	selectorDigest := selectorSnapshotDigest(selectorPolicyDigest, []SupportModel{})
	decisions := []MixtureDecisionBinding{{Name: "quality", Algorithm: "static", ArmIDs: []string{"arm-fast", "arm-strong"}}}
	mixture := &ManifestMixture{
		SchemaVersion: SchemaVersion, ID: id, EntrypointModel: aliases[0],
		Aliases: aliases, RecipeName: recipeName,
		RecipeDigest: recipeDigest, PoolDigest: poolDigest,
		SelectorPolicyDigest: selectorPolicyDigest, SelectorDigest: selectorDigest,
		AdaptationDigest: digestString("adaptation"),
		BindingDigest:    digestString("binding"),
		ModelArms:        arms,
		SupportModels:    []SupportModel{},
		Decisions:        decisions,
	}
	mustFreezeTestRoutingRecipePlan(mixture)
	return mixture
}
