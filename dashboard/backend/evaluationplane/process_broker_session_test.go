package evaluationplane

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func setWorkerBrokerResponseLimit(t *testing.T, broker *workerHTTPBroker, limit int64) {
	t.Helper()
	broker.session.mu.Lock()
	defer broker.session.mu.Unlock()
	if broker.session.responseBytes != 0 || broker.session.terminalErr != nil {
		t.Fatal("cannot replace an active broker session response limit")
	}
	broker.session.responseLimit = limit
}

func TestWorkerHTTPBrokerResponseBudgetIsAtomicAcrossConcurrentReservations(t *testing.T) {
	broker := newWorkerHTTPBroker(RunManifest{Concurrency: maxWorkerBrokerConcurrent}, workerBrokerCredentials{})
	const (
		reservationBytes = int64(128)
		reservationCount = 64
		responseLimit    = reservationBytes * 8
	)
	setWorkerBrokerResponseLimit(t, broker, responseLimit)

	start := make(chan struct{})
	var successful atomic.Int64
	var workers sync.WaitGroup
	for range reservationCount {
		workers.Add(1)
		go func() {
			defer workers.Done()
			<-start
			if broker.reserveResponseBytes(reservationBytes) == nil {
				successful.Add(1)
			}
		}()
	}
	close(start)
	workers.Wait()

	used, limit := broker.sessionResponseUsage()
	if successful.Load() != 8 || used != responseLimit || limit != responseLimit {
		t.Fatalf("concurrent response admission successes=%d used=%d limit=%d", successful.Load(), used, limit)
	}
	if !errors.Is(broker.sessionFailure(), errWorkerBrokerResponseBudget) {
		t.Fatalf("concurrent response budget failure = %v", broker.sessionFailure())
	}
}

func TestWorkerHTTPBrokerBudgetsConsecutiveSuccessAndErrorBodies(t *testing.T) {
	payload := []byte(`{"padding":"` + strings.Repeat("x", 1024*1024) + `"}`)
	broker := newWorkerHTTPBroker(RunManifest{Concurrency: 1}, workerBrokerCredentials{})
	setWorkerBrokerResponseLimit(t, broker, int64(len(payload)*2))

	responseForStatus := func(status int) workerBrokerResponse {
		response, _ := broker.readUpstreamResponse(
			workerBrokerRouterEvaluate,
			&http.Response{
				StatusCode: status,
				Header:     make(http.Header),
				Body:       io.NopCloser(bytes.NewReader(payload)),
			},
			workerBrokerResponse{Headers: map[string]string{}},
		)
		return response
	}
	if response := responseForStatus(http.StatusOK); !response.Success {
		t.Fatalf("first bounded success response = %+v", response)
	}
	if response := responseForStatus(http.StatusServiceUnavailable); response.Success || response.Error == nil ||
		*response.Error != "HTTP 503" {
		t.Fatalf("bounded non-2xx response = %+v", response)
	}
	if response := responseForStatus(http.StatusOK); response.Success || response.Error == nil ||
		*response.Error != "response_budget_exceeded" {
		t.Fatalf("response beyond the session budget = %+v", response)
	}
	used, limit := broker.sessionResponseUsage()
	if used != int64(len(payload)*2) || used > limit || !errors.Is(broker.sessionFailure(), errWorkerBrokerResponseBudget) {
		t.Fatalf("consecutive response budget used=%d limit=%d failure=%v", used, limit, broker.sessionFailure())
	}
}

func TestWorkerHTTPBrokerServeFailsClosedWhenReplyFrameExceedsSessionBudget(t *testing.T) {
	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"attempts":[]}`))
	}))
	t.Cleanup(server.Close)

	manifest := RunManifest{Concurrency: 1, Target: ManifestTarget{AgentTaskLedger: &ServiceEndpoint{
		SchemaVersion: SchemaVersion, URL: server.URL, TimeoutSeconds: 2,
	}}}
	broker := newWorkerHTTPBroker(manifest, workerBrokerCredentials{})
	// The upstream body fits, while any framed broker response does not.
	setWorkerBrokerResponseLimit(t, broker, int64(len(`{"attempts":[]}`)+1))
	request := workerBrokerRequest{
		ID: 1, Operation: workerBrokerAgentTaskLedger, TrackID: "agentic",
		CaseID: "agent-task-ledger", AttemptID: "ledger-fetch",
		Payload: json.RawMessage("null"), TimeoutMS: 1_000,
	}
	encoded, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	frame := make([]byte, 4+len(encoded))
	//nolint:gosec // The request was encoded from one bounded test envelope.
	binary.BigEndian.PutUint32(frame[:4], uint32(len(encoded)))
	copy(frame[4:], encoded)
	var output bytes.Buffer
	err = broker.serve(context.Background(), bytes.NewReader(frame), &output)
	if !errors.Is(err, errWorkerBrokerResponseBudget) || output.Len() != 0 || calls.Load() != 1 {
		t.Fatalf("broker serve error=%v output_bytes=%d calls=%d", err, output.Len(), calls.Load())
	}
}

func TestWorkerHTTPBrokerRejectsMethodLedgerReplayBeforeSecondFetch(t *testing.T) {
	broker, calls := newLedgerContractBroker(t)
	request := workerBrokerRequest{
		ID: 1, Operation: workerBrokerAgentTaskLedger, TrackID: "agentic",
		CaseID: "agent-task-ledger", AttemptID: "ledger-fetch",
		Payload: json.RawMessage("null"), TimeoutMS: 1_000,
	}
	if response := broker.execute(context.Background(), request); !response.Success {
		t.Fatalf("first method ledger response = %+v", response)
	}
	request.ID++
	if response := broker.execute(context.Background(), request); response.Success {
		t.Fatalf("duplicate method ledger response = %+v", response)
	}
	if calls.Load() != 1 || !errors.Is(broker.sessionFailure(), errWorkerBrokerLedgerReplay) {
		t.Fatalf("duplicate method ledger calls=%d failure=%v", calls.Load(), broker.sessionFailure())
	}
}

func TestWorkerHTTPBrokerDoesNotRetainUntypedLedgerOrOrdinaryPayloads(t *testing.T) {
	ledgerServer := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"attempts":[],"adversarial_padding":"` + strings.Repeat("x", 1_024) + `"}`))
	}))
	t.Cleanup(ledgerServer.Close)
	ledgerBroker := newWorkerHTTPBroker(RunManifest{Concurrency: 1, Target: ManifestTarget{
		AgentTaskLedger: &ServiceEndpoint{SchemaVersion: SchemaVersion, URL: ledgerServer.URL, TimeoutSeconds: 2},
	}}, workerBrokerCredentials{})
	ledgerResponse := ledgerBroker.execute(context.Background(), workerBrokerRequest{
		ID: 1, Operation: workerBrokerAgentTaskLedger, TrackID: "agentic",
		CaseID: "agent-task-ledger", AttemptID: "ledger-fetch",
		Payload: json.RawMessage("null"), TimeoutMS: 1_000,
	})
	if ledgerResponse.Success || ledgerBroker.entries[1].responsePayload != nil {
		t.Fatalf("untyped method ledger was retained: response=%+v entry=%+v", ledgerResponse, ledgerBroker.entries[1])
	}

	chatServer := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		writer.Header().Set("x-vsr-selected-model", "arm-fast")
		writer.Header().Set("x-vsr-selected-algorithm", "static")
		_, _ = writer.Write([]byte(`{"choices":[{"message":{"content":"answer"}}],"usage":{"prompt_tokens":1,"completion_tokens":1},"padding":"` + strings.Repeat("x", 1_024) + `"}`))
	}))
	t.Cleanup(chatServer.Close)
	mixture := brokerTestMixture()
	chatBroker := newWorkerHTTPBroker(RunManifest{
		Mode: ModeLive, Concurrency: 1, Seed: 1,
		TrackIDs: []TrackID{"joint"}, Target: ManifestTarget{EnvoyURL: chatServer.URL, Mixture: mixture},
	}, workerBrokerCredentials{})
	chatBroker.models[mixture.EntrypointModel] = mixture.RecipeName
	chatBroker.modelsValid = true
	chatResponse := chatBroker.execute(context.Background(), workerBrokerRequest{
		ID: 1, Operation: workerBrokerRoutedChatCompletion, TrackID: "joint",
		CaseID: "case-1", AttemptID: "attempt-1",
		Payload: json.RawMessage(`{"model":"virtual-entrypoint","messages":[{"role":"user","content":"hello"}],"temperature":0,"stream":false}`), TimeoutMS: 1_000,
	})
	entry := chatBroker.entries[1]
	if !chatResponse.Success || entry.responsePayload != nil || !digestPattern.MatchString(entry.ResponseDigest) ||
		entry.ResponseContentDigest == nil {
		t.Fatalf("ordinary response retention = response=%+v entry=%+v", chatResponse, entry)
	}
}
