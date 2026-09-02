package evaluationplane

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	workerBrokerRequestFD       = 3
	workerBrokerResponseFD      = 4
	maxWorkerBrokerFrameBytes   = 4 * 1024 * 1024
	maxWorkerBrokerRequestBytes = 512 * 1024 * 1024
	// The session response budget covers every upstream body byte and every
	// framed reply byte, including rejected and non-2xx responses. Keeping one
	// shared budget prevents concurrent requests from each consuming the
	// per-frame maximum for the lifetime of a worker.
	maxWorkerBrokerResponseBytes       = 512 * 1024 * 1024
	maxWorkerBrokerResponseHeaderBytes = 16 * 1024
	maxWorkerBrokerRequests            = 50_000
	maxWorkerBrokerTimeoutMS           = 300_000
	maxWorkerBrokerConcurrent          = 128
)

const (
	workerBrokerListModels                 = "models.list"
	workerBrokerRoutedChatCompletion       = "routed-chat.completions"
	workerBrokerArmChatCompletion          = "arm-chat.completions"
	workerBrokerRouterEvaluate             = "router.evaluate"
	workerBrokerAgentTaskLedger            = "agent-task.ledger"
	workerBrokerFaultRecoveryLedger        = "fault-recovery.ledger"
	workerBrokerHardPolicyLedger           = "hard-policy.ledger"
	workerBrokerProductionExperimentLedger = "production.experiment-ledger"
)

type workerBrokerRequest struct {
	ID        uint64          `json:"id"`
	Operation string          `json:"operation"`
	TrackID   TrackID         `json:"track_id,omitempty"`
	CaseID    string          `json:"case_id,omitempty"`
	AttemptID string          `json:"attempt_id,omitempty"`
	Payload   json.RawMessage `json:"payload"`
	TimeoutMS int             `json:"timeout_ms"`
}

type workerBrokerResponse struct {
	ID            uint64            `json:"id"`
	Success       bool              `json:"success"`
	StatusCode    *int              `json:"status_code"`
	Payload       map[string]any    `json:"payload"`
	LatencyMS     float64           `json:"latency_ms"`
	FetchedAt     time.Time         `json:"fetched_at"`
	Headers       map[string]string `json:"headers"`
	Error         *string           `json:"error"`
	BrokerReceipt string            `json:"broker_receipt"`

	// Only a strictly decoded method-ledger payload may cross into the
	// server-side reducer. Ordinary responses are projected into bounded
	// attestation fields and then released after their raw digest is recorded.
	retainedMethodLedgerPayload map[string]any `json:"-"`
}

type workerBrokerOperation struct {
	method       string
	url          string
	credential   string
	maxTimeoutMS int
}

type workerHTTPBroker struct {
	manifest       RunManifest
	operations     map[string]workerBrokerOperation
	client         *http.Client
	semaphore      chan struct{}
	writeMu        sync.Mutex
	modelsMu       sync.RWMutex
	models         map[string]string
	modelsValid    bool
	entriesMu      sync.Mutex
	entries        map[uint64]executionAttestationEntry
	startedAt      time.Time
	requestMax     int
	controlledPair *controlledPairRunContext
	session        workerBrokerSessionState
}

func newWorkerHTTPBroker(manifest RunManifest, credentials workerBrokerCredentials) *workerHTTPBroker {
	operations := make(map[string]workerBrokerOperation, 8)
	add := func(name, method, rawURL, credential string, maxTimeoutMS int) {
		operations[name] = workerBrokerOperation{
			method: method, url: rawURL, credential: credential, maxTimeoutMS: maxTimeoutMS,
		}
	}
	if manifest.Target.EnvoyURL != "" {
		add(workerBrokerListModels, http.MethodGet, manifest.Target.EnvoyURL+"/v1/models", credentials.envoy, 0)
		add(workerBrokerRoutedChatCompletion, http.MethodPost, manifest.Target.EnvoyURL+"/v1/chat/completions", credentials.envoy, 0)
		add(workerBrokerArmChatCompletion, http.MethodPost, manifest.Target.EnvoyURL+"/v1/chat/completions", credentials.envoy, 0)
	}
	if manifest.Target.RouterAPIURL != "" {
		add(workerBrokerRouterEvaluate, http.MethodPost, manifest.Target.RouterAPIURL+"/api/v1/eval?trace=true", credentials.router, 0)
	}
	if endpoint := manifest.Target.AgentTaskLedger; endpoint != nil {
		add(workerBrokerAgentTaskLedger, http.MethodGet, endpoint.URL, credentials.agentTaskLedger, endpointTimeoutMS(endpoint))
	}
	if endpoint := manifest.Target.FaultRecoveryLedger; endpoint != nil {
		add(workerBrokerFaultRecoveryLedger, http.MethodGet, endpoint.URL, credentials.faultRecoveryLedger, endpointTimeoutMS(endpoint))
	}
	if endpoint := manifest.Target.HardPolicyLedger; endpoint != nil {
		add(workerBrokerHardPolicyLedger, http.MethodGet, endpoint.URL, credentials.hardPolicyLedger, endpointTimeoutMS(endpoint))
	}
	if endpoint := manifest.Target.ProductionExperimentLedger; endpoint != nil {
		add(workerBrokerProductionExperimentLedger, http.MethodGet, endpoint.URL, credentials.productionExperimentLedger, endpointTimeoutMS(endpoint))
	}
	concurrency := manifest.Concurrency
	if concurrency < 1 {
		concurrency = 1
	}
	if concurrency > maxWorkerBrokerConcurrent {
		concurrency = maxWorkerBrokerConcurrent
	}
	perCaseRequestBudget := int64(len(manifest.TrackIDs) + 2)
	if manifest.Target.Mixture != nil && containsTrack(manifest.TrackIDs, "model_pool") {
		perCaseRequestBudget += int64(len(manifest.Target.Mixture.ModelArms))
	}
	caseBudget := int64(manifest.SampleLimit)
	if len(manifest.SuiteIDs) > 1 {
		caseBudget *= int64(len(manifest.SuiteIDs))
	}
	requestMaxBudget := int64(64) + caseBudget*perCaseRequestBudget
	if manifest.CapacityLoadProtocol != nil {
		requestMaxBudget += capacityLoadRequestBudget(*manifest.CapacityLoadProtocol)
	}
	if requestMaxBudget > maxWorkerBrokerRequests {
		requestMaxBudget = maxWorkerBrokerRequests
	}
	requestMax := int(requestMaxBudget)
	transport := &http.Transport{
		Proxy: nil,
		DialContext: (&net.Dialer{
			Timeout: 10 * time.Second, KeepAlive: 30 * time.Second,
		}).DialContext,
		DisableCompression:     true,
		ForceAttemptHTTP2:      false,
		MaxIdleConns:           concurrency,
		MaxIdleConnsPerHost:    concurrency,
		IdleConnTimeout:        30 * time.Second,
		TLSHandshakeTimeout:    10 * time.Second,
		MaxResponseHeaderBytes: maxWorkerBrokerResponseHeaderBytes,
	}
	return &workerHTTPBroker{
		manifest:   manifest,
		operations: operations,
		client: &http.Client{
			Transport: transport,
			CheckRedirect: func(*http.Request, []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
		semaphore:  make(chan struct{}, concurrency),
		models:     make(map[string]string),
		entries:    make(map[uint64]executionAttestationEntry),
		startedAt:  time.Now().UTC(),
		requestMax: requestMax,
		session:    newWorkerBrokerSessionState(maxWorkerBrokerResponseBytes),
	}
}

func endpointTimeoutMS(endpoint *ServiceEndpoint) int {
	return int(math.Ceil(endpoint.TimeoutSeconds * 1000))
}

func (broker *workerHTTPBroker) execute(ctx context.Context, request workerBrokerRequest) (response workerBrokerResponse) {
	if err := broker.admitRequest(request); err != nil {
		broker.abortSession(err)
		return failedWorkerBrokerResponse(
			workerBrokerResponse{ID: request.ID, Headers: map[string]string{}},
			"request_error",
		)
	}
	return broker.executeAdmitted(ctx, request)
}

func (broker *workerHTTPBroker) executeAdmitted(ctx context.Context, request workerBrokerRequest) (response workerBrokerResponse) {
	started := time.Now()
	response = workerBrokerResponse{ID: request.ID, Headers: map[string]string{}}
	requestPayload := bytes.TrimSpace(request.Payload)
	var responsePayloadBytes []byte
	upstreamAttempted := false
	var pairing *controlledPairObservation
	var pairLease *controlledPairLease
	defer func() {
		completedAt := time.Now().UTC()
		response.FetchedAt = completedAt
		if pairing != nil {
			if pairing.ObservedAt.IsZero() {
				pairing.ObservedAt = started.UTC()
			}
			pairing.CompletedAt = completedAt
			pairLease.complete(completedAt)
		}
		elapsed := time.Since(started).Microseconds()
		if elapsed < 0 {
			elapsed = 0
		}
		response.LatencyMS = float64(elapsed) / 1000
		entry := broker.attestResponse(
			request, requestPayload, responsePayloadBytes, response, upstreamAttempted, elapsed, pairing,
		)
		response.BrokerReceipt = entry.BrokerReceipt
	}()
	operation, ok := broker.operations[request.Operation]
	if !ok {
		return failedWorkerBrokerResponse(response, "request_error")
	}
	validatedPayload, err := broker.validatedPayload(request.Operation, request.Payload)
	if err != nil {
		return failedWorkerBrokerResponse(response, "request_error")
	}
	requestPayload = validatedPayload
	if broker.controlledPair != nil {
		pairing, pairLease, err = broker.controlledPair.coordinator.before(
			ctx, broker.controlledPair.role, request, requestPayload,
		)
		if err != nil {
			broker.controlledPair.coordinator.abort(err)
			return failedWorkerBrokerResponse(response, "controlled_pair_error")
		}
		started = time.Now()
		if pairing != nil {
			pairing.ObservedAt = started.UTC()
		}
	}
	requestContext, cancel := context.WithTimeout(ctx, time.Duration(request.TimeoutMS)*time.Millisecond)
	defer cancel()
	httpRequest, err := newWorkerBrokerHTTPRequest(requestContext, operation, requestPayload)
	if err != nil {
		return failedWorkerBrokerResponse(response, "request_error")
	}
	upstreamAttempted = true
	httpResponse, err := broker.client.Do(httpRequest)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(requestContext.Err(), context.DeadlineExceeded) {
			return failedWorkerBrokerResponse(response, "request_timeout")
		}
		return failedWorkerBrokerResponse(response, "request_error")
	}
	defer func() {
		if closeErr := httpResponse.Body.Close(); closeErr != nil {
			response = failedWorkerBrokerResponse(response, "response_error")
			response.Payload = nil
			response.retainedMethodLedgerPayload = nil
		}
	}()
	response, responsePayloadBytes = broker.readUpstreamResponse(request.Operation, httpResponse, response)
	return response
}

func newWorkerBrokerHTTPRequest(
	ctx context.Context,
	operation workerBrokerOperation,
	payload []byte,
) (*http.Request, error) {
	var body io.Reader
	if operation.method == http.MethodPost {
		body = bytes.NewReader(payload)
	}
	request, err := http.NewRequestWithContext(ctx, operation.method, operation.url, body)
	if err != nil {
		return nil, err
	}
	request.Header.Set("Accept", "application/json")
	if operation.method == http.MethodPost {
		request.Header.Set("Content-Type", "application/json")
	}
	if operation.credential != "" {
		request.Header.Set("Authorization", "Bearer "+operation.credential)
	}
	return request, nil
}

func (broker *workerHTTPBroker) readUpstreamResponse(
	operation string,
	httpResponse *http.Response,
	response workerBrokerResponse,
) (workerBrokerResponse, []byte) {
	response.StatusCode = &httpResponse.StatusCode
	for _, name := range []string{
		"x-vsr-selected-model", "x-vsr-selected-algorithm",
		"x-vsr-selected-recipe", "x-vsr-selected-decision",
	} {
		value := httpResponse.Header.Get(name)
		if value != "" && len(value) <= 256 && !strings.ContainsAny(value, "\r\n") {
			response.Headers[name] = value
		}
	}
	limited := io.LimitReader(httpResponse.Body, maxWorkerBrokerFrameBytes+1)
	data, readErr := io.ReadAll(limited)
	if err := broker.reserveResponseBytes(int64(len(data))); err != nil {
		return failedWorkerBrokerResponse(response, "response_budget_exceeded"), data
	}
	if readErr != nil || len(data) > maxWorkerBrokerFrameBytes {
		return failedWorkerBrokerResponse(response, "response_error"), data
	}
	if err := rejectDuplicateJSONKeys(data); err != nil {
		return failedWorkerBrokerResponse(response, "response_error"), data
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var responsePayload map[string]any
	if err := decoder.Decode(&responsePayload); err == nil && ensureJSONEOF(decoder) == nil && responsePayload != nil {
		response.Payload = responsePayload
	}
	response.Success = httpResponse.StatusCode >= 200 && httpResponse.StatusCode < 300 && response.Payload != nil
	if response.Success && isMethodLedgerOperation(operation) {
		retained, err := retainMethodLedgerPayload(operation, data, response.Payload)
		if err != nil {
			response.Payload = nil
			return failedWorkerBrokerResponse(response, "response_error"), data
		}
		response.retainedMethodLedgerPayload = retained
	}
	if operation == workerBrokerListModels && response.Payload != nil {
		broker.captureSelectableModels(response.Payload)
	}
	if operation == workerBrokerListModels && response.Success && !broker.frozenEntrypointDiscovered() {
		response.Success = false
	}
	if !response.Success {
		message := "HTTP " + strconv.Itoa(httpResponse.StatusCode)
		response.Error = &message
	}
	return response, data
}

func (broker *workerHTTPBroker) frozenEntrypointDiscovered() bool {
	if broker.manifest.Target.Mixture == nil || len(broker.manifest.Target.Mixture.Aliases) == 0 {
		return false
	}
	broker.modelsMu.RLock()
	valid := broker.modelsValid
	for _, alias := range broker.manifest.Target.Mixture.Aliases {
		recipe, present := broker.models[alias]
		if !present || recipe != broker.manifest.Target.Mixture.RecipeName {
			valid = false
			break
		}
	}
	broker.modelsMu.RUnlock()
	return valid
}

func failedWorkerBrokerResponse(response workerBrokerResponse, message string) workerBrokerResponse {
	response.Success = false
	response.Error = &message
	return response
}
