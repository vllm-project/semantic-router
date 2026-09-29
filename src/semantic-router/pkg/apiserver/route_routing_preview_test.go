//go:build !windows && cgo

package apiserver

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestPreviewHTTPRequiresPublishedClassifierThenEvaluatesHeuristicSignals(t *testing.T) {
	cfg := contextEvalRouterConfig()
	cfg.Decisions[0].Name = "small-request-context-route"
	cfg.Decisions[0].Rules.Name = "small-request-context"
	registry := routerruntime.NewRegistry(cfg)
	api := &ClassificationAPIServer{config: cfg, runtimeRegistry: registry}
	server := httptest.NewServer(api.setupRoutes())
	t.Cleanup(server.Close)
	request := func() (int, []byte) {
		t.Helper()
		response, err := server.Client().Post(server.URL+apiRoutingPreviewPath+"?trace=true", "application/json", strings.NewReader(`{"text":"hello"}`))
		if err != nil {
			t.Fatal(err)
		}
		defer response.Body.Close()
		body, err := io.ReadAll(response.Body)
		if err != nil {
			t.Fatal(err)
		}
		return response.StatusCode, body
	}
	status, body := request()
	if status != http.StatusServiceUnavailable || !strings.Contains(string(body), "CLASSIFIER_UNAVAILABLE") {
		t.Fatalf("Preview before classifier publication = %d %s", status, body)
	}
	ready := newContextEvalServer(t, cfg)
	service := ready.classificationSvc.(*services.ClassificationService)
	t.Cleanup(func() {
		if err := service.Close(); err != nil {
			t.Error(err)
		}
	})
	registry.SetClassificationService(service)
	status, body = request()
	if status != http.StatusOK {
		t.Fatalf("Preview after classifier publication = %d %s", status, body)
	}
	var result services.EvalResponse
	if err := json.Unmarshal(body, &result); err != nil {
		t.Fatal(err)
	}
	if result.DecisionResult == nil || result.DecisionResult.DecisionName != "small-request-context-route" ||
		result.DecisionResult.MatchedSignals == nil || !containsString(result.DecisionResult.MatchedSignals.Context, "small-request-context") ||
		result.Metrics == nil || len(result.EvalTrace) == 0 {
		t.Fatalf("initialized heuristic classifier did not evaluate signals and decision: %+v", result)
	}
}

type blockingPreviewService struct {
	evalCaptureClassificationService
	started chan context.Context
	finish  chan struct{}
}

func (s *blockingPreviewService) ClassifyIntentForEval(ctx context.Context, _ services.IntentRequest) (*services.EvalResponse, error) {
	s.started <- ctx
	<-s.finish // Native work can continue after ctx is canceled.
	if s.evalResp != nil {
		return s.evalResp, s.evalErr
	}
	return &services.EvalResponse{OriginalText: "finished"}, nil
}

func startBlockingPreview(t *testing.T, timeout, concurrency int) (*Server, *blockingPreviewService, *countedAPIResource, *atomic.Int32, func()) {
	t.Helper()
	service := &blockingPreviewService{started: make(chan context.Context, 16), finish: make(chan struct{})}
	var leases atomic.Int32
	owner := &countedAPIResource{}
	cfg := &config.RouterConfig{}
	cfg.API.RoutingPreview = config.RoutingPreviewConfig{RequestTimeoutSeconds: &timeout, MaxConcurrency: &concurrency}
	api := &ClassificationAPIServer{
		config: cfg,
		classificationSvc: newLiveClassificationService(service, nil, func() (classificationService, func(), bool) {
			leases.Add(1)
			return service, func() { leases.Add(-1) }, true
		}),
	}
	api.initRoutingPreviewAdmission(cfg)
	server, err := startHTTPServer(&http.Server{
		Addr: "127.0.0.1:0", ReadHeaderTimeout: time.Second,
		WriteTimeout: 20 * time.Millisecond,
		Handler:      api.setupRoutes(),
	}, owner)
	if err != nil {
		t.Fatal(err)
	}
	var once sync.Once
	unblock := func() { once.Do(func() { close(service.finish) }) }
	t.Cleanup(func() {
		unblock()
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		if err := server.Shutdown(ctx); err != nil {
			t.Error(err)
		}
	})
	return server, service, owner, &leases, unblock
}

func previewHTTP(t *testing.T, server *Server) (int, string) {
	t.Helper()
	client := &http.Client{Timeout: 3 * time.Second}
	response, err := client.Post("http://"+server.httpServer.Addr+apiRoutingPreviewPath, "application/json", strings.NewReader(`{"text":"hello"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	return response.StatusCode, string(body)
}

func TestPreviewHTTPDeadlineRetainsNativeLeaseAndAdmissionUntilDrain(t *testing.T) {
	server, service, owner, leases, unblock := startBlockingPreview(t, 1, 1)
	service.evalResp = &services.EvalResponse{DecisionError: "unresolved"}
	service.evalErr = decision.ErrDecisionUnresolved
	status, body := previewHTTP(t, server)
	if status != http.StatusGatewayTimeout || !strings.Contains(body, "REQUEST_TIMEOUT") {
		t.Fatalf("deadline response = %d %s", status, body)
	}
	workerContext := <-service.started
	if !errors.Is(workerContext.Err(), context.DeadlineExceeded) {
		t.Fatalf("worker context = %v", workerContext.Err())
	}
	status, _ = previewHTTP(t, server)
	if status != http.StatusTooManyRequests || leases.Load() != 1 || owner.closes.Load() != 0 {
		t.Fatalf("timeout freed capacity or owner: status=%d leases=%d closes=%d", status, leases.Load(), owner.closes.Load())
	}
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()
	if err := server.Shutdown(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("shutdown while native worker runs = %v", err)
	}
	if leases.Load() != 1 || owner.closes.Load() != 0 {
		t.Fatal("shutdown released a live native worker")
	}
	unblock()
	select {
	case <-server.ownedDone:
	case <-time.After(time.Second):
		t.Fatal("native owner did not drain after worker completion")
	}
	if leases.Load() != 0 || owner.closes.Load() != 1 {
		t.Fatalf("drained leases=%d owner closes=%d", leases.Load(), owner.closes.Load())
	}
}

func TestPreviewHTTPClientCancellationRetainsWorker(t *testing.T) {
	server, service, owner, leases, _ := startBlockingPreview(t, 120, 1)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://"+server.httpServer.Addr+apiRoutingPreviewPath, strings.NewReader(`{"text":"hello"}`))
	if err != nil {
		t.Fatal(err)
	}
	clientDone := make(chan error, 1)
	go func() {
		response, err := http.DefaultClient.Do(req)
		if response != nil {
			_ = response.Body.Close()
		}
		clientDone <- err
	}()
	var workerContext context.Context
	select {
	case workerContext = <-service.started:
	case <-time.After(time.Second):
		t.Fatal("Preview did not start")
	}
	cancel()
	if err := <-clientDone; !errors.Is(err, context.Canceled) {
		t.Fatalf("client cancellation = %v", err)
	}
	select {
	case <-workerContext.Done():
	case <-time.After(time.Second):
		t.Fatal("client cancellation did not reach inference")
	}
	status, _ := previewHTTP(t, server)
	if status != http.StatusTooManyRequests || leases.Load() != 1 || owner.closes.Load() != 0 {
		t.Fatalf("cancellation released native ownership: status=%d leases=%d closes=%d", status, leases.Load(), owner.closes.Load())
	}
}

func TestPreviewHTTPDefaultAdmissionAcceptsEightConcurrentProbes(t *testing.T) {
	server, service, _, leases, unblock := startBlockingPreview(t, 120, config.DefaultRoutingPreviewMaxConcurrency)
	results := make(chan int, 8)
	for range 8 {
		go func() {
			status, _ := previewHTTP(t, server)
			results <- status
		}()
	}
	for range 8 {
		select {
		case ctx := <-service.started:
			deadline, ok := ctx.Deadline()
			if !ok || time.Until(deadline) < 115*time.Second {
				t.Fatalf("default Preview budget = %v, %v", deadline, ok)
			}
		case <-time.After(time.Second):
			t.Fatal("eight concurrent probes were not admitted")
		}
	}
	if leases.Load() != 8 {
		t.Fatalf("live leases = %d, want 8", leases.Load())
	}
	// Passing the listener's short write timeout must still allow a response.
	time.Sleep(50 * time.Millisecond)
	unblock()
	for range 8 {
		select {
		case status := <-results:
			if status != http.StatusOK {
				t.Fatalf("successful Preview response = %d, want 200", status)
			}
		case <-time.After(time.Second):
			t.Fatal("Preview did not respond after native completion")
		}
	}
}

func TestPreviewServiceCancellationCannotReturnPartialSuccess(t *testing.T) {
	for _, test := range []struct {
		err    error
		status int
		code   string
	}{
		{context.Canceled, http.StatusServiceUnavailable, "INFERENCE_CANCELED"},
		{context.DeadlineExceeded, http.StatusGatewayTimeout, "REQUEST_TIMEOUT"},
	} {
		t.Run(test.code, func(t *testing.T) {
			api := &ClassificationAPIServer{classificationSvc: &evalCaptureClassificationService{
				evalResp: &services.EvalResponse{OriginalText: "partial"}, evalErr: test.err,
			}}
			response := httptest.NewRecorder()
			api.handleEvalClassification(response, httptest.NewRequest(http.MethodPost, apiRoutingPreviewPath, strings.NewReader(`{"text":"hello"}`)))
			if response.Code != test.status || !strings.Contains(response.Body.String(), test.code) {
				t.Fatalf("canceled inference returned %d %s", response.Code, response.Body.String())
			}
		})
	}
}
