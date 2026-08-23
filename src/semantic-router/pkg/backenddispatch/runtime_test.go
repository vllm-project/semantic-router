package backenddispatch

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

type snapshotSourceStub struct {
	snapshot *routingsnapshot.Snapshot
	err      error
	mu       sync.Mutex
	requests []snapshotRequest
}

type snapshotRequest struct {
	pin routingcontext.Generation
}

func (source *snapshotSourceStub) Snapshot(_ context.Context, pin routingcontext.Generation) (*routingsnapshot.Snapshot, error) {
	source.mu.Lock()
	defer source.mu.Unlock()
	source.requests = append(source.requests, snapshotRequest{pin: pin})
	return source.snapshot, source.err
}

type credentialResolverStub struct {
	version    string
	credential backendinvoker.Credential
	pinErr     error
	resolveErr error
	mu         sync.Mutex
	pins       int
	resolves   int
}

func (resolver *credentialResolverStub) Pin(context.Context, backendinvoker.CredentialPublication, string, string, string) (string, error) {
	resolver.mu.Lock()
	defer resolver.mu.Unlock()
	resolver.pins++
	return resolver.version, resolver.pinErr
}

func (resolver *credentialResolverStub) ResolvePinned(context.Context, backendinvoker.CredentialPublication, string, string, string, string) (backendinvoker.Credential, error) {
	resolver.mu.Lock()
	defer resolver.mu.Unlock()
	resolver.resolves++
	return resolver.credential, resolver.resolveErr
}

type journalStub struct {
	mu         sync.Mutex
	dispatches int
	attempts   int
	finished   int
}

type finalizerStub struct{}

func (finalizerStub) Finalize(
	context.Context,
	backendinvoker.Plan,
	backendinvoker.AttemptResult,
	backendinvoker.ResponseTerminal,
) error {
	return nil
}

const testChatResponseBody = `{"id":"chatcmpl-test","object":"chat.completion","created":1,"model":"provider-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`

func (journal *journalStub) BeginDispatch(context.Context, backendinvoker.Plan, time.Time) error {
	journal.mu.Lock()
	defer journal.mu.Unlock()
	journal.dispatches++
	return nil
}

func (journal *journalStub) BeginAttempt(context.Context, backendinvoker.Plan, backendinvoker.Attempt) error {
	journal.mu.Lock()
	defer journal.mu.Unlock()
	journal.attempts++
	return nil
}

func (journal *journalStub) FinishAttempt(context.Context, backendinvoker.Plan, backendinvoker.AttemptResult) error {
	journal.mu.Lock()
	defer journal.mu.Unlock()
	journal.finished++
	return nil
}

type observerStub struct {
	mu      sync.Mutex
	count   int
	err     error
	started chan struct{}
	release chan struct{}
}

func (observer *observerStub) Observe(_ context.Context, _ backendinvoker.Plan, _ backendinvoker.AttemptResult, response *http.Response) (io.ReadCloser, error) {
	observer.mu.Lock()
	observer.count++
	started := observer.started
	release := observer.release
	err := observer.err
	observer.mu.Unlock()
	if started != nil {
		close(started)
	}
	if release != nil {
		<-release
	}
	if err != nil {
		return nil, err
	}
	return response.Body, nil
}

type transportStub struct {
	roundTrip func(*http.Request) (*http.Response, error)
	mu        sync.Mutex
	closed    int
	closeErr  error
}

type codecStub struct {
	protocolcodec.OpenAIChatCodec
	format llmprotocol.WireFormat
}

func (codec codecStub) Format() llmprotocol.WireFormat { return codec.format }

func (transport *transportStub) RoundTrip(request *http.Request) (*http.Response, error) {
	return transport.roundTrip(request)
}

func (transport *transportStub) Close() error {
	transport.mu.Lock()
	defer transport.mu.Unlock()
	transport.closed++
	return transport.closeErr
}

func TestNewRejectsIncompleteAndInvalidComposition(t *testing.T) {
	valid := validOptions(t)
	emptyCodecs, err := protocolcodec.NewRegistry()
	if err != nil {
		emptyCodecs = nil
	}
	var typedNilSnapshots *snapshotSourceStub
	for name, mutate := range map[string]func(*Options){
		"audience":        func(options *Options) { options.Audience = " Invalid " },
		"snapshots":       func(options *Options) { options.Snapshots = nil },
		"typed snapshots": func(options *Options) { options.Snapshots = typedNilSnapshots },
		"credentials":     func(options *Options) { options.Credentials = nil },
		"codecs":          func(options *Options) { options.Codecs = nil },
		"empty codecs":    func(options *Options) { options.Codecs = emptyCodecs },
		"journal":         func(options *Options) { options.Journal = nil },
		"finalizer":       func(options *Options) { options.Finalizer = nil },
		"observer":        func(options *Options) { options.Observer = nil },
		"transport":       func(options *Options) { options.Transport = nil },
		"body zero":       func(options *Options) { options.MaxRequestBodyBytes = 0 },
		"body too large":  func(options *Options) { options.MaxRequestBodyBytes = maximumRequestBodyBytes + 1 },
		"key active":      func(options *Options) { options.CapabilityKeyring.ActiveVersion = "missing" },
		"key size":        func(options *Options) { options.CapabilityKeyring.Keys["v1"] = []byte("short") },
		"key version": func(options *Options) {
			options.CapabilityKeyring.ActiveVersion = "v1.invalid"
			options.CapabilityKeyring.Keys["v1.invalid"] = options.CapabilityKeyring.Keys["v1"]
			delete(options.CapabilityKeyring.Keys, "v1")
		},
		"key lifetime": func(options *Options) { options.CapabilityKeyring.MaxLifetime = 0 },
	} {
		t.Run(name, func(t *testing.T) {
			candidate := valid
			candidate.CapabilityKeyring = cloneTestKeyring(valid.CapabilityKeyring)
			mutate(&candidate)
			if runtime, err := New(candidate); err == nil || runtime != nil {
				t.Fatalf("New() = %+v, %v", runtime, err)
			}
		})
	}
}

func TestNewAcceptsStableRegistryWithoutBuiltinCodecs(t *testing.T) {
	registry, err := protocolcodec.NewRegistry(codecStub{format: "custom.wire.v1"})
	if err != nil {
		t.Fatal(err)
	}
	options := validOptions(t)
	options.Codecs = registry
	runtime, err := New(options)
	if err != nil {
		t.Fatal(err)
	}
	if capabilities := runtime.WireCapabilities(); len(capabilities) != 1 || capabilities[0].Format != "custom.wire.v1" {
		t.Fatalf("protocol capabilities = %+v", capabilities)
	}
	if err := runtime.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestRuntimeRejectsUnsignedRequestBeforeSnapshotLookup(t *testing.T) {
	options := validOptions(t)
	runtime, err := New(options)
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Close()

	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"vllm-sr/blend"}`))
	response := httptest.NewRecorder()
	runtime.ServeHTTP(response, request)
	if response.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d", response.Code)
	}
	source := options.Snapshots.(*snapshotSourceStub)
	source.mu.Lock()
	requestCount := len(source.requests)
	source.mu.Unlock()
	if requestCount != 0 {
		t.Fatalf("snapshot lookups before capability verification = %d", requestCount)
	}
}

func TestRuntimeComposesPinnedSnapshotCredentialAdapterAndAccounting(t *testing.T) {
	now := time.Now().UTC()
	body := []byte(`{"model":"vllm-sr/blend","messages":[{"role":"user","content":"hello"}]}`)
	options := validOptions(t)
	options.Now = func() time.Time { return now }
	transport := options.Transport.(*transportStub)
	transport.roundTrip = func(request *http.Request) (*http.Response, error) {
		if request.URL.String() != "https://backend.example/v1/chat/completions" {
			t.Fatalf("backend URL = %q", request.URL.String())
		}
		if request.Header.Get("Authorization") != "Bearer provider-secret" {
			t.Fatalf("provider authorization = %q", request.Header.Get("Authorization"))
		}
		if request.Header.Get("X-Wire-Version") != "2026-08-22" {
			t.Fatalf("compiled connection header = %q", request.Header.Get("X-Wire-Version"))
		}
		wireBody, err := io.ReadAll(request.Body)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(wireBody), `"model":"provider-model"`) {
			t.Fatalf("provider model was not rewritten: %s", wireBody)
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": {"application/json"}},
			Body:       io.NopCloser(strings.NewReader(testChatResponseBody)),
		}, nil
	}
	token := signCapability(t, options, now, body)
	runtime, err := New(options)
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Close()

	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(body)))
	request.Header.Set(backendinvoker.DispatchCapabilityHeader, token)
	request.Header.Set("Authorization", "Bearer caller-secret")
	response := httptest.NewRecorder()
	runtime.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"content":"ok"`) {
		t.Fatalf("response = %d %s", response.Code, response.Body.String())
	}

	source := options.Snapshots.(*snapshotSourceStub)
	source.mu.Lock()
	requests := append([]snapshotRequest(nil), source.requests...)
	source.mu.Unlock()
	if len(requests) != 1 || requests[0] != (snapshotRequest{pin: routingcontext.Generation{
		NamespaceID: "namespace-1", QuotaPartition: "partition-1", PublicationID: "publication-1",
		RuntimeEpoch: 2, SnapshotRevision: 9, RoutingDigest: options.Snapshots.(*snapshotSourceStub).snapshot.Digest,
	}}) {
		t.Fatalf("snapshot requests = %+v", requests)
	}
	credentials := options.Credentials.(*credentialResolverStub)
	credentials.mu.Lock()
	pins, resolves := credentials.pins, credentials.resolves
	credentials.mu.Unlock()
	if pins != 1 || resolves != 1 {
		t.Fatalf("credential pin/resolve = %d/%d", pins, resolves)
	}
	journal := options.Journal.(*journalStub)
	journal.mu.Lock()
	dispatches, attempts, finished := journal.dispatches, journal.attempts, journal.finished
	journal.mu.Unlock()
	if dispatches != 1 || attempts != 1 || finished != 1 {
		t.Fatalf("journal dispatch/attempt/finish = %d/%d/%d", dispatches, attempts, finished)
	}
	observer := options.Observer.(*observerStub)
	observer.mu.Lock()
	observed := observer.count
	observer.mu.Unlock()
	if observed != 1 {
		t.Fatalf("observer calls = %d", observed)
	}
}

func TestRuntimeExecutesSignedCrossModelPriorityFallbackAndReportsOutcome(t *testing.T) {
	now := time.Now().UTC().Truncate(time.Second)
	body := []byte(`{"model":"vllm-sr/blend","messages":[{"role":"user","content":"hello"}]}`)
	options := validOptions(t)
	options.Now = func() time.Time { return now }
	snapshot := compiledFallbackSnapshot(t)
	options.Snapshots.(*snapshotSourceStub).snapshot = snapshot

	var destinations []string
	transport := options.Transport.(*transportStub)
	transport.roundTrip = func(request *http.Request) (*http.Response, error) {
		destinations = append(destinations, request.URL.Host)
		if request.URL.Host == "primary.example" {
			return nil, backendinvoker.NewKnownZeroTransportFailure(
				backendinvoker.FallbackUnavailable,
				errors.New("primary could not be reached before request write"),
			)
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": {"application/json"}},
			Body:       io.NopCloser(strings.NewReader(testChatResponseBody)),
		}, nil
	}

	capability := backendinvoker.DispatchCapability{
		NamespaceID: "namespace-1", QuotaPartition: "partition-1", RoutingRevision: 9,
		PublicationID: "publication-1", RuntimeEpoch: 2, RoutingDigest: snapshot.Digest,
		AdmissionID: "admission-1", AdmissionDigest: strings.Repeat("b", 64), RequestID: "request-1",
		Candidates: []backendinvoker.DispatchCandidate{
			{
				DispatchID: "dispatch-primary", DispatchType: "primary", Ordinal: 0,
				DispatchPlanDigest: strings.Repeat("a", 64), ModelID: "model-primary", ModelRevision: 1,
				Priority: 0,
			},
			{
				DispatchID: "dispatch-fallback", DispatchType: "fallback", Ordinal: 1,
				DispatchPlanDigest: strings.Repeat("c", 64), ModelID: "model-fallback", ModelRevision: 1,
				Priority: 1,
			},
		},
		Fallback:      backendinvoker.FallbackPolicy{On: []backendinvoker.FallbackTrigger{backendinvoker.FallbackUnavailable}},
		RequestDigest: backendinvoker.RequestDigest(http.MethodPost, "/v1/chat/completions", "", body),
		Method:        http.MethodPost, Path: "/v1/chat/completions", WireFormat: llmprotocol.OpenAIChatV1, Audience: options.Audience,
		IssuedAt: now.Unix(), ExpiresAt: now.Add(time.Minute).Unix(),
	}
	token, err := options.CapabilityKeyring.Sign(capability, now)
	if err != nil {
		t.Fatal(err)
	}
	runtime, err := New(options)
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Close()

	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(body)))
	request.Header.Set(backendinvoker.DispatchCapabilityHeader, token)
	response := httptest.NewRecorder()
	runtime.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("response = %d %s", response.Code, response.Body.String())
	}
	if got, want := destinations, []string{"primary.example", "fallback.example"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("backend destinations = %#v, want %#v", got, want)
	}
	outcome, err := options.CapabilityKeyring.VerifyOutcome(
		response.Header().Get(backendinvoker.DispatchOutcomeHeader), options.Audience, now,
	)
	if err != nil {
		t.Fatal(err)
	}
	if outcome.SelectedDispatchID != "dispatch-fallback" || len(outcome.Attempted) != 2 {
		t.Fatalf("dispatch outcome = %+v", outcome)
	}
	if outcome.Attempted[0].State != backendinvoker.AttemptKnownZero ||
		outcome.Attempted[0].FallbackTrigger != backendinvoker.FallbackUnavailable ||
		outcome.Attempted[1].State != backendinvoker.AttemptResponseStarted {
		t.Fatalf("fallback evidence = %+v", outcome.Attempted)
	}
}

func TestRuntimeFailsClosedWhenSnapshotOrCredentialCannotBePinned(t *testing.T) {
	now := time.Now().UTC()
	body := []byte(`{"model":"vllm-sr/blend"}`)
	for name, testCase := range map[string]struct {
		configure func(*Options)
		want      int
	}{
		"snapshot": {
			configure: func(options *Options) {
				options.Snapshots.(*snapshotSourceStub).err = errors.New("unavailable")
			},
			want: http.StatusNotFound,
		},
		"credential": {
			configure: func(options *Options) {
				options.Credentials.(*credentialResolverStub).pinErr = errors.New("disabled")
			},
			want: http.StatusBadGateway,
		},
	} {
		t.Run(name, func(t *testing.T) {
			options := validOptions(t)
			options.Now = func() time.Time { return now }
			testCase.configure(&options)
			token := signCapability(t, options, now, body)
			runtime, err := New(options)
			if err != nil {
				t.Fatal(err)
			}
			defer runtime.Close()
			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(body)))
			request.Header.Set(backendinvoker.DispatchCapabilityHeader, token)
			response := httptest.NewRecorder()
			runtime.ServeHTTP(response, request)
			if response.Code != testCase.want {
				t.Fatalf("status = %d, want %d", response.Code, testCase.want)
			}
		})
	}
}

func TestRuntimeOwnsCapabilityKeysAndClosesTransportOnce(t *testing.T) {
	options := validOptions(t)
	sourceKey := options.CapabilityKeyring.Keys["v1"]
	runtime, err := New(options)
	if err != nil {
		t.Fatal(err)
	}
	ownedKey := runtime.handler.Keyring.Keys["v1"]
	if len(ownedKey) == 0 || &ownedKey[0] == &sourceKey[0] {
		t.Fatal("capability keyring was not defensively copied")
	}
	capabilities := runtime.WireCapabilities()
	capabilities[0].Format = "mutated"
	if runtime.WireCapabilities()[0].Format == "mutated" {
		t.Fatal("protocol capabilities were not defensively copied")
	}
	if err := runtime.Close(); err != nil {
		t.Fatal(err)
	}
	for _, value := range ownedKey {
		if value != 0 {
			t.Fatal("owned capability key was not zeroed")
		}
	}
	for _, value := range sourceKey {
		if value == 0 {
			t.Fatal("borrowed source capability key was modified")
		}
	}
	transport := options.Transport.(*transportStub)
	transport.mu.Lock()
	closed := transport.closed
	transport.mu.Unlock()
	if closed != 1 {
		t.Fatalf("transport close count = %d", closed)
	}
	if err := runtime.Close(); err != nil {
		t.Fatal(err)
	}
	transport.mu.Lock()
	closed = transport.closed
	transport.mu.Unlock()
	if closed != 1 {
		t.Fatalf("idempotent transport close count = %d", closed)
	}
	response := httptest.NewRecorder()
	runtime.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil))
	if response.Code != http.StatusServiceUnavailable {
		t.Fatalf("closed runtime status = %d", response.Code)
	}
}

func TestCloseWaitsForActiveObservedResponse(t *testing.T) {
	now := time.Now().UTC()
	body := []byte(`{"model":"vllm-sr/blend","messages":[{"role":"user","content":"hello"}]}`)
	options := validOptions(t)
	options.Now = func() time.Time { return now }
	observer := options.Observer.(*observerStub)
	observer.started = make(chan struct{})
	observer.release = make(chan struct{})
	token := signCapability(t, options, now, body)
	runtime, err := New(options)
	if err != nil {
		t.Fatal(err)
	}

	serveDone := make(chan struct{})
	go func() {
		defer close(serveDone)
		request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(body)))
		request.Header.Set(backendinvoker.DispatchCapabilityHeader, token)
		runtime.ServeHTTP(httptest.NewRecorder(), request)
	}()
	select {
	case <-observer.started:
	case <-time.After(time.Second):
		t.Fatal("active response did not reach the observer")
	}
	closeDone := make(chan error, 1)
	go func() { closeDone <- runtime.Close() }()
	waitForQueuedWriter(t, &runtime.mu)
	select {
	case err := <-closeDone:
		t.Fatalf("Close() returned before active response completed: %v", err)
	default:
	}
	close(observer.release)
	<-serveDone
	if err := <-closeDone; err != nil {
		t.Fatal(err)
	}
}

func waitForQueuedWriter(t *testing.T, mutex *sync.RWMutex) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if !mutex.TryRLock() {
			return
		}
		mutex.RUnlock()
		time.Sleep(time.Millisecond)
	}
	t.Fatal("Close() never queued for the active dispatch")
}

func validOptions(t *testing.T) Options {
	t.Helper()
	snapshot := compiledSnapshot(t)
	transport := &transportStub{roundTrip: func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": {"application/json"}},
			Body:       io.NopCloser(strings.NewReader(testChatResponseBody)),
		}, nil
	}}
	return Options{
		Audience:          "backend-invoker",
		CapabilityKeyring: testKeyring(),
		Snapshots:         &snapshotSourceStub{snapshot: snapshot},
		Credentials: &credentialResolverStub{
			version: "version-1",
			credential: backendinvoker.Credential{
				Header: "Authorization", Prefix: "Bearer ", Secret: "provider-secret", Version: "version-1",
			},
		},
		Codecs:              protocolcodec.NewBuiltinRegistry(),
		Journal:             &journalStub{},
		Finalizer:           finalizerStub{},
		Observer:            &observerStub{},
		Transport:           transport,
		MaxRequestBodyBytes: 1 << 20,
	}
}

func testKeyring() backendinvoker.SigningKeyring {
	return backendinvoker.SigningKeyring{
		ActiveVersion: "v1",
		Keys:          map[string][]byte{"v1": []byte(strings.Repeat("k", sha256.Size))},
		MaxLifetime:   time.Minute,
	}
}

func cloneTestKeyring(source backendinvoker.SigningKeyring) backendinvoker.SigningKeyring {
	result := backendinvoker.SigningKeyring{
		ActiveVersion: source.ActiveVersion,
		Keys:          make(map[string][]byte, len(source.Keys)),
		MaxLifetime:   source.MaxLifetime,
	}
	for version, key := range source.Keys {
		result.Keys[version] = append([]byte(nil), key...)
	}
	return result
}

func signCapability(t *testing.T, options Options, now time.Time, body []byte) string {
	t.Helper()
	snapshot := options.Snapshots.(*snapshotSourceStub).snapshot
	capability := backendinvoker.DispatchCapability{
		NamespaceID: "namespace-1", QuotaPartition: "partition-1", RoutingRevision: 9,
		PublicationID: "publication-1", RuntimeEpoch: 2, RoutingDigest: snapshot.Digest,
		AdmissionID: "admission-1", AdmissionDigest: strings.Repeat("b", 64), RequestID: "request-1",
		Candidates: []backendinvoker.DispatchCandidate{{
			DispatchID: "dispatch-1", DispatchType: "primary", Ordinal: 0,
			DispatchPlanDigest: strings.Repeat("a", 64), ModelID: "model-1", ModelRevision: 3,
		}},
		RequestDigest: backendinvoker.RequestDigest(http.MethodPost, "/v1/chat/completions", "", body),
		Method:        http.MethodPost, Path: "/v1/chat/completions", WireFormat: llmprotocol.OpenAIChatV1, Audience: "backend-invoker",
		IssuedAt: now.Unix(), ExpiresAt: now.Add(time.Minute).Unix(),
	}
	token, err := options.CapabilityKeyring.Sign(capability, now)
	if err != nil {
		t.Fatal(err)
	}
	return token
}

func compiledSnapshot(t *testing.T) *routingsnapshot.Snapshot {
	t.Helper()
	document, err := json.Marshal(map[string]any{"kind": "recipe"})
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := routingsnapshot.Compile(routingsnapshot.Bundle{
		NamespaceID: "namespace-1", Revision: 9,
		Models: []routingsnapshot.Model{{
			ID: "model-1", Revision: 3,
			CatalogRevision: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			Name:            "remote/frontier",
			Execution: routingsnapshot.ModelExecution{
				MaxRetries: 1, RequestTimeout: "45s", StreamTimeout: "10m",
			},
			Backends: []routingsnapshot.Backend{{
				ID: "backend-1", ProviderID: "provider-openai",
				WireFormat: llmprotocol.OpenAIChatV1,
				Origin:     "https://backend.example/v1", ProviderModelID: "provider-model",
				ProviderCredentialID: "credential-1",
				Connection: routingsnapshot.BackendConnection{
					Path: "/chat/completions", Headers: map[string]string{"X-Wire-Version": "2026-08-22"},
				},
				Weight: "1",
			}},
		}},
		Recipes: []routingsnapshot.Recipe{{
			ID: "recipe-1", Revision: 1, Name: "balance",
			Decisions: []routingsnapshot.Decision{{ID: "decision-1", Name: "Simple", DispatchCardinality: routingsnapshot.DispatchCardinalitySingle}}, Document: document,
		}},
		Entrypoints: []routingsnapshot.Entrypoint{{
			ID: "entrypoint-1", Revision: 1, Name: "blend", Aliases: []string{"vllm-sr/blend"},
			Rules: []routingsnapshot.EntrypointRule{{
				ID: "rule-1", Name: "default", RecipeID: "recipe-1", RecipeRevision: 1,
				Assignments: map[string]routingsnapshot.AssignmentSet{
					"decision-1": {Models: []routingsnapshot.Assignment{{ModelID: "model-1", ModelRevision: 3, Weight: "1"}}},
				},
			}},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

func compiledFallbackSnapshot(t *testing.T) *routingsnapshot.Snapshot {
	t.Helper()
	document, err := json.Marshal(map[string]any{"kind": "recipe"})
	if err != nil {
		t.Fatal(err)
	}
	models := []routingsnapshot.Model{
		{
			ID: "model-primary", Revision: 1,
			CatalogRevision: "sha256:" + strings.Repeat("a", 64), Name: "remote/primary",
			Execution: routingsnapshot.ModelExecution{RequestTimeout: "45s", StreamTimeout: "10m"},
			Backends: []routingsnapshot.Backend{{
				ID: "backend-primary", ProviderID: "private", WireFormat: llmprotocol.OpenAIChatV1,
				Origin: "https://primary.example", ProviderModelID: "primary-provider-model",
				ProviderCredentialID: "credential-1",
				Connection:           routingsnapshot.BackendConnection{Path: "/v1/chat/completions"}, Weight: "1",
			}},
		},
		{
			ID: "model-fallback", Revision: 1,
			CatalogRevision: "sha256:" + strings.Repeat("b", 64), Name: "remote/fallback",
			Execution: routingsnapshot.ModelExecution{RequestTimeout: "45s", StreamTimeout: "10m"},
			Backends: []routingsnapshot.Backend{{
				ID: "backend-fallback", ProviderID: "private", WireFormat: llmprotocol.OpenAIChatV1,
				Origin: "https://fallback.example", ProviderModelID: "fallback-provider-model",
				ProviderCredentialID: "credential-1",
				Connection:           routingsnapshot.BackendConnection{Path: "/v1/chat/completions"}, Weight: "1",
			}},
		},
	}
	snapshot, err := routingsnapshot.Compile(routingsnapshot.Bundle{
		NamespaceID: "namespace-1", Revision: 9, Models: models,
		Recipes: []routingsnapshot.Recipe{{
			ID: "recipe-1", Revision: 1, Name: "balance", Document: document,
			Decisions: []routingsnapshot.Decision{{
				ID: "decision-1", Name: "Default", DispatchCardinality: routingsnapshot.DispatchCardinalitySingle,
			}},
		}},
		Entrypoints: []routingsnapshot.Entrypoint{{
			ID: "entrypoint-1", Revision: 1, Name: "blend", Aliases: []string{"vllm-sr/blend"},
			Rules: []routingsnapshot.EntrypointRule{{
				ID: "rule-1", Name: "default", RecipeID: "recipe-1", RecipeRevision: 1,
				Assignments: map[string]routingsnapshot.AssignmentSet{
					"decision-1": {
						Models: []routingsnapshot.Assignment{
							{ModelID: "model-primary", ModelRevision: 1, Priority: 0, Weight: "1"},
							{ModelID: "model-fallback", ModelRevision: 1, Priority: 1, Weight: "1"},
						},
						Fallback: &routingsnapshot.FallbackPolicy{Strategy: "priority", On: []string{"unavailable"}},
					},
				},
			}},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}
