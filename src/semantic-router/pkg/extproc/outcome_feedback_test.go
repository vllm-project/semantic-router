package extproc

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/outcomefeedback"
)

type outcomeRuntimeStub struct {
	receipt outcomefeedback.Receipt
	err     error
	calls   int
	caller  outcomefeedback.Caller
	key     string
	request outcomefeedback.Request
}

func (stub *outcomeRuntimeStub) Submit(
	_ context.Context,
	caller outcomefeedback.Caller,
	key string,
	request outcomefeedback.Request,
) (outcomefeedback.Receipt, error) {
	stub.calls++
	stub.caller, stub.key, stub.request = caller, key, request
	return stub.receipt, stub.err
}

func TestPublicOutcomeFeedbackUsesAuthenticatedInferenceIdentity(t *testing.T) {
	runtime := &outcomeRuntimeStub{receipt: outcomefeedback.Receipt{
		ID: "receipt-001", ReplayID: "replay-001", ProjectionRevision: 4,
		CreatedAt: time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC),
	}}
	router := &OpenAIRouter{
		Config:          &config.RouterConfig{Access: config.AccessServiceConfig{Enabled: true}},
		OutcomeFeedback: runtime,
	}
	tenant := inferenceTestTenant("")
	tenant.TeamID = "00000000-0000-4000-8000-000000000104"
	ctx := &RequestContext{
		Headers: make(map[string]string),
		TraceContext: withInferenceAuthentication(context.Background(), accessruntime.Authentication{
			Tenant: tenant, Source: accessruntime.AuthenticationSourceDelegated,
		}),
	}
	headers := newRequestHeaders("POST", publicOutcomeFeedbackPath)
	headers.RequestHeaders.Headers.Headers = append(headers.RequestHeaders.Headers.Headers,
		&core.HeaderValue{Key: "authorization", RawValue: []byte("Bearer delegated-secret")},
		&core.HeaderValue{Key: "idempotency-key", RawValue: []byte("feedback-001")},
	)
	headerResponse, err := router.handleRequestHeaders(headers, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if headerResponse.GetRequestHeaders() == nil || ctx.OutcomeFeedback == nil {
		t.Fatalf("header response/state = %+v / %+v", headerResponse, ctx.OutcomeFeedback)
	}
	removed := headerResponse.GetRequestHeaders().Response.GetHeaderMutation().RemoveHeaders
	if !slices.Contains(removed, "authorization") || !slices.Contains(removed, "idempotency-key") {
		t.Fatalf("sensitive headers not removed: %v", removed)
	}

	body := []byte(`{"replay_id":"replay-001","target":"route","target_ref":"balanced","verdict":"good_fit"}`)
	response, err := router.handleRequestBodyDispatch(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: body, EndOfStream: true},
	}, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if response.GetImmediateResponse() == nil || int(response.GetImmediateResponse().GetStatus().GetCode()) != 201 {
		t.Fatalf("outcome response = %+v", response)
	}
	if runtime.calls != 1 || runtime.key != "feedback-001" || runtime.request.ReplayID != "replay-001" {
		t.Fatalf("outcome submission = %+v", runtime)
	}
	wantCaller := outcomefeedback.Caller{
		NamespaceID: tenant.NamespaceID, APIKeyID: tenant.APIKeyID,
		UserID: tenant.UserID, TeamID: tenant.TeamID,
		Source: outcomefeedback.SourceDelegated,
	}
	if runtime.caller != wantCaller {
		t.Fatalf("derived caller = %+v, want %+v", runtime.caller, wantCaller)
	}
}

func TestPublicOutcomeFeedbackRejectsMissingAuthenticationAndBodyProvenance(t *testing.T) {
	runtime := &outcomeRuntimeStub{}
	router := &OpenAIRouter{
		Config:          &config.RouterConfig{Access: config.AccessServiceConfig{Enabled: true}},
		OutcomeFeedback: runtime,
	}
	unauthenticated := &RequestContext{Headers: make(map[string]string), TraceContext: context.Background()}
	headers := newRequestHeaders("POST", publicOutcomeFeedbackPath)
	headers.RequestHeaders.Headers.Headers = append(headers.RequestHeaders.Headers.Headers,
		&core.HeaderValue{Key: "idempotency-key", RawValue: []byte("feedback-001")},
	)
	response, err := router.handleRequestHeaders(headers, unauthenticated)
	if err != nil {
		t.Fatal(err)
	}
	if status := int(response.GetImmediateResponse().GetStatus().GetCode()); status != 401 {
		t.Fatalf("unauthenticated status = %d, want 401", status)
	}

	tenant := inferenceTestTenant("")
	ctx := &RequestContext{
		Headers: make(map[string]string),
		TraceContext: withInferenceAuthentication(context.Background(), accessruntime.Authentication{
			Tenant: tenant, Source: accessruntime.AuthenticationSourceAPIKey,
		}),
	}
	headers = newRequestHeaders("POST", publicOutcomeFeedbackPath)
	headers.RequestHeaders.Headers.Headers = append(headers.RequestHeaders.Headers.Headers,
		&core.HeaderValue{Key: "idempotency-key", RawValue: []byte("feedback-002")},
	)
	if response, err = router.handleRequestHeaders(headers, ctx); err != nil || response.GetRequestHeaders() == nil {
		t.Fatalf("authenticated headers = (%+v, %v)", response, err)
	}
	spoofed := []byte(`{
  "replay_id":"replay-001","target":"route","verdict":"good_fit",
  "namespace_id":"00000000-0000-4000-8000-000000000999"
}`)
	response, err = router.handleRequestBodyDispatch(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: spoofed, EndOfStream: true},
	}, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if status := int(response.GetImmediateResponse().GetStatus().GetCode()); status != 400 {
		t.Fatalf("caller-provenance status = %d, want 400", status)
	}
	if runtime.calls != 0 {
		t.Fatalf("runtime calls = %d, want zero", runtime.calls)
	}
}

func TestPublicOutcomeFeedbackErrorContractIsBoundedAndNondisclosing(t *testing.T) {
	router := &OpenAIRouter{}
	missing := router.outcomeFeedbackResponseForError(outcomefeedback.ErrNotFound)
	wrongOwner := router.outcomeFeedbackResponseForError(errors.Join(outcomefeedback.ErrNotFound, errors.New("owner mismatch")))
	wrongModel := router.outcomeFeedbackResponseForError(errors.Join(outcomefeedback.ErrNotFound, errors.New("model mismatch")))
	for name, response := range map[string]*ext_proc.ProcessingResponse{
		"missing": missing, "owner": wrongOwner, "model": wrongModel,
	} {
		if response.GetImmediateResponse() == nil || int(response.GetImmediateResponse().GetStatus().GetCode()) != 404 {
			t.Fatalf("%s response = %+v", name, response)
		}
		if string(response.GetImmediateResponse().GetBody()) != string(missing.GetImmediateResponse().GetBody()) {
			t.Fatalf("%s response disclosed claim type: %s", name, response.GetImmediateResponse().GetBody())
		}
	}
	limited := router.outcomeFeedbackResponseForError(&outcomefeedback.RateLimitError{RetryAfter: 1200 * time.Millisecond})
	if int(limited.GetImmediateResponse().GetStatus().GetCode()) != 429 ||
		headerFromImmediateResponse(limited, "retry-after") != "2" {
		t.Fatalf("rate-limit response = %+v", limited)
	}
}

func TestPublicOutcomeFeedbackRejectsOversizedBodyBeforeStorage(t *testing.T) {
	runtime := &outcomeRuntimeStub{}
	router := &OpenAIRouter{
		Config:          &config.RouterConfig{Access: config.AccessServiceConfig{Enabled: true}},
		OutcomeFeedback: runtime,
	}
	ctx := &RequestContext{
		Headers: make(map[string]string),
		TraceContext: withInferenceAuthentication(context.Background(), accessruntime.Authentication{
			Tenant: inferenceTestTenant(""), Source: accessruntime.AuthenticationSourceAPIKey,
		}),
	}
	headers := newRequestHeaders("POST", publicOutcomeFeedbackPath)
	headers.RequestHeaders.Headers.Headers = append(headers.RequestHeaders.Headers.Headers,
		&core.HeaderValue{Key: "idempotency-key", RawValue: []byte("feedback-large")},
	)
	if response, err := router.handleRequestHeaders(headers, ctx); err != nil || response.GetRequestHeaders() == nil {
		t.Fatalf("headers = (%+v, %v)", response, err)
	}
	response, err := router.handleRequestBodyDispatch(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: make([]byte, outcomefeedback.MaximumBodyBytes+1), EndOfStream: true},
	}, ctx)
	if err != nil {
		t.Fatal(err)
	}
	if status := int(response.GetImmediateResponse().GetStatus().GetCode()); status != 413 {
		t.Fatalf("oversized status = %d, want 413", status)
	}
	if runtime.calls != 0 {
		t.Fatalf("runtime calls = %d, want zero", runtime.calls)
	}
}

func headerFromImmediateResponse(response *ext_proc.ProcessingResponse, name string) string {
	if response == nil || response.GetImmediateResponse() == nil || response.GetImmediateResponse().Headers == nil {
		return ""
	}
	for _, header := range response.GetImmediateResponse().Headers.SetHeaders {
		if header.GetHeader().GetKey() == name {
			return extractHeaderValue(header.GetHeader())
		}
	}
	return ""
}
