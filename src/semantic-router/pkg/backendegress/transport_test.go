package backendegress

import (
	"context"
	"errors"
	"net"
	"net/netip"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type transportResolverStub struct {
	addresses []netip.Addr
	err       error
}

func (resolver transportResolverStub) LookupNetIP(context.Context, string, string) ([]netip.Addr, error) {
	return append([]netip.Addr(nil), resolver.addresses...), resolver.err
}

type transportJournalStub struct{}

func (transportJournalStub) BeginDispatch(context.Context, backendinvoker.Plan, time.Time) error {
	return nil
}

func (transportJournalStub) BeginAttempt(context.Context, backendinvoker.Plan, backendinvoker.Attempt) error {
	return nil
}

func (transportJournalStub) FinishAttempt(context.Context, backendinvoker.Plan, backendinvoker.AttemptResult) error {
	return nil
}

func TestTransportResolverFailureAuthorizesOnlyProvenKnownZeroFallback(t *testing.T) {
	for name, resolverError := range map[string]error{
		"unavailable": errors.New("resolver unavailable"),
		"timeout":     &net.DNSError{Err: "timeout", IsTimeout: true},
	} {
		t.Run(name, func(t *testing.T) {
			transport := testBackendTransport(t, transportResolverStub{err: resolverError})
			chain := testTransportPlanChain()
			if name == "timeout" {
				chain.Fallback.On = []backendinvoker.FallbackTrigger{backendinvoker.FallbackTimeout}
			}
			result, err := (&backendinvoker.Invoker{
				Transport: transport, Journal: transportJournalStub{},
			}).InvokeChain(context.Background(), chain)
			if err == nil || len(result.Outcomes) != 2 {
				t.Fatalf("InvokeChain() error=%v outcomes=%+v", err, result.Outcomes)
			}
			want := backendinvoker.FallbackUnavailable
			if name == "timeout" {
				want = backendinvoker.FallbackTimeout
			}
			if result.Outcomes[0].State != backendinvoker.AttemptKnownZero ||
				result.Outcomes[0].FallbackTrigger != want {
				t.Fatalf("first outcome = %+v", result.Outcomes[0])
			}
		})
	}
}

func TestTransportDoesNotFallbackWhenTriggerIsDisabled(t *testing.T) {
	transport := testBackendTransport(t, transportResolverStub{err: errors.New("resolver unavailable")})
	chain := testTransportPlanChain()
	chain.Fallback = backendinvoker.FallbackPolicy{}
	result, err := (&backendinvoker.Invoker{
		Transport: transport, Journal: transportJournalStub{},
	}).InvokeChain(context.Background(), chain)
	if err == nil || len(result.Outcomes) != 1 {
		t.Fatalf("InvokeChain() error=%v outcomes=%+v", err, result.Outcomes)
	}
}

func TestTransportCancellationRemainsUnknown(t *testing.T) {
	transport := testBackendTransport(t, transportResolverStub{err: context.Canceled})
	result, err := (&backendinvoker.Invoker{
		Transport: transport, Journal: transportJournalStub{},
	}).InvokeChain(context.Background(), testTransportPlanChain())
	if err == nil || len(result.Outcomes) != 1 || result.Outcomes[0].State != backendinvoker.AttemptUnknown {
		t.Fatalf("InvokeChain() error=%v outcomes=%+v", err, result.Outcomes)
	}
}

func TestTransportDialFailureIsKnownZeroBeforeHTTPWrite(t *testing.T) {
	transport := testBackendTransport(t, transportResolverStub{
		addresses: []netip.Addr{netip.MustParseAddr("203.0.113.8")},
	})
	transport.dial = func(context.Context, string, string) (net.Conn, error) {
		return nil, errors.New("connection refused")
	}
	result, err := (&backendinvoker.Invoker{
		Transport: transport, Journal: transportJournalStub{},
	}).InvokeChain(context.Background(), testTransportPlanChain())
	if err == nil || len(result.Outcomes) != 2 ||
		result.Outcomes[0].State != backendinvoker.AttemptKnownZero {
		t.Fatalf("InvokeChain() error=%v outcomes=%+v", err, result.Outcomes)
	}
}

func testBackendTransport(t *testing.T, resolver Resolver) *Transport {
	t.Helper()
	policy, err := Compile(Config{
		Version: "v1", Schemes: []string{"https"},
		Hosts: []HostConfig{{Host: "model.example", Ports: []uint16{443}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	transport, err := NewTransport(TransportOptions{Guard: Guard{Policy: policy, Resolver: resolver}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(transport.CloseIdleConnections)
	return transport
}

func testTransportPlanChain() backendinvoker.PlanChain {
	body := []byte(`{"model":"public","messages":[{"role":"user","content":"hello"}]}`)
	digest := backendinvoker.RequestDigest("POST", "/v1/chat/completions", "", body)
	base := backendinvoker.Plan{
		NamespaceID: "namespace", QuotaPartition: "partition", PublicationID: "publication",
		RuntimeEpoch: 1, RoutingRevision: 1, RoutingDigest: strings.Repeat("a", 64),
		AdmissionID: "admission", AdmissionDigest: strings.Repeat("b", 64), RequestID: "request",
		DispatchType: "primary", Method: "POST", Path: "/v1/chat/completions", SourceFormat: llmprotocol.OpenAIChatV1,
		Body: body, RequestDigest: digest,
		Execution: backendinvoker.Execution{RequestTimeout: time.Second},
		Backends: []backendinvoker.Backend{{
			ID: "backend", Origin: "https://model.example", ProviderID: "openai",
			WireFormat:      llmprotocol.OpenAIChatV1,
			ProviderModelID: "provider-model",
			Connection:      backendinvoker.Connection{Path: "/v1/chat/completions"}, Weight: 1,
		}},
	}
	first := base
	first.DispatchID = "dispatch-0"
	first.Ordinal = 0
	first.Priority = 0
	first.DispatchPlanDigest = strings.Repeat("c", 64)
	first.ModelID = "model-0"
	first.ModelRevision = 1
	second := base
	second.DispatchID = "dispatch-1"
	second.Ordinal = 1
	second.Priority = 1
	second.DispatchPlanDigest = strings.Repeat("d", 64)
	second.ModelID = "model-1"
	second.ModelRevision = 1
	return backendinvoker.PlanChain{
		Fallback:   backendinvoker.FallbackPolicy{On: []backendinvoker.FallbackTrigger{backendinvoker.FallbackUnavailable}},
		Candidates: []backendinvoker.Plan{first, second},
	}
}
