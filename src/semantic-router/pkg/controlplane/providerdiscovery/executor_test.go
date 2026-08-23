package providerdiscovery

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendegress"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
)

const (
	testNamespaceID     = "11111111-1111-4111-8111-111111111111"
	testCredentialID    = "22222222-2222-4222-8222-222222222222"
	testCredentialVerID = "33333333-3333-4333-8333-333333333333"
	testCatalogRevision = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	testAuthorityDigest = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
)

type captureTransport struct {
	request *http.Request
	status  int
	body    string
	calls   int
}

func (transport *captureTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	transport.calls++
	transport.request = request.Clone(request.Context())
	transport.request.Header = request.Header.Clone()
	status := transport.status
	if status == 0 {
		status = http.StatusOK
	}
	return &http.Response{
		StatusCode: status, Header: make(http.Header),
		Body: io.NopCloser(strings.NewReader(transport.body)), Request: request,
	}, nil
}

type credentialMetadataStub struct {
	credential providercredential.Credential
}

func (stub credentialMetadataStub) GetProviderCredential(
	_ context.Context,
	_ accesscontrol.NamespaceID,
	_ string,
) (providercredential.Credential, error) {
	return stub.credential, nil
}

type credentialResolverStub struct{}

func (credentialResolverStub) Pin(context.Context, string, string, string) (string, error) {
	return testCredentialVerID, nil
}

type anthropicCredentialResolverStub struct{}

func (anthropicCredentialResolverStub) Pin(context.Context, string, string, string) (string, error) {
	return testCredentialVerID, nil
}

func (anthropicCredentialResolverStub) ResolvePinned(
	context.Context,
	string,
	string,
	string,
	string,
) (backendinvoker.Credential, error) {
	return backendinvoker.Credential{
		Header: "X-Api-Key", Secret: "provider-secret", Version: testCredentialVerID,
	}, nil
}

func (credentialResolverStub) ResolvePinned(
	context.Context,
	string,
	string,
	string,
	string,
) (backendinvoker.Credential, error) {
	return backendinvoker.Credential{
		Header: "Authorization", Prefix: "Bearer ", Secret: "provider-secret", Version: testCredentialVerID,
	}, nil
}

func TestExecutorDiscoversWithCompiledProviderPlanAndIssuesSelectionClaim(t *testing.T) {
	now := time.Date(2026, 8, 22, 10, 0, 0, 0, time.UTC)
	transport := &captureTransport{body: `{"object":"list","data":[{"id":"model-b","object":"model"},{"id":"model-a","object":"model"}]}`}
	executor := testExecutor(t, transport, now)
	result, err := executor.Execute(context.Background(), ExecuteRequest{
		Plan: testPlan(), AuthorityDigest: testAuthorityDigest,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Models) != 1 || result.Models[0].ProviderModelID != "model-a" ||
		!result.HasMore || result.NextCursor != "model-a" || result.CatalogRevision != testCatalogRevision {
		t.Fatalf("discovery result = %+v", result)
	}
	if transport.calls != 1 || transport.request.URL.String() != "https://api.example.com/v1/models" ||
		transport.request.Header.Get("Authorization") != "Bearer provider-secret" ||
		transport.request.Header.Get("X-Provider-Version") != "1" {
		t.Fatalf("outbound request = %+v", transport.request)
	}
	if len(result.Models[0].Capabilities) != 2 || result.Models[0].CatalogItemID == "" {
		t.Fatalf("normalized model = %+v", result.Models[0])
	}
	selected, err := executor.Claims.VerifySelection(result.DiscoveryRevision, ClaimExpectation{
		NamespaceID: testNamespaceID, AuthorityDigest: testAuthorityDigest,
		CatalogRevision: testCatalogRevision, ProviderID: "provider-a",
	}, []string{result.Models[0].CatalogItemID}, now.Add(time.Minute))
	if err != nil || len(selected.Models) != 1 || selected.Models[0].ProviderModelID != "model-a" ||
		selected.Binding.Origin != "https://api.example.com/v1" ||
		selected.Binding.CredentialID != testCredentialID ||
		selected.Binding.CredentialVersion != testCredentialVerID {
		t.Fatalf("verified selection = %+v, err = %v", selected, err)
	}
}

func TestExecutorDiscoversAnthropicWireShapeWithOpaquePagination(t *testing.T) {
	now := time.Date(2026, 8, 22, 10, 0, 0, 0, time.UTC)
	transport := &captureTransport{body: `{
		"data":[{"id":"claude-model","display_name":"Claude Model","type":"model"}],
		"has_more":true,"last_id":"claude-model"
	}`}
	executor := testExecutor(t, transport, now)
	executor.Credentials = anthropicCredentialResolverStub{}
	metadata := executor.CredentialMetadata.(credentialMetadataStub)
	metadata.credential.CredentialAdapterID = "x-api-key"
	metadata.credential.NormalizedOrigin = "https://api.example.com"
	if err := metadata.credential.Validate(); err != nil {
		t.Fatalf("Anthropic credential fixture is invalid: %v", err)
	}
	executor.CredentialMetadata = metadata
	plan := testPlan()
	plan.DiscoveryAdapterID = anthropicModelsAdapterID
	plan.CredentialAdapterID = "x-api-key"
	plan.NormalizedOrigin = "https://api.example.com"
	plan.Path = "/v1/models"
	plan.Headers = map[string]string{"Anthropic-Version": "2023-06-01"}
	plan.ProviderCursor = "claude-before"
	if metadata.credential.NamespaceID != plan.NamespaceID ||
		metadata.credential.Status != providercredential.StatusActive ||
		metadata.credential.ProviderID != plan.ProviderID ||
		metadata.credential.CredentialMode != providercredential.Mode(plan.CredentialMode) ||
		metadata.credential.CredentialAdapterID != plan.CredentialAdapterID ||
		metadata.credential.NormalizedOrigin != plan.NormalizedOrigin {
		t.Fatalf("Anthropic credential fixture does not match discovery plan: credential=%+v plan=%+v", metadata.credential, plan)
	}
	result, err := executor.Execute(context.Background(), ExecuteRequest{
		Plan: plan, AuthorityDigest: testAuthorityDigest,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Models) != 1 || result.Models[0].ProviderModelID != "claude-model" ||
		!result.HasMore || result.NextCursor != "claude-model" {
		t.Fatalf("Anthropic discovery result = %+v", result)
	}
	if transport.request.URL.String() != "https://api.example.com/v1/models?after_id=claude-before&limit=1" ||
		transport.request.Header.Get("X-API-Key") != "provider-secret" ||
		transport.request.Header.Get("Anthropic-Version") != "2023-06-01" ||
		transport.request.Header.Get("Authorization") != "" {
		t.Fatalf("Anthropic discovery request = %s, %v", transport.request.URL, transport.request.Header)
	}
}

func TestExecutorRejectsCredentialBindingBeforeNetwork(t *testing.T) {
	for name, mutate := range map[string]func(*providercredential.Credential){
		"provider": func(value *providercredential.Credential) { value.ProviderID = "different-provider" },
		"mode":     func(value *providercredential.Credential) { value.CredentialMode = providercredential.ModeOptional },
	} {
		t.Run(name, func(t *testing.T) {
			now := time.Date(2026, 8, 22, 10, 0, 0, 0, time.UTC)
			transport := &captureTransport{body: `{"object":"list","data":[]}`}
			executor := testExecutor(t, transport, now)
			metadata := executor.CredentialMetadata.(credentialMetadataStub)
			mutate(&metadata.credential)
			executor.CredentialMetadata = metadata
			if _, err := executor.Execute(context.Background(), ExecuteRequest{
				Plan: testPlan(), AuthorityDigest: testAuthorityDigest,
			}); err == nil || transport.calls != 0 {
				t.Fatalf("credential mismatch error = %v, network calls = %d", err, transport.calls)
			}
		})
	}
}

func TestExecutorRejectsUpstreamErrorsAndOversizedResponses(t *testing.T) {
	now := time.Date(2026, 8, 22, 10, 0, 0, 0, time.UTC)
	for name, transport := range map[string]*captureTransport{
		"status":    {status: http.StatusUnauthorized, body: `{"secret":"must-not-surface"}`},
		"oversized": {body: strings.Repeat("x", maximumDiscoveryResponseBytes+1)},
	} {
		t.Run(name, func(t *testing.T) {
			executor := testExecutor(t, transport, now)
			if _, err := executor.Execute(context.Background(), ExecuteRequest{
				Plan: testPlan(), AuthorityDigest: testAuthorityDigest,
			}); err == nil || strings.Contains(err.Error(), "must-not-surface") {
				t.Fatalf("upstream error = %v", err)
			}
		})
	}
}

func testExecutor(t *testing.T, transport http.RoundTripper, now time.Time) Executor {
	t.Helper()
	registry, err := BuiltinRegistry()
	if err != nil {
		t.Fatal(err)
	}
	policy, err := backendegress.Compile(backendegress.Config{
		Version: "v1", Schemes: []string{"https"},
		Hosts: []backendegress.HostConfig{{Host: "api.example.com", Ports: []uint16{443}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	claims, err := NewClaimCodec(ClaimKeyset{
		ActiveKeyID: "current", Keys: map[string][]byte{"current": []byte(strings.Repeat("k", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	activeVersion := testCredentialVerID
	return Executor{
		Registry: registry, EgressPolicy: policy, Transport: transport, Claims: claims,
		CredentialMetadata: credentialMetadataStub{credential: providercredential.Credential{
			ID: testCredentialID, NamespaceID: testNamespaceID, Name: "Provider key",
			ProviderID: "provider-a", CredentialMode: providercredential.ModeRequired,
			CredentialAdapterID: "bearer",
			CatalogRevision:     testCatalogRevision, NormalizedOrigin: "https://api.example.com/v1",
			Status: providercredential.StatusActive, ActiveVersionID: &activeVersion, Revision: 1,
			CreatedAt: now.Add(-time.Hour), UpdatedAt: now.Add(-time.Hour),
		}},
		Credentials: credentialResolverStub{}, ClaimTTL: 5 * time.Minute,
		Now: func() time.Time { return now },
	}
}

func testPlan() providercatalog.DiscoveryPlan {
	return providercatalog.DiscoveryPlan{
		CatalogRevision: testCatalogRevision, NamespaceID: testNamespaceID,
		ProviderID: "provider-a", DiscoveryAdapterID: openAIModelsAdapterID,
		CredentialMode:      providercatalog.CredentialRequired,
		CredentialAdapterID: "bearer", CredentialID: testCredentialID,
		NormalizedOrigin: "https://api.example.com/v1", Path: "/models",
		Headers:      map[string]string{"X-Provider-Version": "1"},
		Capabilities: []string{"streaming", "tools"}, PageSize: 1,
	}
}
