package extproc

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responsestore"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
)

func TestResponseObjectOwnerUsesOnlyTrustedInferenceContext(t *testing.T) {
	managed := &OpenAIRouter{Config: &config.RouterConfig{
		Access: config.AccessServiceConfig{Enabled: true},
	}}
	ctx := &RequestContext{
		Headers: map[string]string{
			"x-vsr-namespace-id": "spoofed-namespace",
			"x-vsr-api-key-id":   "spoofed-key",
			"x-vsr-user-id":      "spoofed-user",
		},
		InferenceAccess: &inferenceRequestAccess{tenant: accessruntime.TenantContext{
			NamespaceID: "trusted-namespace", APIKeyID: "trusted-key", UserID: "trusted-user",
		}},
		SemanticRequest: &llmprotocol.Request{Metadata: map[string]string{
			"namespace_id": "spoofed-body-namespace",
			"api_key_id":   "spoofed-body-key",
			"user_id":      "spoofed-body-user",
		}},
	}
	owner, ok := managed.responseObjectOwner(ctx)
	want := responseapi.ResponseOwner{
		Mode:        responseapi.ResponseOwnerAuthenticated,
		NamespaceID: "trusted-namespace", APIKeyID: "trusted-key", UserID: "trusted-user",
	}
	if !ok || owner != want {
		t.Fatalf("managed owner = (%+v, %t), want %+v", owner, ok, want)
	}

	publicTrace := testResponseObjectGeneration(t, "public-namespace")
	public := &OpenAIRouter{Config: &config.RouterConfig{}}
	publicOwner, ok := public.responseObjectOwner(&RequestContext{TraceContext: publicTrace})
	if !ok || publicOwner != (responseapi.ResponseOwner{
		Mode: responseapi.ResponseOwnerAnonymousPublicNamespace, NamespaceID: "public-namespace",
	}) {
		t.Fatalf("public owner = (%+v, %t)", publicOwner, ok)
	}

	if fallback, ok := managed.responseObjectOwner(&RequestContext{TraceContext: publicTrace}); ok || fallback.Valid() {
		t.Fatalf("managed access fell back to public owner: %+v", fallback)
	}
}

func TestResponseObjectEndpointsDoNotDiscloseCrossOwnerObjects(t *testing.T) {
	ownerA := responseapi.ResponseOwner{
		Mode:        responseapi.ResponseOwnerAuthenticated,
		NamespaceID: "namespace", APIKeyID: "key-a", UserID: "user-a",
	}
	ownerB := responseapi.ResponseOwner{
		Mode:        responseapi.ResponseOwnerAuthenticated,
		NamespaceID: "namespace", APIKeyID: "key-b", UserID: "user-b",
	}
	newFilter := func(t *testing.T, seed bool) (*ResponseAPIFilter, *responsestore.MemoryStore) {
		t.Helper()
		store, err := responsestore.NewMemoryStore(responsestore.StoreConfig{Enabled: true})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = store.Close() })
		if seed {
			err = store.StoreResponse(context.Background(), ownerA, &responseapi.StoredResponse{
				Owner: ownerA, ID: "resp_private", Status: responseapi.StatusCompleted,
				Input: []responseapi.InputItem{{ID: "item_private", Type: responseapi.ItemTypeMessage}},
			})
			if err != nil {
				t.Fatal(err)
			}
		}
		return NewResponseAPIFilter(store), store
	}

	type endpointResult struct {
		status int
		body   []byte
	}
	for _, operation := range []struct {
		name string
		call func(*ResponseAPIFilter) endpointResult
	}{
		{name: "get", call: func(filter *ResponseAPIFilter) endpointResult {
			response, _ := filter.HandleGetResponse(context.Background(), ownerB, "resp_private")
			return endpointResult{status: int(response.GetImmediateResponse().GetStatus().GetCode()), body: response.GetImmediateResponse().GetBody()}
		}},
		{name: "input_items", call: func(filter *ResponseAPIFilter) endpointResult {
			response, _ := filter.HandleGetInputItems(context.Background(), ownerB, "resp_private")
			return endpointResult{status: int(response.GetImmediateResponse().GetStatus().GetCode()), body: response.GetImmediateResponse().GetBody()}
		}},
		{name: "delete", call: func(filter *ResponseAPIFilter) endpointResult {
			response, _ := filter.HandleDeleteResponse(context.Background(), ownerB, "resp_private")
			return endpointResult{status: int(response.GetImmediateResponse().GetStatus().GetCode()), body: response.GetImmediateResponse().GetBody()}
		}},
	} {
		t.Run(operation.name, func(t *testing.T) {
			seeded, seededStore := newFilter(t, true)
			empty, _ := newFilter(t, false)
			crossOwner := operation.call(seeded)
			unknown := operation.call(empty)
			if crossOwner.status != 404 || unknown.status != 404 || !bytes.Equal(crossOwner.body, unknown.body) {
				t.Fatalf("cross-owner result %+v differs from unknown result %+v", crossOwner, unknown)
			}
			if bytes.Contains(crossOwner.body, []byte(ownerA.UserID)) || bytes.Contains(crossOwner.body, []byte(ownerA.APIKeyID)) {
				t.Fatalf("not-found response disclosed owner: %s", crossOwner.body)
			}
			if _, err := seededStore.GetResponse(context.Background(), ownerA, "resp_private"); err != nil {
				t.Fatalf("cross-owner %s changed owner A object: %v", operation.name, err)
			}
		})
	}
}

func TestNonStreamingProductionEdgeRetainsStandaloneResponse(t *testing.T) {
	store, err := responsestore.NewMemoryStore(responsestore.StoreConfig{Enabled: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	router := &OpenAIRouter{
		Config: &config.RouterConfig{}, ResponseAPIFilter: NewResponseAPIFilter(store),
	}
	traceContext := testResponseObjectGeneration(t, "standalone-namespace")
	ctx := &RequestContext{
		TraceContext: traceContext, SourceFormat: llmprotocol.OpenAIResponsesV1,
		RequestID: "retention-request", UpstreamStatusCode: 200,
	}
	requestBody := []byte(`{"model":"public-model","input":"hello","metadata":{"purpose":"retention"}}`)
	if _, early := router.prepareProtocolRequest(requestBody, ctx); early != nil {
		t.Fatal("request preparation returned an error response")
	}

	semantic := llmprotocol.Response{
		Generation: 1, ID: "resp_retained", CreatedAt: time.Unix(100, 0).UTC(), Model: "public-model",
		Output: []llmprotocol.OutputItem{{
			ID: "item_output", Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "retained answer"}},
		}},
		StopReason: llmprotocol.StopEndTurn, Usage: authoritativeZeroUsage(),
	}
	encoded, err := router.encodeClientResponse(semantic, ctx)
	if err != nil {
		t.Fatal(err)
	}
	ctx.SemanticResponse = nil
	router.handleNonStreamingResponseBody(encoded, ctx, 0)

	owner := responseapi.ResponseOwner{
		Mode: responseapi.ResponseOwnerAnonymousPublicNamespace, NamespaceID: "standalone-namespace",
	}
	stored, err := store.GetResponse(context.Background(), owner, semantic.ID)
	if err != nil {
		t.Fatalf("production response edge did not call StoreResponse: %v", err)
	}
	if stored.Owner != owner || stored.OutputText != "retained answer" ||
		len(stored.Input) != 1 || string(stored.Input[0].Content) != `"hello"` ||
		stored.Metadata["purpose"] != "retention" {
		t.Fatalf("stored standalone response = %+v", stored)
	}
	publicResponse, err := router.ResponseAPIFilter.HandleGetResponse(context.Background(), owner, semantic.ID)
	if err != nil || int(publicResponse.GetImmediateResponse().GetStatus().GetCode()) != 200 {
		t.Fatalf("standalone retained GET = (%v, %v)", publicResponse, err)
	}
	otherNamespace := owner
	otherNamespace.NamespaceID = "other-standalone-namespace"
	if _, err := store.GetResponse(context.Background(), otherNamespace, semantic.ID); !errors.Is(err, responsestore.ErrNotFound) {
		t.Fatalf("standalone namespace crossover error = %v", err)
	}
}

func TestCachedResponsesRetentionUsesRetrievableResponseID(t *testing.T) {
	response := &llmprotocol.Response{}
	refreshCachedSemanticResponse(response, &RequestContext{
		RequestID: "cache-retention", SourceFormat: llmprotocol.OpenAIResponsesV1,
	})
	if !strings.HasPrefix(response.ID, "resp_") || extractResponseIDFromPath("/v1/responses/"+response.ID) != response.ID {
		t.Fatalf("cached Responses ID %q is not retrievable", response.ID)
	}
}

func testResponseObjectGeneration(t *testing.T, namespaceID string) context.Context {
	t.Helper()
	ctx, err := routingcontext.WithGeneration(context.Background(), routingcontext.Generation{
		NamespaceID: namespaceID, QuotaPartition: "public-partition", PublicationID: "publication",
		RuntimeEpoch: 1, SnapshotRevision: 1, RoutingDigest: strings.Repeat("a", 64),
	})
	if err != nil {
		t.Fatal(err)
	}
	return ctx
}
