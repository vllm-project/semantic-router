package kvtransfer

import (
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
)

func testModel(name, revision string) ModelIdentity {
	return ModelIdentity{
		Model:             name,
		WeightRevision:    revision,
		Tokenizer:         "Qwen/Qwen3-Tokenizer",
		TokenizerRevision: strings.Repeat("c", 40),
		Precision:         "bf16",
		TensorParallel:    1,
		KVHeads:           8,
		HeadDim:           128,
		HeadOrder:         "contiguous",
	}
}

func testHandoff(t *testing.T) (Request, SourceCache, Mapper) {
	t.Helper()
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-only-kv-scope-secret")
	now := time.Date(2026, time.October, 4, 0, 0, 0, 0, time.UTC)
	sourceModel := testModel("Qwen/Qwen3-14B", strings.Repeat("a", 40))
	targetModel := testModel("Qwen/Qwen3-32B", strings.Repeat("b", 40))
	return Request{
			AuthenticatedPrincipal: "tenant-a",
			SessionProvenance:      "header",
			SessionID:              "session-1",
			Target:                 targetModel,
			SourceCanExport:        true,
			TargetCanLoad:          true,
			Now:                    now,
		}, SourceCache{
			Namespace: cache.UserScopeNamespace("tenant-a"),
			SessionID: "session-1",
			CacheID:   "opaque-cache-1",
			Endpoint:  "10.0.1.5:8000",
			Model:     sourceModel,
			ExpiresAt: now.Add(time.Minute),
		}, Mapper{
			ID:     "published-mapper-1",
			Source: sourceModel,
			Target: targetModel,
		}
}

func TestPlanHandoffReturnsCandidateForExactIdentity(t *testing.T) {
	request, source, mapper := testHandoff(t)
	hint, reason := PlanHandoff(request, source, mapper)
	if reason != ReasonEligible || hint == nil {
		t.Fatalf("PlanHandoff() = (%v, %q), want eligible hint", hint, reason)
	}
	if hint.Namespace != source.Namespace || hint.CacheID != source.CacheID ||
		hint.MapperID != mapper.ID || hint.SourceEndpoint != source.Endpoint {
		t.Fatalf("PlanHandoff() hint = %+v, want source-scoped mapper hint", hint)
	}
}

func TestPlanHandoffRejectsUnsafeCandidates(t *testing.T) {
	cases := []struct {
		name   string
		change func(*Request, *SourceCache, *Mapper)
		want   Reason
	}{
		{"tenant mismatch", func(_ *Request, source *SourceCache, _ *Mapper) {
			source.Namespace = cache.UserScopeNamespace("tenant-b")
		}, ReasonScopeMismatch},
		{"missing authenticated principal", func(request *Request, _ *SourceCache, _ *Mapper) {
			request.AuthenticatedPrincipal = ""
		}, ReasonUntrustedSession},
		{"untrusted derived session", func(request *Request, _ *SourceCache, _ *Mapper) {
			request.SessionProvenance = "message_hash"
		}, ReasonUntrustedSession},
		{"unknown session provenance", func(request *Request, _ *SourceCache, _ *Mapper) {
			request.SessionProvenance = "unknown"
		}, ReasonUntrustedSession},
		{"different authenticated principal", func(request *Request, _ *SourceCache, _ *Mapper) {
			request.AuthenticatedPrincipal = "tenant-b"
		}, ReasonScopeMismatch},
		{"session mismatch", func(_ *Request, source *SourceCache, _ *Mapper) {
			source.SessionID = "session-2"
		}, ReasonScopeMismatch},
		{"expired source", func(request *Request, source *SourceCache, _ *Mapper) {
			source.ExpiresAt = request.Now
		}, ReasonStaleSource},
		{"producer cannot export", func(request *Request, _ *SourceCache, _ *Mapper) {
			request.SourceCanExport = false
		}, ReasonUnavailableBackend},
		{"consumer cannot load", func(request *Request, _ *SourceCache, _ *Mapper) {
			request.TargetCanLoad = false
		}, ReasonUnavailableBackend},
		{"routing alias instead of served model", func(_ *Request, source *SourceCache, _ *Mapper) {
			source.Model.Model = "fast-model"
		}, ReasonIdentityMismatch},
		{"wrong weight revision", func(_ *Request, source *SourceCache, _ *Mapper) {
			source.Model.WeightRevision = strings.Repeat("d", 40)
		}, ReasonIdentityMismatch},
		{"wrong tokenizer", func(request *Request, _ *SourceCache, _ *Mapper) {
			request.Target.TokenizerRevision = strings.Repeat("d", 40)
		}, ReasonIdentityMismatch},
		{"mutable weight revision", func(request *Request, _ *SourceCache, _ *Mapper) {
			request.Target.WeightRevision = "main"
		}, ReasonUnpinnedIdentity},
		{"mutable tokenizer revision", func(request *Request, _ *SourceCache, _ *Mapper) {
			request.Target.TokenizerRevision = "main"
		}, ReasonUnpinnedIdentity},
		{"wrong precision", func(request *Request, _ *SourceCache, _ *Mapper) {
			request.Target.Precision = "fp16"
		}, ReasonIdentityMismatch},
		{"wrong TP", func(request *Request, _ *SourceCache, _ *Mapper) {
			request.Target.TensorParallel = 2
		}, ReasonIdentityMismatch},
		{"wrong KV geometry", func(request *Request, _ *SourceCache, _ *Mapper) {
			request.Target.KVHeads = 4
		}, ReasonIdentityMismatch},
		{"adapter mismatch", func(request *Request, _ *SourceCache, _ *Mapper) {
			request.Target.AdapterID = "adapter-a"
		}, ReasonIdentityMismatch},
		{"matching adapter is not supported", func(request *Request, _ *SourceCache, mapper *Mapper) {
			request.Target.AdapterID = "adapter-a"
			mapper.Target.AdapterID = "adapter-a"
		}, ReasonUnsupportedServing},
		{"matching fp16 artifact is not supported", func(request *Request, _ *SourceCache, mapper *Mapper) {
			request.Target.Precision = "fp16"
			mapper.Target.Precision = "fp16"
		}, ReasonUnsupportedServing},
		{"different source and target tokenizers", func(request *Request, _ *SourceCache, mapper *Mapper) {
			request.Target.Tokenizer = "other/tokenizer"
			mapper.Target.Tokenizer = "other/tokenizer"
		}, ReasonVocabularyMismatch},
		{"same model", func(request *Request, source *SourceCache, mapper *Mapper) {
			request.Target = source.Model
			mapper.Target = source.Model
		}, ReasonSameModel},
		{"missing actual endpoint", func(_ *Request, source *SourceCache, _ *Mapper) {
			source.Endpoint = ""
		}, ReasonMissingInput},
		{"missing mapper ID", func(_ *Request, _ *SourceCache, mapper *Mapper) {
			mapper.ID = ""
		}, ReasonMissingInput},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			request, source, mapper := testHandoff(t)
			test.change(&request, &source, &mapper)
			hint, reason := PlanHandoff(request, source, mapper)
			if hint != nil || reason != test.want {
				t.Fatalf("PlanHandoff() = (%v, %q), want (nil, %q)", hint, reason, test.want)
			}
		})
	}
}

func TestPlanHandoffRequiresConfiguredScopeSecret(t *testing.T) {
	request, source, mapper := testHandoff(t)
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "")
	if hint, reason := PlanHandoff(request, source, mapper); hint != nil || reason != ReasonUntrustedSession {
		t.Fatalf("PlanHandoff() = (%v, %q), want untrusted session", hint, reason)
	}
}

func TestPlanHandoffAcceptsRetainedResponseSession(t *testing.T) {
	request, source, mapper := testHandoff(t)
	request.SessionProvenance = "response_api"
	if hint, reason := PlanHandoff(request, source, mapper); hint == nil || reason != ReasonEligible {
		t.Fatalf("PlanHandoff() = (%v, %q), want eligible hint", hint, reason)
	}
}
