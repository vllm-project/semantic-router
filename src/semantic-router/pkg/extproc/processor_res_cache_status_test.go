package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A non-2xx upstream response must NOT be written to the semantic cache.
// Otherwise a later semantically-similar request hits the cached error body and
// CreateCacheHitResponse replays it to the client as an HTTP 200 success
// (cache poisoning). See processor_res_cache.go / req_filter_cache.go.

func TestShouldSkipCacheWriteForStatus(t *testing.T) {
	cases := []struct {
		name     string
		status   int
		wantSkip bool
	}{
		{"unknown_zero_does_not_block", 0, false},
		{"ok_200", 200, false},
		{"no_content_204", 204, false},
		{"max_2xx_299", 299, false},
		{"redirect_301", 301, true},
		{"bad_request_400", 400, true},
		{"rate_limited_429", 429, true},
		{"server_error_500", 500, true},
		{"unavailable_503", 503, true},
		{"informational_100", 100, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			skip, reason := shouldSkipCacheWriteForStatus(&RequestContext{UpstreamStatusCode: tc.status})
			if skip != tc.wantSkip {
				t.Fatalf("status %d: got skip=%v, want %v", tc.status, skip, tc.wantSkip)
			}
			if skip && reason == "" {
				t.Fatalf("status %d: skip must carry a non-empty reason", tc.status)
			}
		})
	}
	if skip, _ := shouldSkipCacheWriteForStatus(nil); skip {
		t.Fatal("nil ctx must not block cache write")
	}
}

func statusCacheRouter() (*mockStreamingCache, *OpenAIRouter, *config.Decision) {
	return cacheRouterForDecision(retentionCacheDecision("cache-decision", true))
}

func TestUpdateResponseCacheSkipsNon2xx(t *testing.T) {
	mockCache, router, decision := statusCacheRouter()
	ctx := withSelectedDecision(&RequestContext{
		RequestID:          "req-status-400",
		UpstreamStatusCode: 400,
	}, decision)
	router.updateResponseCache(ctx, []byte(`{"error":{"message":"bad model"}}`))
	if mockCache.addEntryCalled {
		t.Fatal("a non-2xx upstream response must not be cached (cache poisoning)")
	}
}

func TestUpdateResponseCacheWritesOn2xx(t *testing.T) {
	mockCache, router, decision := statusCacheRouter()
	ctx := withSelectedDecision(&RequestContext{
		RequestID:          "req-status-200",
		UpstreamStatusCode: 200,
		RequestModel:       "test",
		RequestQuery:       "hello",
		SemanticRequest:    testNeutralRequest("test", "hello"),
	}, decision)
	router.updateResponseCache(ctx, []byte(`{"choices":[]}`))
	if !mockCache.addEntryCalled {
		t.Fatal("a 2xx upstream response must still be cached")
	}
}

func TestUpdateResponseCacheWritesWhenStatusUnknown(t *testing.T) {
	mockCache, router, decision := statusCacheRouter()
	ctx := withSelectedDecision(&RequestContext{
		RequestID:       "req-status-unknown",
		RequestModel:    "test",
		RequestQuery:    "hello",
		SemanticRequest: testNeutralRequest("test", "hello"),
		// UpstreamStatusCode left 0: never observed (e.g. headers not processed).
	}, decision)
	router.updateResponseCache(ctx, []byte(`{"choices":[]}`))
	if !mockCache.addEntryCalled {
		t.Fatal("unknown upstream status must not block caching (backward compatible)")
	}
}
