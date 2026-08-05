package apiserver

import (
	"testing"

	routerruntime "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// TestBuildClassificationResolverReturnsUntypedNil guards against the
// typed-nil interface bug: before the runtime finishes initializing, the
// registry holds a nil *services.ClassificationService. If the resolver
// returns that pointer directly, it becomes a non-nil interface value,
// bypasses the nil check in liveClassificationService.current(), and the
// first classification request panics with a nil receiver.
func TestBuildClassificationResolverReturnsUntypedNil(t *testing.T) {
	resolver := buildClassificationResolver(&routerruntime.Registry{})
	if svc := resolver(); svc != nil {
		t.Fatalf("resolver with empty registry must return untyped nil, got %T", svc)
	}
}

// TestLiveClassificationServiceFallsBackDuringStartup exercises the full
// request path that panicked: a live service whose resolver has nothing yet
// must serve the placeholder response instead of panicking.
func TestLiveClassificationServiceFallsBackDuringStartup(t *testing.T) {
	svc := newLiveClassificationService(
		nil,
		buildClassificationResolver(&routerruntime.Registry{}),
	)

	resp, err := svc.ClassifyIntent(services.IntentRequest{Text: "What is 2+2?"})
	if err != nil {
		t.Fatalf("ClassifyIntent during startup: unexpected error: %v", err)
	}
	if resp == nil {
		t.Fatal("ClassifyIntent during startup: expected placeholder response, got nil")
	}
}
