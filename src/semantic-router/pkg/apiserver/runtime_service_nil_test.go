package apiserver

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// A resolver that fails construction returns a typed nil pointer in a
// non-nil interface. current must fall back, never panic: the nil check
// on the interface alone cannot see the typed nil, and the first field
// access on the nil receiver panics inside the service's own nil check.
func TestCurrentFallsBackOnTypedNilService(t *testing.T) {
	var nilService *services.ClassificationService
	live := newLiveClassificationService(
		services.NewPlaceholderClassificationService(),
		func() classificationService { return nilService },
		nil,
	)
	livePtr, ok := live.(*liveClassificationService)
	if !ok {
		t.Fatal("the live service must expose its current path")
	}
	svc := livePtr.current()
	if svc == nil {
		t.Fatal("current returned a nil service instead of the fallback")
	}
	// The placeholder fallback carries no classifier, so its eval degrades
	// to the graceful ErrClassifierUnavailable. Before the typed-nil guard
	// this path panicked inside the service's own nil check; the fix is
	// proven by the error, not by a served classification.
	_, err := svc.ClassifyIntentForEval(t.Context(), services.IntentRequest{Text: "hello"})
	if err == nil {
		t.Fatal("the placeholder eval served a classification without a classifier")
	}
	if !errors.Is(err, services.ErrClassifierUnavailable) {
		t.Fatalf("fallback eval returned an unexpected error: %v", err)
	}
}

func TestAcquireFallsBackOnTypedNilService(t *testing.T) {
	var nilService *services.ClassificationService
	live := newLiveClassificationService(
		services.NewPlaceholderClassificationService(),
		nil,
		func() (classificationService, func(), bool) {
			return nilService, func() {}, true
		},
	)
	livePtr, ok := live.(*liveClassificationService)
	if !ok {
		t.Fatal("the live service must expose its acquire path")
	}
	svc, release := livePtr.acquire()
	defer release()
	if svc == nil {
		t.Fatal("acquire returned a nil service instead of the fallback")
	}
}

func TestIsNilClassificationServiceClassifiesNilVariants(t *testing.T) {
	var nilPtr *services.ClassificationService
	if !isNilClassificationService(nil) {
		t.Fatal("a nil interface must be nil")
	}
	if !isNilClassificationService(nilPtr) {
		t.Fatal("a typed nil pointer must be nil")
	}
	if isNilClassificationService(services.NewPlaceholderClassificationService()) {
		t.Fatal("a real service must not be nil")
	}
}
