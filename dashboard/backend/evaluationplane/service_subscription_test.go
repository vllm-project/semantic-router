package evaluationplane

import (
	"errors"
	"testing"
)

func TestSubscribeEnforcesPerRunAndGlobalBounds(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRun(t.Context(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	unsubscribers := make([]func(), 0, maxSubscribersPerRun)
	for range maxSubscribersPerRun {
		_, unsubscribe, subscribeErr := service.Subscribe(run.ID)
		if subscribeErr != nil {
			t.Fatalf("Subscribe below per-run limit: %v", subscribeErr)
		}
		unsubscribers = append(unsubscribers, unsubscribe)
	}
	if _, _, err := service.Subscribe(run.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("Subscribe above per-run limit error=%v, want ErrConflict", err)
	}
	for _, unsubscribe := range unsubscribers {
		unsubscribe()
		unsubscribe() // Idempotent cleanup must not underflow the global bound.
	}
	if service.subscriberCount != 0 {
		t.Fatalf("subscriber count after cleanup=%d, want 0", service.subscriberCount)
	}

	service.mu.Lock()
	service.subscriberCount = maxSubscribersGlobal
	service.mu.Unlock()
	if _, _, err := service.Subscribe(run.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("Subscribe above global limit error=%v, want ErrConflict", err)
	}
}
