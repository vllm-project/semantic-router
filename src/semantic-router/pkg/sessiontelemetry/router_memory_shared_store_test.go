package sessiontelemetry

import (
	"testing"
	"time"
)

type fakeRouterSessionStateStore struct {
	snapshot RouterSessionSnapshot
	found    bool
	saved    int
}

func (f *fakeRouterSessionStateStore) Load(string) (RouterSessionSnapshot, bool, error) {
	return f.snapshot, f.found, nil
}

func (f *fakeRouterSessionStateStore) Save(
	snapshot RouterSessionSnapshot,
	_ time.Duration,
) error {
	f.snapshot = snapshot
	f.found = true
	f.saved++
	return nil
}

func (f *fakeRouterSessionStateStore) Close() error { return nil }

func TestRouterSessionStateStoreRestoresAfterLocalReset(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	store := &fakeRouterSessionStateStore{}
	SetRouterSessionStateStore(store)
	t.Cleanup(func() {
		SetRouterSessionStateStore(nil)
		ResetRouterSessionMemoryForTesting()
	})

	RecordSessionDecision(SessionDecisionParams{
		SessionID:     "session-a",
		SelectedModel: "model-a",
		Timestamp:     time.Now(),
	})
	if store.saved == 0 {
		t.Fatal("shared store did not receive session snapshot")
	}

	ResetRouterSessionMemoryForTesting()
	snapshot, ok := GetRouterSessionSnapshot("session-a", time.Now())
	if !ok {
		t.Fatal("shared session snapshot was not restored")
	}
	if snapshot.CurrentModel != "model-a" {
		t.Fatalf("restored model = %q", snapshot.CurrentModel)
	}
}
