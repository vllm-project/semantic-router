package sessiontelemetry

import (
	"testing"
	"time"
)

func TestPreviewPeekDoesNotHydrateSharedStore(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	at := time.Now()
	store := &fakeRouterSessionStateStore{snapshot: RouterSessionSnapshot{SessionID: "preview", CurrentModel: "model-a", LastSeen: at, ModelTurns: map[string]int{"model-a": 1}}, found: true}
	SetRouterSessionStateStore(store)
	t.Cleanup(func() { SetRouterSessionStateStore(nil); ResetRouterSessionMemoryForTesting() })
	got, ok := PeekRouterSessionSnapshot("preview", at)
	if !ok || got.CurrentModel != "model-a" {
		t.Fatal("shared snapshot unavailable")
	}
	got.ModelTurns["model-a"] = 9
	if routerSessionCount() != 0 || store.saved != 0 || store.snapshot.ModelTurns["model-a"] != 1 {
		t.Fatal("peek hydrated/saved or shared mutable maps")
	}
}

func TestPreviewPeekDoesNotEvictExpiredState(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	ResetLastModelForTesting()
	t.Cleanup(func() {
		ResetRouterSessionMemoryForTesting()
		ResetLastModelForTesting()
		setLastModelNowForTesting(nil)
	})
	at := time.Now()
	setLastModelNowForTesting(func() time.Time { return at })
	RecordSessionDecision(SessionDecisionParams{SessionID: "preview", SelectedModel: "model-a", Timestamp: at})
	RecordLastModel("preview", "model-a")
	future := at.Add(routerMemoryTTL + ttl + time.Hour)
	if _, ok := PeekRouterSessionSnapshot("preview", future); ok {
		t.Fatal("expired session returned")
	}
	if _, _, ok := PeekLastModelInfo("preview", future); ok {
		t.Fatal("expired last model returned")
	}
	if routerSessionCount() != 1 || lastModelSessionCount() != 1 {
		t.Fatal("peek evicted expired state")
	}
}
