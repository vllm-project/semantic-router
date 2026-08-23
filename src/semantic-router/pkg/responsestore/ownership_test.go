package responsestore

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

func TestMemoryStoreResponseOwnershipIsExactAndNondisclosing(t *testing.T) {
	store, testMemoryStoreResponseOwnershipIsExactAndNondisclosingErr := NewMemoryStore(StoreConfig{Enabled: true})
	if testMemoryStoreResponseOwnershipIsExactAndNondisclosingErr != nil {
		t.Fatal(testMemoryStoreResponseOwnershipIsExactAndNondisclosingErr)
	}
	t.Cleanup(func() { _ = store.Close() })
	ctx := context.Background()
	ownerA := responseapi.ResponseOwner{
		Mode:        responseapi.ResponseOwnerAuthenticated,
		NamespaceID: "namespace-a", APIKeyID: "key-a", UserID: "user-a",
	}
	ownerB := responseapi.ResponseOwner{
		Mode:        responseapi.ResponseOwnerAuthenticated,
		NamespaceID: "namespace-b", APIKeyID: "key-b", UserID: "user-b",
	}

	firstA := &responseapi.StoredResponse{Owner: ownerA, ID: "resp_shared", Status: responseapi.StatusCompleted}
	secondA := &responseapi.StoredResponse{
		Owner: ownerA, ID: "resp_second", PreviousResponseID: firstA.ID,
		Status: responseapi.StatusCompleted,
	}
	firstB := &responseapi.StoredResponse{Owner: ownerB, ID: "resp_shared", Status: responseapi.StatusCompleted}
	for owner, response := range map[responseapi.ResponseOwner]*responseapi.StoredResponse{
		ownerA: firstA,
		ownerB: firstB,
	} {
		if err := store.StoreResponse(ctx, owner, response); err != nil {
			t.Fatalf("store %v: %v", owner.Mode, err)
		}
	}
	if err := store.StoreResponse(ctx, ownerA, secondA); err != nil {
		t.Fatal(err)
	}

	gotA, testMemoryStoreResponseOwnershipIsExactAndNondisclosingErr := store.GetResponse(ctx, ownerA, firstA.ID)
	if testMemoryStoreResponseOwnershipIsExactAndNondisclosingErr != nil || gotA.Owner != ownerA {
		t.Fatalf("owner A get = (%v, %v)", gotA, testMemoryStoreResponseOwnershipIsExactAndNondisclosingErr)
	}
	gotB, testMemoryStoreResponseOwnershipIsExactAndNondisclosingErr := store.GetResponse(ctx, ownerB, firstB.ID)
	if testMemoryStoreResponseOwnershipIsExactAndNondisclosingErr != nil || gotB.Owner != ownerB {
		t.Fatalf("owner B get = (%v, %v)", gotB, testMemoryStoreResponseOwnershipIsExactAndNondisclosingErr)
	}
	if _, err := store.GetResponse(ctx, ownerB, secondA.ID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("cross-owner get error = %v, want ErrNotFound", err)
	}
	if err := store.DeleteResponse(ctx, ownerB, secondA.ID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("cross-owner delete error = %v, want ErrNotFound", err)
	}
	chain, testMemoryStoreResponseOwnershipIsExactAndNondisclosingErr := store.GetConversationChain(ctx, ownerA, secondA.ID)
	if testMemoryStoreResponseOwnershipIsExactAndNondisclosingErr != nil || len(chain) != 2 || chain[0].Owner != ownerA || chain[1].Owner != ownerA {
		t.Fatalf("owner A chain = (%v, %v)", chain, testMemoryStoreResponseOwnershipIsExactAndNondisclosingErr)
	}
	if chain, err := store.GetConversationChain(ctx, ownerB, secondA.ID); !errors.Is(err, ErrNotFound) || chain != nil {
		t.Fatalf("cross-owner chain = (%v, %v), want nil ErrNotFound", chain, err)
	}
	if _, err := store.GetResponse(ctx, ownerA, secondA.ID); err != nil {
		t.Fatalf("cross-owner delete affected owner A: %v", err)
	}
}

func TestMemoryStoreRejectsMissingOrMismatchedOwner(t *testing.T) {
	store, err := NewMemoryStore(StoreConfig{Enabled: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	owner := responseapi.ResponseOwner{
		Mode:        responseapi.ResponseOwnerAuthenticated,
		NamespaceID: "namespace", APIKeyID: "key", UserID: "user",
	}
	other := owner
	other.UserID = "other-user"
	response := &responseapi.StoredResponse{Owner: other, ID: "resp_owner"}
	if err := store.StoreResponse(context.Background(), owner, response); !errors.Is(err, ErrInvalidInput) {
		t.Fatalf("mismatched owner error = %v, want ErrInvalidInput", err)
	}
	if err := store.StoreResponse(context.Background(), responseapi.ResponseOwner{}, response); !errors.Is(err, ErrInvalidInput) {
		t.Fatalf("missing owner error = %v, want ErrInvalidInput", err)
	}
}
