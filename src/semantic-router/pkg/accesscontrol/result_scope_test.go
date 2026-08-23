package accesscontrol

import (
	"fmt"
	"testing"
)

func TestResultScopeCanonicalizesLargeExactResourceSet(t *testing.T) {
	const count = 10_000
	namespaceID := NamespaceID("11111111-1111-4111-8111-111111111111")
	forward := make([]ResourceID, count)
	reverse := make([]ResourceID, count)
	for index := range forward {
		forward[index] = ResourceID(fmt.Sprintf("%08x-0000-4000-8000-000000000000", index))
		reverse[count-index-1] = forward[index]
	}
	first := ResultScope{NamespaceID: namespaceID, ResourceIDs: map[ScopeResourceType][]ResourceID{
		ScopeResourceProviderCredential: forward,
	}}
	second := ResultScope{NamespaceID: namespaceID, ResourceIDs: map[ScopeResourceType][]ResourceID{
		ScopeResourceProviderCredential: reverse,
	}}

	canonical, err := second.Canonical()
	if err != nil {
		t.Fatalf("Canonical() error = %v", err)
	}
	ids := canonical.IDs(ScopeResourceProviderCredential)
	if len(ids) != count || ids[0] != forward[0] || ids[len(ids)-1] != forward[len(forward)-1] {
		t.Fatalf("canonical resource set has %d IDs, first/last = %q/%q", len(ids), ids[0], ids[len(ids)-1])
	}
	firstDigest, firstErr := first.Digest()
	secondDigest, secondErr := second.Digest()
	if firstErr != nil || secondErr != nil || firstDigest == "" || firstDigest != secondDigest {
		t.Fatalf("large-set digests = %q/%q, errors = %v/%v", firstDigest, secondDigest, firstErr, secondErr)
	}
}

func TestResultScopeCanonicalDropsEmptyTypedDimensions(t *testing.T) {
	scope := ResultScope{
		NamespaceID: NamespaceID("11111111-1111-4111-8111-111111111111"),
		ResourceIDs: map[ScopeResourceType][]ResourceID{ScopeResourceProviderCredential: nil},
	}
	canonical, err := scope.Canonical()
	if err != nil {
		t.Fatalf("Canonical() error = %v", err)
	}
	if !canonical.Empty() || canonical.ResourceIDs != nil {
		t.Fatalf("Canonical() = %#v, want empty scope", canonical)
	}
}
