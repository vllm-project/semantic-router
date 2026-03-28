package cache

import (
	"strings"
	"testing"
)

func TestScopeQueryToUserReturnsOriginalWithoutUserID(t *testing.T) {
	query := "explain mitosis versus meiosis"

	scoped := ScopeQueryToUser(query, "")

	if scoped != query {
		t.Fatalf("expected original query without user scope, got %q", scoped)
	}
}

func TestScopeQueryToUserIsStableForSameUser(t *testing.T) {
	query := "explain mitosis versus meiosis"

	first := ScopeQueryToUser(query, "user-a")
	second := ScopeQueryToUser(query, "user-a")

	if first != second {
		t.Fatalf("expected deterministic scoped query, got %q and %q", first, second)
	}
}

func TestScopeQueryToUserSeparatesDifferentUsers(t *testing.T) {
	query := "explain mitosis versus meiosis"

	firstScoped := ScopeQueryToUser(query, "user-a")
	secondScoped := ScopeQueryToUser(query, "user-b")

	if firstScoped == secondScoped {
		t.Fatal("expected different users to produce different scoped queries")
	}
}

func TestScopeQueryToUserDoesNotExposeRawUserID(t *testing.T) {
	query := "explain mitosis versus meiosis"
	scoped := ScopeQueryToUser(query, "user-a")

	if strings.Contains(scoped, "user-a") {
		t.Fatalf("expected scoped query to hide raw user id, got %q", scoped)
	}
}

func TestScopeQueryToUserPreservesOriginalQueryText(t *testing.T) {
	query := "explain mitosis versus meiosis"
	scoped := ScopeQueryToUser(query, "user-a")

	if !strings.Contains(scoped, query) {
		t.Fatalf("expected scoped query to retain original text, got %q", scoped)
	}
}

func TestScopeQueryToUserRepeatsNamespaceMarker(t *testing.T) {
	query := "explain mitosis versus meiosis"
	scoped := ScopeQueryToUser(query, "user-a")

	if strings.Count(scoped, "cache-scope") != 1 {
		t.Fatalf("expected a single cache scope prefix block, got %q", scoped)
	}
	if strings.Count(scoped, userScopeNamespace("user-a")) != scopeNamespaceRepeat {
		t.Fatalf("expected repeated namespace marker, got %q", scoped)
	}
}

func TestUserScopeNamespaceIsStableForSameUser(t *testing.T) {
	first := userScopeNamespace("user-a")
	second := userScopeNamespace("user-a")

	if first != second {
		t.Fatalf("expected stable namespace, got %q and %q", first, second)
	}
}

func TestUserScopeNamespaceSeparatesDifferentUsers(t *testing.T) {
	first := userScopeNamespace("user-a")
	second := userScopeNamespace("user-b")

	if first == second {
		t.Fatalf("expected different namespaces, got %q", first)
	}
}

func TestUserScopeNamespaceHasCompactHashLength(t *testing.T) {
	namespace := userScopeNamespace("user-a")

	if len(namespace) != 16 {
		t.Fatalf("expected 16-char namespace hash, got %d (%q)", len(namespace), namespace)
	}
	for _, char := range namespace {
		if (char < '0' || char > '9') && (char < 'a' || char > 'f') {
			t.Fatalf("expected lowercase hex namespace, got %q", namespace)
		}
	}
}
