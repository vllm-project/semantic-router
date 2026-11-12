package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontools"
)

func trustedResponseAPICtx() *RequestContext {
	return &RequestContext{
		AuthenticatedPrincipal: "user-42",
		SessionID:              "resp-session-abc",
		SessionProvenance:      SessionProvenanceResponseAPI,
	}
}

func trustedHeaderCtx() *RequestContext {
	return &RequestContext{
		AuthenticatedPrincipal: "user-42",
		SessionID:              "header-session-abc",
		SessionProvenance:      SessionProvenanceHeader,
	}
}

func TestResolveStickyToolIdentity_TrustedResponseAPI_OK(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-secret")
	got := ResolveStickyToolIdentity(trustedResponseAPICtx(), "recipe-a", "policy-fp-1234567890abcdef")
	if !got.Trusted {
		t.Fatalf("expected trusted, got %+v", got)
	}
	if got.Reason != stickyToolIdentityReasonOK {
		t.Fatalf("reason = %q, want %q", got.Reason, stickyToolIdentityReasonOK)
	}
	if got.StorageKey == "" {
		t.Fatal("expected a non-empty StorageKey")
	}
	if got.QuotaKey.Namespace != "recipe-a" {
		t.Fatalf("QuotaKey.Namespace = %q, want %q", got.QuotaKey.Namespace, "recipe-a")
	}
	if got.QuotaKey.Principal == "" {
		t.Fatal("expected a non-empty QuotaKey.Principal")
	}
}

func TestResolveStickyToolIdentity_TrustedHeader_OK(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-secret")
	got := ResolveStickyToolIdentity(trustedHeaderCtx(), "recipe-a", "policy-fp")
	if !got.Trusted {
		t.Fatalf("expected trusted, got %+v", got)
	}
}

func TestResolveStickyToolIdentity_MissingPrincipal_Err(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-secret")
	ctx := trustedResponseAPICtx()
	ctx.AuthenticatedPrincipal = ""
	got := ResolveStickyToolIdentity(ctx, "recipe-a", "policy-fp")
	if got.Trusted {
		t.Fatal("expected untrusted when AuthenticatedPrincipal is empty")
	}
	if got.Reason != stickyToolIdentityReasonMissingPrincipal {
		t.Fatalf("reason = %q, want %q", got.Reason, stickyToolIdentityReasonMissingPrincipal)
	}
	if got.StorageKey != "" || got.QuotaKey != (sessiontools.QuotaKey{}) {
		t.Fatalf("expected zero-value keys on rejection, got %+v", got)
	}
}

func TestResolveStickyToolIdentity_WhitespaceOnlyPrincipal_Err(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-secret")
	ctx := trustedResponseAPICtx()
	ctx.AuthenticatedPrincipal = "   "
	got := ResolveStickyToolIdentity(ctx, "recipe-a", "policy-fp")
	if got.Trusted {
		t.Fatal("expected untrusted for a whitespace-only principal")
	}
	if got.Reason != stickyToolIdentityReasonMissingPrincipal {
		t.Fatalf("reason = %q, want %q", got.Reason, stickyToolIdentityReasonMissingPrincipal)
	}
}

func TestResolveStickyToolIdentity_UntrustedProvenance_Err(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-secret")
	untrusted := []SessionProvenance{
		SessionProvenanceMessageHash,
		SessionProvenanceRequestID,
		SessionProvenanceAnthropicPromptCache,
		SessionProvenanceNone,
	}
	for _, provenance := range untrusted {
		t.Run(string(provenance), func(t *testing.T) {
			ctx := trustedResponseAPICtx()
			ctx.SessionProvenance = provenance
			got := ResolveStickyToolIdentity(ctx, "recipe-a", "policy-fp")
			if got.Trusted {
				t.Fatalf("expected untrusted for provenance %q", provenance)
			}
			if got.Reason != stickyToolIdentityReasonUntrustedProvenance {
				t.Fatalf("reason = %q, want %q", got.Reason, stickyToolIdentityReasonUntrustedProvenance)
			}
		})
	}
}

func TestResolveStickyToolIdentity_EmptySessionID_Err(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-secret")
	ctx := trustedResponseAPICtx()
	ctx.SessionID = ""
	got := ResolveStickyToolIdentity(ctx, "recipe-a", "policy-fp")
	if got.Trusted {
		t.Fatal("expected untrusted for an empty SessionID even with a trusted provenance tag")
	}
	if got.Reason != stickyToolIdentityReasonUntrustedProvenance {
		t.Fatalf("reason = %q, want %q", got.Reason, stickyToolIdentityReasonUntrustedProvenance)
	}
}

func TestResolveStickyToolIdentity_MissingRecipe_Err(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-secret")
	got := ResolveStickyToolIdentity(trustedResponseAPICtx(), "", "policy-fp")
	if got.Trusted {
		t.Fatal("expected untrusted for an empty recipe name (concrete-model passthrough)")
	}
	if got.Reason != stickyToolIdentityReasonMissingRecipe {
		t.Fatalf("reason = %q, want %q", got.Reason, stickyToolIdentityReasonMissingRecipe)
	}
}

func TestResolveStickyToolIdentity_MissingSecret_Err(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "")
	got := ResolveStickyToolIdentity(trustedResponseAPICtx(), "recipe-a", "policy-fp")
	if got.Trusted {
		t.Fatal("expected untrusted when USER_SCOPE_NAMESPACE_SECRET is not configured")
	}
	if got.Reason != stickyToolIdentityReasonMissingSecret {
		t.Fatalf("reason = %q, want %q", got.Reason, stickyToolIdentityReasonMissingSecret)
	}
}

func TestResolveStickyToolIdentity_NilContext_Err(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-secret")
	got := ResolveStickyToolIdentity(nil, "recipe-a", "policy-fp")
	if got.Trusted {
		t.Fatal("expected untrusted for a nil context")
	}
	if got.Reason != stickyToolIdentityReasonMissingPrincipal {
		t.Fatalf("reason = %q, want %q", got.Reason, stickyToolIdentityReasonMissingPrincipal)
	}
}

func TestResolveStickyToolIdentity_KeysAreOpaque_NoPlaintextLeakage(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-secret")
	const principal = "extremely-identifiable-user-id-42"
	const sessionID = "extremely-identifiable-session-id-99"
	ctx := &RequestContext{
		AuthenticatedPrincipal: principal,
		SessionID:              sessionID,
		SessionProvenance:      SessionProvenanceResponseAPI,
	}
	got := ResolveStickyToolIdentity(ctx, "recipe-a", "policy-fp")
	if !got.Trusted {
		t.Fatalf("expected trusted, got %+v", got)
	}
	if strings.Contains(got.StorageKey, principal) {
		t.Fatalf("StorageKey leaks the raw principal: %q", got.StorageKey)
	}
	if strings.Contains(got.StorageKey, sessionID) {
		t.Fatalf("StorageKey leaks the raw session ID: %q", got.StorageKey)
	}
	if strings.Contains(got.QuotaKey.Principal, principal) {
		t.Fatalf("QuotaKey.Principal leaks the raw principal: %q", got.QuotaKey.Principal)
	}
	if got.QuotaKey.Principal == principal {
		t.Fatal("QuotaKey.Principal must be an HMAC digest, not the raw principal")
	}
}

func TestResolveStickyToolIdentity_Deterministic(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-secret")
	ctx := trustedResponseAPICtx()
	a := ResolveStickyToolIdentity(ctx, "recipe-a", "policy-fp")
	b := ResolveStickyToolIdentity(ctx, "recipe-a", "policy-fp")
	if a.StorageKey != b.StorageKey {
		t.Fatalf("StorageKey not deterministic: %q vs %q", a.StorageKey, b.StorageKey)
	}
	if a.QuotaKey != b.QuotaKey {
		t.Fatalf("QuotaKey not deterministic: %+v vs %+v", a.QuotaKey, b.QuotaKey)
	}
}

func TestResolveStickyToolIdentity_DifferentSessionsDifferentKeys(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-secret")
	a := ResolveStickyToolIdentity(trustedResponseAPICtx(), "recipe-a", "policy-fp")
	ctxB := trustedResponseAPICtx()
	ctxB.SessionID = "a-completely-different-session"
	b := ResolveStickyToolIdentity(ctxB, "recipe-a", "policy-fp")
	if a.StorageKey == b.StorageKey {
		t.Fatal("different sessions must produce different storage keys")
	}
	// Same principal, so the quota bucket (which partitions by principal,
	// not session) must stay the same.
	if a.QuotaKey != b.QuotaKey {
		t.Fatalf("same principal should share a quota bucket across sessions: %+v vs %+v", a.QuotaKey, b.QuotaKey)
	}
}

func TestResolveStickyToolIdentity_DifferentPrincipalsDifferentQuotaKeys(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-secret")
	a := ResolveStickyToolIdentity(trustedResponseAPICtx(), "recipe-a", "policy-fp")
	ctxB := trustedResponseAPICtx()
	ctxB.AuthenticatedPrincipal = "a-different-user"
	b := ResolveStickyToolIdentity(ctxB, "recipe-a", "policy-fp")
	if a.QuotaKey == b.QuotaKey {
		t.Fatal("different principals must produce different quota keys")
	}
}

func TestResolveStickyToolIdentity_DifferentPolicyFingerprintDifferentStorageKey(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-secret")
	a := ResolveStickyToolIdentity(trustedResponseAPICtx(), "recipe-a", "policy-fp-1")
	b := ResolveStickyToolIdentity(trustedResponseAPICtx(), "recipe-a", "policy-fp-2")
	if a.StorageKey == b.StorageKey {
		t.Fatal("a different policy fingerprint must change the storage key")
	}
}

func TestResolveStickyToolIdentity_DifferentRecipeDifferentKeys(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-secret")
	a := ResolveStickyToolIdentity(trustedResponseAPICtx(), "recipe-a", "policy-fp")
	b := ResolveStickyToolIdentity(trustedResponseAPICtx(), "recipe-b", "policy-fp")
	if a.StorageKey == b.StorageKey {
		t.Fatal("different recipes must produce different storage keys")
	}
	if a.QuotaKey == b.QuotaKey {
		t.Fatal("different recipes must produce different quota keys (recipe partitioning)")
	}
}
