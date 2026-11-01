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
	const tenant = "extremely-identifiable-tenant-id-7"
	const sessionID = "extremely-identifiable-session-id-99"
	ctx := &RequestContext{
		AuthenticatedPrincipal: principal,
		AuthenticatedTenant:    tenant,
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
	if strings.Contains(got.StorageKey, tenant) || strings.Contains(got.QuotaKey.Principal, tenant) {
		t.Fatalf("keys leak the raw tenant: %+v", got)
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

// A user ID is unique only within its tenant: the same user and session in
// another tenant, or with no tenant, gets its own storage key and quota bucket.
func TestResolveStickyToolIdentity_TenantsNeverShareKeys(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-secret")
	storage := map[string]string{}
	quota := map[sessiontools.QuotaKey]string{}
	for _, tenant := range []string{"tenant-a", "tenant-b", ""} {
		ctx := trustedHeaderCtx()
		ctx.AuthenticatedTenant = tenant
		got := ResolveStickyToolIdentity(ctx, "recipe-a", "policy-fp")
		if !got.Trusted {
			t.Fatalf("tenant %q: expected trusted, got %+v", tenant, got)
		}
		if other, seen := storage[got.StorageKey]; seen {
			t.Fatalf("tenants %q and %q share a storage key", other, tenant)
		}
		if other, seen := quota[got.QuotaKey]; seen {
			t.Fatalf("tenants %q and %q share a quota bucket", other, tenant)
		}
		storage[got.StorageKey], quota[got.QuotaKey] = tenant, tenant

		ctx.SessionID = "another-session"
		if again := ResolveStickyToolIdentity(ctx, "recipe-a", "policy-fp"); again.QuotaKey != got.QuotaKey {
			t.Fatalf("tenant %q: one principal's sessions must share its quota bucket", tenant)
		}
	}
}

// Moving characters between the tenant, principal, and session never
// produces the same identity, although each pair below joins to the same
// string with a separator.
func TestResolveStickyToolIdentity_FieldBoundariesAreUnambiguous(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-secret")
	identities := []struct{ tenant, principal, session string }{
		{"a", "b:c", "s"},
		{"a:b", "c", "s"},
		{"a", "b", "c:s"},
		{"", "a:b", "c:s"},
	}
	storage := map[string]bool{}
	quota := map[sessiontools.QuotaKey]bool{}
	for _, id := range identities {
		got := ResolveStickyToolIdentity(&RequestContext{
			AuthenticatedTenant: id.tenant, AuthenticatedPrincipal: id.principal,
			SessionID: id.session, SessionProvenance: SessionProvenanceHeader,
		}, "recipe-a", "policy-fp")
		if storage[got.StorageKey] || quota[got.QuotaKey] {
			t.Fatalf("identity %+v collides with another identity's keys", id)
		}
		storage[got.StorageKey], quota[got.QuotaKey] = true, true
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
