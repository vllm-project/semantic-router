package extproc

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontools"
)

// Closed enum of ResolvedStickyIdentity.Reason values. Bounded and safe to
// use as a metric/log label — never a raw identifier.
const (
	stickyToolIdentityReasonOK                  = "ok"
	stickyToolIdentityReasonMissingPrincipal    = "missing_principal"
	stickyToolIdentityReasonUntrustedProvenance = "untrusted_provenance"
	stickyToolIdentityReasonMissingRecipe       = "missing_recipe"
	stickyToolIdentityReasonMissingSecret       = "missing_secret"
)

const (
	// stickyToolStorageKeyPrefix namespaces every sticky tool-set storage
	// key from every other key shape sharing the same backend.
	stickyToolStorageKeyPrefix = "vsr:st:v1"

	// stickyToolIdentityHMACHexLen truncates each HMAC-SHA256 digest (64
	// hex chars) to this length before use, matching
	// pkg/cache's own userScopeNamespace truncation convention for the
	// same secret: an internal partition token does not need the full
	// digest, and a shorter one keeps storage keys more legible in logs
	// and Redis tooling.
	stickyToolIdentityHMACHexLen = 16

	// stickyToolIdentityFingerprintPrefixLen bounds how much of the
	// (already-hashed) policy fingerprint appears in the storage key, so a
	// policy change reliably changes the key without the key growing
	// unbounded.
	stickyToolIdentityFingerprintPrefixLen = 16
)

// ResolvedStickyIdentity is the outcome of evaluating a request against the
// trust rules for session-scoped sticky tool-set selection (issue #3347,
// PL-0042 section 2.4). Trusted is the sole authority on whether sticky
// state may be read or written for this request — StorageKey and QuotaKey
// are the zero value and must not be used when Trusted is false.
type ResolvedStickyIdentity struct {
	// StorageKey addresses this session's sticky state in the configured
	// store. Opaque: derived entirely from HMAC-SHA256 digests of the
	// principal and session ID, plus the (already-hashed) policy
	// fingerprint and the operator-chosen recipe name — never the raw
	// principal or session ID.
	StorageKey string
	// QuotaKey is the cardinality-bucket identity for
	// config.ToolSessionStoreConfig's max_sessions_per_identity bound (see
	// sessiontools.Store.CompareAndSwap). QuotaKey.Principal is the same
	// opaque HMAC digest embedded in StorageKey, not the raw principal.
	QuotaKey sessiontools.QuotaKey
	// Trusted reports whether sticky state may be read or written for this
	// request. False for any request that fails the trust rules below,
	// regardless of the reason — callers must fall back to ordinary
	// per-turn selection and must not read or write sticky state.
	Trusted bool
	// Reason is one of the stickyToolIdentityReason* constants above.
	Reason string
}

// ResolveStickyToolIdentity evaluates whether ctx's session may participate
// in sticky tool-set selection and, if so, derives its opaque storage and
// quota keys. recipeName is the resolved routing recipe (concrete-model
// passthrough, i.e. an empty recipeName, is always untrusted — sticky
// state requires an isolated recipe partition); policyFingerprint is the
// effective selection-policy fingerprint (pkg/tools.ToolPolicyFingerprint)
// for the matched decision.
//
// Every returned key is an HMAC-SHA256 keyed digest of the raw identity,
// never the raw identity itself. Callers must never log, return to a
// client, or persist StorageKey/QuotaKey.Principal anywhere but the
// sticky-tools store itself, and must never compare them against anything
// but another value this same function produced — there is deliberately no
// way to recover the raw principal or session ID from them.
func ResolveStickyToolIdentity(ctx *RequestContext, recipeName string, policyFingerprint string) ResolvedStickyIdentity {
	if reason := stickyToolTrustViolation(ctx, recipeName); reason != "" {
		return ResolvedStickyIdentity{Reason: reason}
	}
	secret, ok := stickyToolIdentitySecret()
	if !ok {
		// Defense in depth: buildRouterFromConfig refuses to construct a
		// router with sticky enabled anywhere unless the secret is
		// configured (router_build.go), so this should be unreachable in a
		// correctly admitted config. Failing closed here rather than
		// trusting that invariant end-to-end is the point — an unkeyed
		// fallback is exactly the "second weaker hash implementation"
		// issue #3347's blueprint says not to introduce.
		return ResolvedStickyIdentity{Reason: stickyToolIdentityReasonMissingSecret}
	}

	principalHMAC := stickyToolHMACHex(secret, ctx.AuthenticatedPrincipal)
	sessionHMAC := stickyToolHMACHex(secret, ctx.AuthenticatedPrincipal+":"+ctx.SessionID)
	fingerprintPrefix := truncateStickyToolString(policyFingerprint, stickyToolIdentityFingerprintPrefixLen)

	storageKey := strings.Join([]string{
		stickyToolStorageKeyPrefix, recipeName, principalHMAC, sessionHMAC, fingerprintPrefix,
	}, ":")

	return ResolvedStickyIdentity{
		StorageKey: storageKey,
		QuotaKey:   sessiontools.QuotaKey{Principal: principalHMAC, Namespace: recipeName},
		Trusted:    true,
		Reason:     stickyToolIdentityReasonOK,
	}
}

// stickyToolTrustViolation returns the first trust rule ctx/recipeName
// fails, or "" if none. Order matches the blueprint's own rule order
// (principal, then provenance, then recipe) so Reason is deterministic
// when a request fails more than one rule at once.
func stickyToolTrustViolation(ctx *RequestContext, recipeName string) string {
	if ctx == nil || strings.TrimSpace(ctx.AuthenticatedPrincipal) == "" {
		return stickyToolIdentityReasonMissingPrincipal
	}
	if strings.TrimSpace(ctx.SessionID) == "" || !isTrustedStickySessionProvenance(ctx.SessionProvenance) {
		return stickyToolIdentityReasonUntrustedProvenance
	}
	if strings.TrimSpace(recipeName) == "" {
		return stickyToolIdentityReasonMissingRecipe
	}
	return ""
}

// isTrustedStickySessionProvenance reports whether provenance is trusted
// enough to bind sticky tool-set state to. Only an explicit session
// declaration qualifies — see SessionProvenance's doc comments in
// request_context.go for why each of the other values is excluded.
func isTrustedStickySessionProvenance(provenance SessionProvenance) bool {
	switch provenance {
	case SessionProvenanceResponseAPI, SessionProvenanceHeader:
		return true
	default:
		return false
	}
}

// stickyToolIdentitySecret returns the raw USER_SCOPE_NAMESPACE_SECRET
// value when configured.
//
// This reads the environment variable directly rather than going through
// pkg/cache.UserScopeNamespace: that function silently falls back to an
// unkeyed SHA-256 when the secret is absent, which is a defensible choice
// for cache-scope isolation (a stale deployment without the secret
// shouldn't suddenly stop caching), but sticky tool-set identity must fail
// closed instead. pkg/cache exposes no accessor for the raw secret value —
// only the presence check (UserScopeSecretConfigured, used by both this
// function and router_build.go's admission gate) and that unsuitable
// derived-namespace function. This reads the same env var pkg/cache does,
// with the same HMAC-SHA256 primitive, just without the fallback — not a
// second, weaker hash implementation, the same one used more strictly.
func stickyToolIdentitySecret() (string, bool) {
	if !cache.UserScopeSecretConfigured() {
		return "", false
	}
	secret := strings.TrimSpace(os.Getenv("USER_SCOPE_NAMESPACE_SECRET"))
	return secret, secret != ""
}

func stickyToolHMACHex(secret, input string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	_, _ = mac.Write([]byte(input)) // hash.Hash.Write never returns an error
	return truncateStickyToolString(hex.EncodeToString(mac.Sum(nil)), stickyToolIdentityHMACHexLen)
}

func truncateStickyToolString(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
