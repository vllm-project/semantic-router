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
	stickyToolIdentityReasonMissingPolicy       = "missing_policy_fingerprint"
	stickyToolIdentityReasonMissingSecret       = "missing_secret"
)

const (
	// stickyToolStorageKeyPrefix namespaces every sticky tool-set storage
	// key from every other key shape sharing the same backend. Version 2
	// removes the policy fingerprint from the key so policy changes load and
	// explicitly invalidate the existing state instead of leaking orphaned
	// policy-specific keys.
	stickyToolStorageKeyPrefix = "vsr:st:v2"

	// sha256DigestHexLen keeps key-shape assertions and future helpers tied to
	// the complete HMAC-SHA256 digest rather than a collision-prone truncation.
	sha256DigestHexLen = sha256.Size * 2
)

// ResolvedStickyIdentity is the outcome of evaluating a request against the
// trust rules for session-scoped sticky tool-set selection (issue #3347,
// PL-0042 section 2.4). Trusted is the sole authority on whether sticky
// state may be read or written for this request — StorageKey and QuotaKey
// are the zero value and must not be used when Trusted is false.
type ResolvedStickyIdentity struct {
	// StorageKey addresses this session's sticky state in the configured
	// store. Opaque: derived entirely from domain-separated HMAC-SHA256
	// digests of the recipe, principal, and session ID — never their raw
	// values. The policy fingerprint belongs in State and is intentionally
	// not part of this stable identity key.
	StorageKey string
	// QuotaKey is the cardinality-bucket identity for
	// config.ToolSessionStoreConfig's max_sessions_per_identity bound (see
	// sessiontools.Store.CompareAndSwap). Both fields are the same opaque
	// HMAC digests embedded in StorageKey, not raw principal or recipe values.
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
	if reason := stickyToolTrustViolation(ctx, recipeName, policyFingerprint); reason != "" {
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

	recipeHMAC := stickyToolHMACHex(secret, "recipe", recipeName)
	principalHMAC := stickyToolHMACHex(secret, "principal", ctx.AuthenticatedPrincipal)
	sessionHMAC := stickyToolHMACHex(secret, "session", ctx.SessionID)

	storageKey := strings.Join([]string{
		stickyToolStorageKeyPrefix, recipeHMAC, principalHMAC, sessionHMAC,
	}, ":")

	return ResolvedStickyIdentity{
		StorageKey: storageKey,
		QuotaKey:   sessiontools.QuotaKey{Principal: principalHMAC, Namespace: recipeHMAC},
		Trusted:    true,
		Reason:     stickyToolIdentityReasonOK,
	}
}

// stickyToolTrustViolation returns the first trust rule ctx/recipeName
// fails, or "" if none. Order matches the blueprint's own rule order
// (principal, provenance, recipe, then policy) so Reason is deterministic
// when a request fails more than one rule at once.
func stickyToolTrustViolation(ctx *RequestContext, recipeName string, policyFingerprint string) string {
	if ctx == nil || strings.TrimSpace(ctx.AuthenticatedPrincipal) == "" {
		return stickyToolIdentityReasonMissingPrincipal
	}
	if strings.TrimSpace(ctx.SessionID) == "" || !isTrustedStickySessionProvenance(ctx.SessionProvenance) {
		return stickyToolIdentityReasonUntrustedProvenance
	}
	if strings.TrimSpace(recipeName) == "" {
		return stickyToolIdentityReasonMissingRecipe
	}
	if strings.TrimSpace(policyFingerprint) == "" {
		return stickyToolIdentityReasonMissingPolicy
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

func stickyToolHMACHex(secret, domain, input string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	_, _ = mac.Write([]byte(stickyToolStorageKeyPrefix)) // hash.Hash.Write never returns an error
	_, _ = mac.Write([]byte{0})
	_, _ = mac.Write([]byte(domain))
	_, _ = mac.Write([]byte{0})
	_, _ = mac.Write([]byte(input)) // hash.Hash.Write never returns an error
	return hex.EncodeToString(mac.Sum(nil))
}
