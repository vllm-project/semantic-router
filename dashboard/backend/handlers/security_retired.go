package handlers

import (
	"encoding/json"
	"net/http"
)

// retiredSecurityPolicyMessage explains where the two independent concerns the retired
// page used to bundle now live, so an operator hitting the old API knows what to edit.
const retiredSecurityPolicyMessage = "The dashboard security policy surface has been retired. " +
	"Identity-based routing signals live in routing.signals.role_bindings and request " +
	"guardrails live in global.services.ratelimit; edit them through the config surfaces."

// RetiredSecurityPolicyHandler answers every retired /api/security/ request with a stable
// 410, for every method, without reading or writing router config.
//
// This has to stay registered rather than simply disappearing: registerSmartAPIRouter owns
// an /api/ catch-all, so an unregistered path would be proxied to an unrelated backend and
// answer 502 instead of telling the caller the surface is gone.
func RetiredSecurityPolicyHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		w.WriteHeader(http.StatusGone)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error":   "gone",
			"message": retiredSecurityPolicyMessage,
		})
	}
}
