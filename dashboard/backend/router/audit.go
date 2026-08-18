package router

import (
	"net/http"

	auth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

// auditMutation wraps a handler so that state-changing requests are recorded
// in the audit log. Read-only and preflight methods (GET, HEAD, OPTIONS) pass
// through unrecorded so the log stays a record of writes; actionFor maps the
// remaining methods to the audit action, and an empty action passes through
// unrecorded as well.
func auditMutation(svc *auth.Service, actionFor func(method string) string, resource string, next http.HandlerFunc) http.HandlerFunc {
	if svc == nil {
		return next
	}
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet, http.MethodHead, http.MethodOptions:
			next(w, r)
			return
		}
		action := actionFor(r.Method)
		if action == "" {
			next(w, r)
			return
		}
		svc.Audit(action, resource, next)(w, r)
	}
}

// fixedAuditAction returns an actionFor mapping that audits every method that
// reaches the wrapped handler under the given action. Used with auditMutation
// for routes whose handlers already guard the accepted methods.
func fixedAuditAction(action string) func(method string) string {
	return func(string) string { return action }
}

// kbAuditAction maps knowledge-base proxy methods to the router-side audit
// action names so the dashboard records KB writes with the same vocabulary as
// the router apiserver.
func kbAuditAction(method string) string {
	switch method {
	case http.MethodPost, http.MethodPut, http.MethodPatch:
		return "knowledge_base.save"
	case http.MethodDelete:
		return "knowledge_base.delete"
	default:
		return ""
	}
}
