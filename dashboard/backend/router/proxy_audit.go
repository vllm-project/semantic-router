package router

import (
	"context"
	"log"
	"net/http"
	"strings"

	backendapp "github.com/vllm-project/semantic-router/dashboard/backend/app"
	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/console"
	"github.com/vllm-project/semantic-router/dashboard/backend/proxy"
)

func appendEmbeddedServiceAudit(app *backendapp.App, r *http.Request, serviceName, targetID string) {
	if app == nil || app.Console == nil || app.Console.Audit == nil || r == nil {
		return
	}

	session, ok := dashboardauth.SessionFromRequest(r)
	if !ok || session == nil || !session.Authenticated {
		return
	}

	metadata := map[string]interface{}{
		"service": serviceName,
		"method":  r.Method,
		"path":    r.URL.Path,
	}
	if referer := strings.TrimSpace(r.Header.Get("Referer")); referer != "" {
		metadata["referer"] = referer
	}

	event := &console.AuditEvent{
		ActorType:  console.PrincipalTypeUser,
		ActorID:    session.User.ID,
		Action:     "proxy.embedded_service_access",
		TargetType: "embedded_service",
		TargetID:   strings.TrimSpace(targetID),
		Outcome:    console.AuditOutcomeSuccess,
		Message:    "Accessed embedded service via dashboard proxy.",
		Metadata:   metadata,
	}
	if err := app.Console.Audit.AppendAuditEvent(context.Background(), event); err != nil {
		log.Printf("proxy audit: failed to append audit event: %v", err)
	}
}

func rejectCrossOriginProxyAccess(w http.ResponseWriter, r *http.Request) bool {
	if err := proxy.ValidateDashboardOrigin(r); err != nil {
		http.Error(w, err.Error(), http.StatusForbidden)
		return true
	}
	return false
}
