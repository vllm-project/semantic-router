package router

import (
	"net/http"

	backendapp "github.com/vllm-project/semantic-router/dashboard/backend/app"
	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

type routeAccess struct {
	auth *dashboardauth.Service
}

func newRouteAccess(app *backendapp.App) routeAccess {
	return routeAccess{auth: resolveAuth(app)}
}

func resolveAuth(app *backendapp.App) *dashboardauth.Service {
	if app == nil {
		return nil
	}
	return app.Auth
}

func (a routeAccess) wrap(role console.ConsoleRole, next http.Handler) http.Handler {
	if a.auth == nil || next == nil {
		return next
	}
	return a.auth.RequireRole(role, next)
}

func (a routeAccess) viewer(next http.Handler) http.Handler {
	return a.wrap(console.ConsoleRoleViewer, next)
}

func (a routeAccess) editor(next http.Handler) http.Handler {
	return a.wrap(console.ConsoleRoleEditor, next)
}

func (a routeAccess) operator(next http.Handler) http.Handler {
	return a.wrap(console.ConsoleRoleOperator, next)
}

func (a routeAccess) admin(next http.Handler) http.Handler {
	return a.wrap(console.ConsoleRoleAdmin, next)
}
