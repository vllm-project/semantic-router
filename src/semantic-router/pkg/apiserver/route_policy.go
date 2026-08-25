//go:build !windows && cgo

package apiserver

// RoutePermission names the authorization required for a management route.
type RoutePermission string

const (
	PermHealthRead         RoutePermission = "health.read"
	PermReadyRead          RoutePermission = "ready.read"
	PermDocsRead           RoutePermission = "docs.read"
	PermClassifyInvoke     RoutePermission = "classify.invoke"
	PermConfigRead         RoutePermission = "config.read"
	PermSecretView         RoutePermission = "secret_view"
	PermReplayRead         RoutePermission = "replay.read"
	PermReplayDetail       RoutePermission = "replay.detail"
	PermDataRead           RoutePermission = "data.read"
	PermDataWrite          RoutePermission = "data.write"
	PermMetricsRead        RoutePermission = "metrics.read"
	PermCacheRead          RoutePermission = "cache.read"
	PermCacheInvalidate    RoutePermission = "cache.invalidate"
	PermCacheManage        RoutePermission = "cache.manage"
	PermCompressionRead    RoutePermission = "compression.read"
	PermCompressionPreview RoutePermission = "compression.preview"
	PermCompressionManage  RoutePermission = "compression.manage"
)

// RouteSensitivity classifies response risk for inventory and policy.
type RouteSensitivity string

const (
	SensitivityPublic      RouteSensitivity = "public"
	SensitivityOperational RouteSensitivity = "operational"
	SensitivityConfig      RouteSensitivity = "config"
	SensitivityReplay      RouteSensitivity = "replay"
	SensitivitySecretView  RouteSensitivity = "secret_view"
	SensitivityMutation    RouteSensitivity = "mutation"
)

// RouteAuditAction names immutable audit events for mutation routes.
type RouteAuditAction string

const (
	AuditActionNone                  RouteAuditAction = ""
	AuditActionMemoryDelete          RouteAuditAction = "memory.delete"
	AuditActionDataWrite             RouteAuditAction = "data.write"
	AuditActionCacheInvalidate       RouteAuditAction = "cache.invalidate"
	AuditActionCacheFlush            RouteAuditAction = "cache.flush"
	AuditActionCompressionPreview    RouteAuditAction = "compression.preview"
	AuditActionCompressionInvalidate RouteAuditAction = "compression.recovery.invalidate"
)

type routePolicy struct {
	Permission  RoutePermission
	Sensitivity RouteSensitivity
	AuditAction RouteAuditAction
}

func authorizedRoute(
	meta EndpointMetadata,
	policy routePolicy,
	handler apiRouteHandler,
	body ...apiRequestBody,
) apiRoute {
	route := apiRoute{
		EndpointMetadata: meta,
		Handler:          handler,
		Permission:       policy.Permission,
		Sensitivity:      policy.Sensitivity,
		AuditAction:      policy.AuditAction,
	}
	if len(body) > 0 {
		route.RequestBody = body[0]
	}
	return route
}
