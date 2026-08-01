//go:build !windows && cgo

package apiserver

// RoutePermission names the authorization required for a management route.
type RoutePermission string

const (
	PermHealthRead     RoutePermission = "health.read"
	PermReadyRead      RoutePermission = "ready.read"
	PermDocsRead       RoutePermission = "docs.read"
	PermClassifyInvoke RoutePermission = "classify.invoke"
	PermConfigRead     RoutePermission = "config.read"
	PermConfigWrite    RoutePermission = "config.write"
	PermSecretView     RoutePermission = "secret_view"
	PermLearningIngest RoutePermission = "learning.ingest"
	PermDataRead       RoutePermission = "data.read"
	PermDataWrite      RoutePermission = "data.write"
	PermMetricsRead    RoutePermission = "metrics.read"
)

// RouteSensitivity classifies response risk for inventory and policy.
type RouteSensitivity string

const (
	SensitivityPublic      RouteSensitivity = "public"
	SensitivityOperational RouteSensitivity = "operational"
	SensitivityConfig      RouteSensitivity = "config"
	SensitivitySecretView  RouteSensitivity = "secret_view"
	SensitivityMutation    RouteSensitivity = "mutation"
)

// RouteAuditAction names immutable audit events for mutation routes.
type RouteAuditAction string

const (
	AuditActionNone              RouteAuditAction = ""
	AuditActionConfigPatch       RouteAuditAction = "config.patch"
	AuditActionConfigPut         RouteAuditAction = "config.put"
	AuditActionConfigRollback    RouteAuditAction = "config.rollback"
	AuditActionKnowledgeBaseSave RouteAuditAction = "knowledge_base.save"
	AuditActionKnowledgeBaseDel  RouteAuditAction = "knowledge_base.delete"
	AuditActionOutcomeIngest     RouteAuditAction = "outcome.ingest"
	AuditActionMemoryDelete      RouteAuditAction = "memory.delete"
	AuditActionDataWrite         RouteAuditAction = "data.write"
)

type routePolicy struct {
	Permission  RoutePermission
	Sensitivity RouteSensitivity
	AuditAction RouteAuditAction
}

func managedRoute(
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
