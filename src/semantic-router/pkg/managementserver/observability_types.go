package managementserver

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/auditlog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/subjectmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

type UsageQueryService interface {
	Summary(context.Context, usageledger.UsageQuery) (usageledger.UsageSummary, error)
	Series(context.Context, usageledger.UsageQuery) (usageledger.UsageSeries, error)
	Breakdown(context.Context, usageledger.BreakdownQuery) (usageledger.UsageBreakdown, error)
	ListLogs(context.Context, usageledger.LogQuery, *usageledger.LogCursorCodec) (usageledger.LogPage, error)
	RequestDetail(context.Context, string, string, usageledger.QueryVisibility) (usageledger.RequestDetail, error)
}

type ResultScopeResolver interface {
	ResolveResultScope(
		context.Context,
		accesscontrol.ManagementPrincipalID,
		accesscontrol.NamespaceID,
		accesscontrol.Permission,
	) (managementauthorization.ResultScope, error)
}

type AuditQueryService interface {
	List(context.Context, auditlog.Query, *auditlog.CursorCodec) (auditlog.Page, error)
}

// UsageResourceReader resolves the exact resource named by a subject-scoped
// usage route before any aggregate is returned. Authorization alone is not an
// existence oracle: an authorized-but-absent resource still returns 404.
type UsageResourceReader interface {
	GetUser(context.Context, string, string) (subjectmanagement.User, error)
	GetTeam(context.Context, string, string) (subjectmanagement.Team, error)
	GetAPIKey(context.Context, string, string) (accesscontrol.APIKey, error)
}

type ObservabilityRoutesOptions struct {
	Queries         UsageQueryService
	LogCursors      *usageledger.LogCursorCodec
	Audit           AuditQueryService
	AuditCursors    *auditlog.CursorCodec
	Resources       UsageResourceReader
	Authorization   Authorizer
	Scopes          ResultScopeResolver
	Namespaces      NamespaceResolver
	Sessions        SessionAuthenticator
	Now             func() time.Time
	MaximumRange    time.Duration
	DefaultRange    time.Duration
	DefaultPageSize int
}
