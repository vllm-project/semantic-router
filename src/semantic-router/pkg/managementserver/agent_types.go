package managementserver

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

type AgentDefaults interface {
	Ready(context.Context) error
}

type AgentPublicationCommitRequest struct {
	NamespaceID    string
	PlanID         string
	PlanDigest     string
	ExpectedETag   string
	IdempotencyKey string
	Mutation       agentmanagement.MutationContext
	Access         agentmanagement.AccessContext
}

type AgentPublicationCommitResult struct {
	OperationID string
	Revision    int64
	Replayed    bool
}

type AgentPublicationCommitter interface {
	Ready(context.Context) error
	Commit(context.Context, AgentPublicationCommitRequest) (AgentPublicationCommitResult, error)
}

type AgentRoutesOptions struct {
	Service       *agentmanagement.Service
	Defaults      AgentDefaults
	Publications  AgentPublicationCommitter
	LiveEvents    agentmanagement.LiveEventSubscriber
	Namespaces    NamespaceResolver
	Sessions      SessionAuthenticator
	Authorization Authorizer
	Scopes        ResultScopeResolver
	Now           func() time.Time
}

type agentPage[T any] struct {
	Data []T           `json:"data"`
	Page agentPageInfo `json:"page"`
}

type agentPageInfo struct {
	NextCursor string `json:"nextCursor,omitempty"`
	HasMore    bool   `json:"hasMore"`
	PageSize   int    `json:"pageSize"`
}

type agentDetail[T any] struct {
	Data T `json:"data"`
}

type agentToolPage struct {
	Data             []agentmanagement.ToolDefinition `json:"data"`
	Page             agentPageInfo                    `json:"page"`
	RegistryRevision string                           `json:"registryRevision"`
}

type agentSessionCreateWire struct {
	Mode            agentmanagement.SessionMode `json:"mode"`
	ProfileID       string                      `json:"profileId,omitempty"`
	KeyID           string                      `json:"keyId"`
	EffectiveTeamID string                      `json:"effectiveTeamId,omitempty"`
	Target          agentmanagement.Target      `json:"target"`
	Title           string                      `json:"title,omitempty"`
}

type agentTurnCreateWire struct {
	Input agentmanagement.TurnInput `json:"input"`
}

type agentCredentialCreateWire struct {
	Name   string `json:"name"`
	Secret string `json:"secret"`
}

type agentCredentialRotateWire struct {
	Secret string `json:"secret"`
}

type agentSourceApprovalWire struct {
	DiscoveryDigest string `json:"discoveryDigest"`
}

type agentPublicationCommitWire struct {
	PlanDigest string `json:"planDigest"`
}

type agentAuthenticatedRequest struct {
	NamespaceID string
	Session     managementauth.AuthenticatedSession
}
