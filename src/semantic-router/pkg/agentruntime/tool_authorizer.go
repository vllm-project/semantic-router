package agentruntime

import (
	"context"
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementpermission"
)

type LiveToolAuthorizerOptions struct {
	Store     agentmanagement.Store
	Sessions  agentmanagement.SessionAuthority
	Authority managementauthorization.SnapshotLoader
}

// LiveToolAuthorizer rechecks the Management principal, Profile lifecycle,
// delegated inference authority, API-key target grants, and every tool
// permission immediately before the handler receives control. The resulting
// digest is explicit handler input and cannot be supplied by the model.
type LiveToolAuthorizer struct {
	store     agentmanagement.Store
	sessions  agentmanagement.SessionAuthority
	authority managementauthorization.SnapshotLoader
}

func NewLiveToolAuthorizer(options LiveToolAuthorizerOptions) (*LiveToolAuthorizer, error) {
	if options.Store == nil || options.Sessions == nil || options.Authority == nil {
		return nil, errors.New("agent live Tool authorizer dependencies are incomplete")
	}
	return &LiveToolAuthorizer{
		store: options.Store, sessions: options.Sessions, authority: options.Authority,
	}, nil
}

func (authorizer *LiveToolAuthorizer) AuthorizeTool(
	ctx context.Context,
	invocation agentmanagement.ToolInvocationContext,
	definition agentmanagement.ToolDefinition,
) (agentmanagement.ToolInvocationContext, error) {
	session, authorizeToolErr := authorizer.store.GetSession(ctx, invocation.NamespaceID, invocation.SessionID)
	if authorizeToolErr != nil {
		return agentmanagement.ToolInvocationContext{}, authorizeToolErr
	}
	if session.Status != agentmanagement.SessionActive || session.OwnerPrincipalID != invocation.PrincipalID ||
		session.Target != invocation.Target {
		return agentmanagement.ToolInvocationContext{}, agentmanagement.ErrDenied
	}
	currentProfile, authorizeToolErr := authorizer.store.GetProfile(ctx, invocation.NamespaceID, session.ProfileID)
	if authorizeToolErr != nil || currentProfile.Status != agentmanagement.StatusActive {
		if authorizeToolErr != nil {
			return agentmanagement.ToolInvocationContext{}, authorizeToolErr
		}
		return agentmanagement.ToolInvocationContext{}, agentmanagement.ErrDenied
	}
	pinnedProfile, authorizeToolErr := authorizer.store.GetProfileRevision(
		ctx, invocation.NamespaceID, session.ProfileID, session.ProfileRevision,
	)
	if authorizeToolErr != nil {
		return agentmanagement.ToolInvocationContext{}, authorizeToolErr
	}
	if err := authorizer.sessions.Reauthorize(ctx, session, pinnedProfile.MinimumTargetCapabilities); err != nil {
		return agentmanagement.ToolInvocationContext{}, err
	}
	snapshot, authorizeToolErr := authorizer.authority.Load(
		ctx, accesscontrol.ManagementPrincipalID(invocation.PrincipalID),
		accesscontrol.NamespaceID(invocation.NamespaceID),
	)
	if authorizeToolErr != nil || snapshot.AuthorityDigest == "" ||
		snapshot.Principal.ID != accesscontrol.ManagementPrincipalID(invocation.PrincipalID) {
		if authorizeToolErr != nil {
			return agentmanagement.ToolInvocationContext{}, authorizeToolErr
		}
		return agentmanagement.ToolInvocationContext{}, agentmanagement.ErrDenied
	}
	permissions := append([]accesscontrol.Permission(nil), definition.RequiredPermissions...)
	if !containsPermission(permissions, accesscontrol.PermissionToolInvoke) {
		permissions = append(permissions, accesscontrol.PermissionToolInvoke)
	}
	for _, permission := range permissions {
		if invocation.Origin.Kind == agentmanagement.ToolOriginRouter &&
			!agentSessionPermission(permission) {
			// Router-native handlers resolve every caller-supplied resource ID
			// and authorize the resulting Recipe/Entrypoint/Model scope. The
			// registry check remains the coarse Agent/Tool gate only.
			continue
		}
		target, targetErr := toolPermissionTarget(session, permission)
		if targetErr != nil {
			return agentmanagement.ToolInvocationContext{}, targetErr
		}
		authorizeToolErr = managementauthorization.Evaluate(
			managementpermission.Require(string(permission), "target"),
			managementauthorization.EvaluationContext{
				Authenticated: true, RoleGrants: snapshot.RoleGrants, TeamGrants: snapshot.TeamGrants,
				Targets: map[string][]accesscontrol.ScopedTarget{"target": {target}},
			},
		)
		if authorizeToolErr != nil {
			if errors.Is(authorizeToolErr, managementauthorization.ErrDenied) {
				return agentmanagement.ToolInvocationContext{}, agentmanagement.ErrDenied
			}
			return agentmanagement.ToolInvocationContext{}, authorizeToolErr
		}
	}
	invocation.AuthorityDigest = snapshot.AuthorityDigest
	return invocation, nil
}

func agentSessionPermission(permission accesscontrol.Permission) bool {
	switch permission {
	case accesscontrol.PermissionToolRead, accesscontrol.PermissionToolInvoke,
		accesscontrol.PermissionAgentRead, accesscontrol.PermissionAgentUse:
		return true
	default:
		return false
	}
}

func toolPermissionTarget(
	session agentmanagement.Session, permission accesscontrol.Permission,
) (accesscontrol.ScopedTarget, error) {
	namespaceID := accesscontrol.NamespaceID(session.NamespaceID)
	switch permission {
	case accesscontrol.PermissionToolRead, accesscontrol.PermissionToolInvoke,
		accesscontrol.PermissionAgentRead, accesscontrol.PermissionAgentUse:
		target := accesscontrol.ScopedTarget{Scope: accesscontrol.ResourceScope(
			namespaceID, accesscontrol.ScopeResourceAgentSession, accesscontrol.ResourceID(session.ID),
		)}
		if session.EffectiveUserID != "" {
			target.Ancestors = append(target.Ancestors, accesscontrol.UserScope(
				namespaceID, accesscontrol.UserID(session.EffectiveUserID),
			))
		}
		if session.EffectiveTeamID != "" {
			target.Ancestors = append(target.Ancestors, accesscontrol.TeamScope(
				namespaceID, accesscontrol.TeamID(session.EffectiveTeamID),
			))
		}
		return target, target.Validate()
	default:
		resourceType := accesscontrol.ScopeResourceModel
		if session.Target.Kind == agentmanagement.TargetEntrypoint {
			resourceType = accesscontrol.ScopeResourceEntrypoint
		}
		target := accesscontrol.ScopedTarget{Scope: accesscontrol.ResourceScope(
			namespaceID, resourceType, accesscontrol.ResourceID(session.TargetResourceID),
		)}
		return target, target.Validate()
	}
}

func containsPermission(values []accesscontrol.Permission, expected accesscontrol.Permission) bool {
	for _, value := range values {
		if value == expected {
			return true
		}
	}
	return false
}

var _ agentmanagement.ToolAuthorizer = (*LiveToolAuthorizer)(nil)
