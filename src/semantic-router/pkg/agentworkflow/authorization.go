package agentworkflow

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementpermission"
)

func (provider *Provider) authorizeResources(
	ctx context.Context,
	invocation agentmanagement.ToolInvocationContext,
	permission accesscontrol.Permission,
	resourceType accesscontrol.ScopeResourceType,
	resourceIDs ...string,
) error {
	if provider == nil || invocation.AuthorityDigest == "" || len(resourceIDs) == 0 {
		return agentmanagement.ErrDenied
	}
	targets := make([]accesscontrol.ScopedTarget, 0, len(resourceIDs))
	for _, resourceID := range resourceIDs {
		target := accesscontrol.ScopedTarget{Scope: accesscontrol.ResourceScope(
			accesscontrol.NamespaceID(invocation.NamespaceID), resourceType,
			accesscontrol.ResourceID(resourceID),
		)}
		if err := target.Validate(); err != nil {
			return agentmanagement.ErrInvalid
		}
		targets = append(targets, target)
	}
	decision, err := provider.authorization.Authorize(ctx, managementauthorization.Request{
		PrincipalID:   accesscontrol.ManagementPrincipalID(invocation.PrincipalID),
		NamespaceID:   accesscontrol.NamespaceID(invocation.NamespaceID),
		Permission:    managementpermission.Require(string(permission), "target"),
		Targets:       map[string][]accesscontrol.ScopedTarget{"target": targets},
		Authenticated: true,
	})
	if err != nil {
		if errors.Is(err, managementauthorization.ErrDenied) {
			return agentmanagement.ErrDenied
		}
		return err
	}
	if decision.AuthorityDigest == "" {
		return agentmanagement.ErrDenied
	}
	return nil
}

func (provider *Provider) authorizeNamespace(
	ctx context.Context,
	invocation agentmanagement.ToolInvocationContext,
	permission accesscontrol.Permission,
) error {
	target := accesscontrol.ScopedTarget{Scope: accesscontrol.NamespaceScope(
		accesscontrol.NamespaceID(invocation.NamespaceID),
	)}
	decision, err := provider.authorization.Authorize(ctx, managementauthorization.Request{
		PrincipalID:   accesscontrol.ManagementPrincipalID(invocation.PrincipalID),
		NamespaceID:   accesscontrol.NamespaceID(invocation.NamespaceID),
		Permission:    managementpermission.Require(string(permission), "target"),
		Targets:       map[string][]accesscontrol.ScopedTarget{"target": {target}},
		Authenticated: true,
	})
	if err != nil {
		if errors.Is(err, managementauthorization.ErrDenied) {
			return agentmanagement.ErrDenied
		}
		return err
	}
	if invocation.AuthorityDigest == "" || decision.AuthorityDigest == "" {
		return agentmanagement.ErrDenied
	}
	return nil
}

func routingMutation(
	invocation agentmanagement.ToolInvocationContext,
	command *managementcommand.Command,
	action string,
) routingmanagement.MutationContext {
	return routingmanagement.MutationContext{
		PrincipalID: invocation.PrincipalID,
		ActorChain:  []string{invocation.PrincipalID},
		RequestID:   "agent-tool-" + invocation.InvocationID,
		Reason:      action,
		Command:     command,
	}
}

func (provider *Provider) bindCommand(
	invocation agentmanagement.ToolInvocationContext,
	toolName string,
	input json.RawMessage,
) (*managementcommand.Command, error) {
	var decoded any
	if err := json.Unmarshal(input, &decoded); err != nil {
		return nil, agentmanagement.ErrInvalid
	}
	canonical, err := json.Marshal(decoded)
	if err != nil {
		return nil, agentmanagement.ErrInvalid
	}
	now := provider.now().UTC()
	command, err := provider.commands.Bind(
		managementcommand.NamespaceCommandScope(invocation.NamespaceID),
		invocation.PrincipalID,
		"/internal/agent-tools/"+toolName,
		invocation.InvocationID,
		canonical,
		now,
		now.Add(workflowCommandTTL),
	)
	if err != nil {
		return nil, fmt.Errorf("bind Agent workflow command: %w", err)
	}
	return &command, nil
}

func mapRoutingError(err error) error {
	switch {
	case errors.Is(err, routingmanagement.ErrNotFound):
		return agentmanagement.ErrNotFound
	case errors.Is(err, routingmanagement.ErrConflict), errors.Is(err, routingmanagement.ErrImmutable),
		errors.Is(err, routingmanagement.ErrReferenced):
		return agentmanagement.ErrConflict
	case errors.Is(err, routingmanagement.ErrInvalid), errors.Is(err, routingmanagement.ErrClaim):
		return agentmanagement.ErrInvalid
	case errors.Is(err, routingmanagement.ErrProbeUnavailable):
		return agentmanagement.ErrToolUnavailable
	default:
		return err
	}
}
