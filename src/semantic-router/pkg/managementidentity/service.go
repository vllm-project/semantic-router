package managementidentity

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

type BarrierAdmin interface {
	managementauth.RevocationBarrierStore
	RemoveDeny(context.Context, managementauth.BarrierKind, string) error
	Ready(context.Context) error
}

type Service struct {
	repository Repository
	barriers   BarrierAdmin
}

func NewService(repository Repository, barriers BarrierAdmin) (*Service, error) {
	if repository == nil || barriers == nil {
		return nil, errors.New("management identity service requires a repository and global revocation barriers")
	}
	return &Service{repository: repository, barriers: barriers}, nil
}

func (service *Service) Ready(ctx context.Context) error {
	if service == nil || service.repository == nil || service.barriers == nil {
		return errors.New("management identity service is unavailable")
	}
	if err := service.repository.Ready(ctx); err != nil {
		return err
	}
	return service.barriers.Ready(ctx)
}

func (service *Service) GetPrincipal(ctx context.Context, id string) (Principal, error) {
	return service.repository.GetPrincipal(ctx, id)
}

func (service *Service) ListPrincipals(ctx context.Context, request ListRequest) (PrincipalPage, error) {
	return service.repository.ListPrincipals(ctx, request)
}

func (service *Service) CreatePrincipal(ctx context.Context, request CreatePrincipal) (MutationResult, error) {
	if err := validateCreatePrincipal(request); err != nil {
		return MutationResult{}, err
	}
	return service.repository.CreatePrincipal(ctx, request)
}

func (service *Service) UpdatePrincipal(ctx context.Context, request UpdatePrincipal) (MutationResult, error) {
	if err := validateUpdatePrincipal(request); err != nil {
		return MutationResult{}, err
	}
	if request.Status != nil && *request.Status == accesscontrol.PrincipalStatusDisabled {
		if err := service.barriers.InstallDeny(ctx, managementauth.BarrierManagementPrincipal, request.ID); err != nil {
			return MutationResult{}, fmt.Errorf("install principal revocation barrier: %w", err)
		}
	}
	result, err := service.repository.UpdatePrincipal(ctx, request)
	if err != nil {
		return MutationResult{}, err
	}
	if request.Status != nil && *request.Status == accesscontrol.PrincipalStatusDisabled {
		if _, err := service.repository.RevokePrincipalSessions(ctx, request.ID); err != nil {
			return result, fmt.Errorf("revoke principal Management sessions: %w", err)
		}
	}
	if request.Status != nil && *request.Status == accesscontrol.PrincipalStatusActive {
		if err := service.barriers.RemoveDeny(ctx, managementauth.BarrierManagementPrincipal, request.ID); err != nil {
			return result, fmt.Errorf("release principal revocation barrier: %w", err)
		}
	}
	return result, nil
}

func (service *Service) DeletePrincipal(ctx context.Context, id string, expected uint64, actor MutationActor) (MutationResult, error) {
	if !canonicalUUID(id) || expected == 0 || validateActor(actor) != nil {
		return MutationResult{}, errors.New("management principal delete request is invalid")
	}
	if err := service.barriers.InstallDeny(ctx, managementauth.BarrierManagementPrincipal, id); err != nil {
		return MutationResult{}, err
	}
	if _, err := service.repository.RevokePrincipalSessions(ctx, id); err != nil {
		return MutationResult{}, err
	}
	return service.repository.DeletePrincipal(ctx, id, expected, actor)
}

func (service *Service) GetRole(ctx context.Context, id string) (Role, error) {
	return service.repository.GetRole(ctx, id)
}

func (service *Service) ListRoles(ctx context.Context, namespaceID string, request ListRequest) (RolePage, error) {
	if namespaceID != "" && !canonicalUUID(namespaceID) {
		return RolePage{}, errors.New("management role namespace is invalid")
	}
	return service.repository.ListRoles(ctx, namespaceID, request)
}

func (service *Service) CreateRole(ctx context.Context, request CreateRole) (MutationResult, error) {
	if !canonicalUUID(request.ID) || !canonicalUUID(request.NamespaceID) || validateActor(request.Actor) != nil {
		return MutationResult{}, errors.New("management role create request is invalid")
	}
	role := accesscontrol.ManagementRole{
		ID: accesscontrol.ManagementRoleID(request.ID), NamespaceID: accesscontrol.NamespaceID(request.NamespaceID),
		Name: request.Name, DisplayName: request.DisplayName, BuiltIn: false,
		Permissions: request.Permissions, Status: accesscontrol.RoleStatusActive, Revision: 1,
	}
	if err := role.Validate(); err != nil {
		return MutationResult{}, err
	}
	return service.repository.CreateRole(ctx, request)
}

func (service *Service) UpdateRole(ctx context.Context, request UpdateRole) (MutationResult, error) {
	if !canonicalUUID(request.ID) || request.ExpectedRevision == 0 ||
		(request.DisplayName == nil && request.Description == nil) || validateActor(request.Actor) != nil {
		return MutationResult{}, errors.New("management role update request is invalid")
	}
	return service.repository.UpdateRole(ctx, request)
}

func (service *Service) DeleteRole(ctx context.Context, id string, expected uint64, actor MutationActor) (MutationResult, error) {
	if !canonicalUUID(id) || expected == 0 || validateActor(actor) != nil {
		return MutationResult{}, errors.New("management role delete request is invalid")
	}
	return service.repository.DeleteRole(ctx, id, expected, actor)
}

func (service *Service) GetRoleBinding(ctx context.Context, id string) (RoleBinding, error) {
	return service.repository.GetRoleBinding(ctx, id)
}

func (service *Service) ListRoleBindings(ctx context.Context, principalID string, request ListRequest) (RoleBindingPage, error) {
	if principalID != "" && !canonicalUUID(principalID) {
		return RoleBindingPage{}, errors.New("management role-binding principal is invalid")
	}
	return service.repository.ListRoleBindings(ctx, principalID, request)
}

func (service *Service) CreateRoleBinding(ctx context.Context, request CreateRoleBinding) (MutationResult, error) {
	if !canonicalUUID(request.ID) || !canonicalUUID(request.PrincipalID) || validateActor(request.Actor) != nil {
		return MutationResult{}, errors.New("management role-binding create request is invalid")
	}
	binding := accesscontrol.ManagementRoleBinding{
		ID:          accesscontrol.ManagementRoleBindingID(request.ID),
		PrincipalID: accesscontrol.ManagementPrincipalID(request.PrincipalID),
		RoleID:      accesscontrol.ManagementRoleID(request.RoleID), Scope: request.Scope,
		DelegationCeiling: request.DelegationCeiling, Status: accesscontrol.BindingStatusActive, Revision: 1,
	}
	if err := binding.Validate(); err != nil {
		return MutationResult{}, err
	}
	return service.repository.CreateRoleBinding(ctx, request)
}

func (service *Service) UpdateRoleBinding(ctx context.Context, request UpdateRoleBinding) (MutationResult, error) {
	if !canonicalUUID(request.ID) || request.ExpectedRevision == 0 || !request.Status.Valid() || validateActor(request.Actor) != nil {
		return MutationResult{}, errors.New("management role-binding update request is invalid")
	}
	return service.repository.UpdateRoleBinding(ctx, request)
}

func (service *Service) DeleteRoleBinding(ctx context.Context, id string, expected uint64, actor MutationActor) (MutationResult, error) {
	if !canonicalUUID(id) || expected == 0 || validateActor(actor) != nil {
		return MutationResult{}, errors.New("management role-binding delete request is invalid")
	}
	return service.repository.DeleteRoleBinding(ctx, id, expected, actor)
}

func (service *Service) GetPrincipalUserLink(ctx context.Context, principalID, namespaceID string) (PrincipalUserLink, error) {
	return service.repository.GetPrincipalUserLink(ctx, principalID, namespaceID)
}

func (service *Service) PutPrincipalUserLink(ctx context.Context, request LinkMutation) (MutationResult, error) {
	if !canonicalUUID(request.PrincipalID) || !canonicalUUID(request.NamespaceID) || !canonicalUUID(request.UserID) || validateActor(request.Actor) != nil {
		return MutationResult{}, errors.New("principal User-link request is invalid")
	}
	return service.repository.PutPrincipalUserLink(ctx, request)
}

func (service *Service) DeletePrincipalUserLink(ctx context.Context, request LinkMutation) (MutationResult, error) {
	if !canonicalUUID(request.PrincipalID) || !canonicalUUID(request.NamespaceID) || request.ExpectedRevision == nil || *request.ExpectedRevision == 0 || validateActor(request.Actor) != nil {
		return MutationResult{}, errors.New("principal User-link delete request is invalid")
	}
	return service.repository.DeletePrincipalUserLink(ctx, request)
}

func (service *Service) LoadSessionPolicy(ctx context.Context) (managementauth.SessionPolicy, error) {
	return service.repository.LoadSessionPolicy(ctx)
}

func (service *Service) UpdateSessionPolicy(ctx context.Context, policy managementauth.SessionPolicy, expected uint64, actor MutationActor) (MutationResult, error) {
	if err := policy.Validate(); err != nil || expected == 0 || validateActor(actor) != nil {
		return MutationResult{}, errors.New("management session policy update is invalid")
	}
	if err := service.barriers.InstallDeny(ctx, managementauth.BarrierClusterSessionPolicy, "singleton"); err != nil {
		return MutationResult{}, err
	}
	result, err := service.repository.UpdateSessionPolicy(ctx, policy, expected, actor)
	if err != nil {
		return MutationResult{}, err
	}
	if err := service.barriers.RemoveDeny(ctx, managementauth.BarrierClusterSessionPolicy, "singleton"); err != nil {
		return result, fmt.Errorf("release Management session-policy barrier: %w", err)
	}
	return result, nil
}

func validateCreatePrincipal(request CreatePrincipal) error {
	if request.Issuer == ServiceAccountIssuer {
		return errors.New("service-account principals must be created through the service-account resource")
	}
	principal := accesscontrol.ManagementPrincipal{
		ID: accesscontrol.ManagementPrincipalID(request.ID), Issuer: request.Issuer,
		Subject: request.Subject, Status: accesscontrol.PrincipalStatusActive,
		Attributes: request.Attributes, CreatedAt: timeSentinel, UpdatedAt: timeSentinel,
	}
	if err := principal.Validate(); err != nil || request.DisplayName == "" || validateActor(request.Actor) != nil {
		return errors.New("management principal create request is invalid")
	}
	return nil
}

func validateUpdatePrincipal(request UpdatePrincipal) error {
	if !canonicalUUID(request.ID) || request.ExpectedRevision == 0 ||
		(request.DisplayName == nil && request.VerifiedEmail == nil && request.Status == nil) || validateActor(request.Actor) != nil {
		return errors.New("management principal update request is invalid")
	}
	if request.Status != nil && !request.Status.Valid() {
		return errors.New("management principal status is invalid")
	}
	return nil
}

func validateActor(actor MutationActor) error {
	if !canonicalUUID(actor.PrincipalID) || actor.RequestID == "" || actor.Reason == "" {
		return errors.New("management mutation actor is invalid")
	}
	for _, principalID := range actor.ActorChain {
		if !canonicalUUID(principalID) {
			return errors.New("management actor chain is invalid")
		}
	}
	return nil
}

func canonicalUUID(value string) bool {
	parsed, err := uuid.Parse(value)
	return err == nil && parsed.String() == value
}

var timeSentinel = time.Unix(1, 0).UTC()
