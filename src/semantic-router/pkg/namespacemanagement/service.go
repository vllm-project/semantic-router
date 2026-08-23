package namespacemanagement

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"time"
	"unicode"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	defaultPageSize = 50
	maximumPageSize = 200
)

type Options struct {
	Repository     Repository
	CommandCodec   *managementcommand.Codec
	CursorKeyring  securitykeyring.Symmetric
	IdempotencyTTL time.Duration
	Now            func() time.Time
	NewID          func() string
}

type Service struct {
	repository     Repository
	commands       *managementcommand.Codec
	cursors        cursorCodec
	idempotencyTTL time.Duration
	now            func() time.Time
	newID          func() string
}

func NewService(options Options) (*Service, error) {
	if options.Repository == nil || options.CommandCodec == nil ||
		options.IdempotencyTTL < time.Minute || options.IdempotencyTTL > 7*24*time.Hour {
		return nil, ErrUnavailable
	}
	cursors, err := newCursorCodec(options.CursorKeyring)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	newID := options.NewID
	if newID == nil {
		newID = uuid.NewString
	}
	return &Service{
		repository: options.Repository, commands: options.CommandCodec, cursors: cursors,
		idempotencyTTL: options.IdempotencyTTL, now: now, newID: newID,
	}, nil
}

func (service *Service) Close() { service.cursors.close() }

func (service *Service) Ready(ctx context.Context) error {
	if service == nil || service.repository == nil || service.commands == nil {
		return ErrUnavailable
	}
	if err := service.repository.Ready(ctx, service.commands); err != nil {
		return fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
	return nil
}

func (service *Service) GetNamespace(ctx context.Context, namespaceID string) (Namespace, error) {
	if service == nil || !canonicalUUID(namespaceID) {
		return Namespace{}, ErrInvalidRequest
	}
	return service.repository.GetNamespace(ctx, namespaceID)
}

func (service *Service) ListNamespaces(ctx context.Context, request ListRequest) (Page[Namespace], error) {
	if service == nil || (request.Status != "" && request.Status != string(accesscontrol.NamespaceStatusActive) &&
		request.Status != string(accesscontrol.NamespaceStatusDisabled)) {
		return Page[Namespace]{}, ErrInvalidRequest
	}
	scope, listNamespacesErr := request.Scope.Canonical()
	if listNamespacesErr != nil {
		return Page[Namespace]{}, ErrInvalidRequest
	}
	pageSize := request.PageSize
	if pageSize == 0 {
		pageSize = defaultPageSize
	}
	if pageSize < 1 || pageSize > maximumPageSize {
		return Page[Namespace]{}, ErrInvalidRequest
	}
	digest, listNamespacesErr := scope.Digest()
	if listNamespacesErr != nil {
		return Page[Namespace]{}, ErrInvalidRequest
	}
	query := NamespaceQuery{Scope: scope, Status: request.Status, Limit: pageSize}
	if request.Cursor != "" {
		cursor, err := service.cursors.decode(request.Cursor)
		if err != nil || cursor.Status != request.Status || cursor.ScopeDigest != digest ||
			!canonicalUUID(cursor.ID) || cursor.CreatedAt.IsZero() {
			return Page[Namespace]{}, ErrInvalidRequest
		}
		query.After = &NamespaceCursor{CreatedAt: cursor.CreatedAt, ID: cursor.ID}
	}
	if !scope.All && len(scope.NamespaceIDs) == 0 {
		return Page[Namespace]{Items: []Namespace{}, PageSize: pageSize}, nil
	}
	page, listNamespacesErr := service.repository.ListNamespaces(ctx, query)
	if listNamespacesErr != nil {
		return Page[Namespace]{}, listNamespacesErr
	}
	result := Page[Namespace]{Items: page.Items, HasMore: page.HasMore, PageSize: pageSize}
	if page.HasMore {
		if len(page.Items) == 0 {
			return Page[Namespace]{}, ErrUnavailable
		}
		last := page.Items[len(page.Items)-1]
		result.NextCursor, listNamespacesErr = service.cursors.encode(cursorPayload{
			Status:      request.Status,
			ScopeDigest: digest, CreatedAt: last.CreatedAt, ID: last.ID,
		})
	}
	return result, listNamespacesErr
}

func (service *Service) CreateNamespace(ctx context.Context, request CreateNamespaceRequest) (MutationResult, error) {
	request.Name = strings.TrimSpace(request.Name)
	request.BillingCurrency = strings.TrimSpace(request.BillingCurrency)
	if service == nil || validateActor(request.Actor) != nil || validateName(request.Name) != nil ||
		!validCurrency(request.BillingCurrency) {
		return MutationResult{}, ErrInvalidRequest
	}
	canonical, err := json.Marshal(struct {
		Name            string `json:"name"`
		BillingCurrency string `json:"billingCurrency"`
		Reason          string `json:"reason"`
	}{request.Name, request.BillingCurrency, request.Actor.Reason})
	if err != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	now := service.now().UTC()
	command, err := service.commands.Bind(managementcommand.ClusterCommandScope(), request.Actor.PrincipalID,
		"/management/v1/namespaces", request.IdempotencyKey, canonical, now, now.Add(service.idempotencyTTL))
	if err != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	if replay, found, err := service.repository.Replay(ctx, command); err != nil || found {
		return replay, err
	}
	id := service.newID()
	if !canonicalUUID(id) {
		return MutationResult{}, ErrUnavailable
	}
	return service.repository.CreateNamespace(ctx, CreateNamespaceMutation{
		Namespace: Namespace{
			ID: id, Name: request.Name, QuotaPartitionID: id,
			BillingCurrency: request.BillingCurrency, Status: accesscontrol.NamespaceStatusActive,
			Revision: 1, RuntimeEpoch: 1, CreatedAt: now, UpdatedAt: now,
		},
		SelfService: SelfServicePolicy{
			NamespaceID: id, DelegatedSessionTTL: 15 * time.Minute,
			Revision: 1, SeedVersion: 1, UpdatedAt: now,
		},
		Security: ManagementSecurityPolicy{
			NamespaceID: id, ActionRequirements: restrictiveSecurityRequirements(),
			SeedVersion: SecurityPolicySeedVersion, Revision: 1, UpdatedAt: now,
		},
		RoutingClaims: RoutingClaimSchema{
			NamespaceID: id, Definitions: map[string]accessmanagement.ClaimDefinition{},
			Revision: 1, UpdatedAt: now,
		},
		Command: command, Actor: request.Actor,
	})
}

func (service *Service) PatchNamespace(ctx context.Context, request PatchNamespaceRequest) (MutationResult, error) {
	if service == nil || !canonicalUUID(request.NamespaceID) || request.ExpectedRevision == 0 ||
		validateActor(request.Actor) != nil || !request.Status.Valid() {
		return MutationResult{}, ErrInvalidRequest
	}
	current, err := service.repository.GetNamespace(ctx, request.NamespaceID)
	if err != nil {
		return MutationResult{}, err
	}
	if current.Status == request.Status {
		return MutationResult{}, ErrInvalidRequest
	}
	current.Status = request.Status
	return service.repository.PatchNamespace(ctx, current, request.ExpectedRevision, request.Actor)
}

func (service *Service) DeleteNamespace(ctx context.Context, request DeleteNamespaceRequest) (MutationResult, error) {
	if service == nil || !canonicalUUID(request.NamespaceID) || request.ExpectedRevision == 0 ||
		validateActor(request.Actor) != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	return service.repository.DeleteNamespace(ctx, request.NamespaceID, request.ExpectedRevision, request.Actor)
}

func (service *Service) GetSelfServicePolicy(ctx context.Context, namespaceID string) (SelfServicePolicy, error) {
	if service == nil || !canonicalUUID(namespaceID) {
		return SelfServicePolicy{}, ErrInvalidRequest
	}
	return service.repository.GetSelfServicePolicy(ctx, namespaceID)
}

func (service *Service) PatchSelfServicePolicy(ctx context.Context, request PatchSelfServicePolicyRequest) (MutationResult, error) {
	if service == nil || !canonicalUUID(request.NamespaceID) || request.ExpectedRevision == 0 ||
		validateActor(request.Actor) != nil ||
		(request.MaxKeysPerUser == nil && request.MaxDelegatedSessions == nil &&
			request.DelegatedSessionTTLSeconds == nil && request.AllowTeamKeyDelegation == nil &&
			request.AutomaticFirstKey == nil && request.TeamAdminCapabilities == nil &&
			request.DefaultAccessPolicyID == nil && request.DefaultRateLimitPolicyID == nil) {
		return MutationResult{}, ErrInvalidRequest
	}
	current, err := service.repository.GetSelfServicePolicy(ctx, request.NamespaceID)
	if err != nil {
		return MutationResult{}, err
	}
	if request.MaxKeysPerUser != nil {
		current.MaxKeysPerUser = *request.MaxKeysPerUser
	}
	if request.MaxDelegatedSessions != nil {
		current.MaxDelegatedSessions = *request.MaxDelegatedSessions
	}
	if request.DelegatedSessionTTLSeconds != nil {
		current.DelegatedSessionTTL = time.Duration(*request.DelegatedSessionTTLSeconds) * time.Second
	}
	if request.AllowTeamKeyDelegation != nil {
		current.AllowTeamKeyDelegation = *request.AllowTeamKeyDelegation
	}
	if request.AutomaticFirstKey != nil {
		current.AutomaticFirstKey = *request.AutomaticFirstKey
	}
	if request.TeamAdminCapabilities != nil {
		current.TeamAdminCapabilities = slices.Clone(*request.TeamAdminCapabilities)
	}
	if request.DefaultAccessPolicyID != nil {
		current.DefaultAccessPolicyID = *request.DefaultAccessPolicyID
	}
	if request.DefaultRateLimitPolicyID != nil {
		current.DefaultRateLimitPolicyID = *request.DefaultRateLimitPolicyID
	}
	if validateSelfService(current) != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	return service.repository.PatchSelfServicePolicy(ctx, current, request.ExpectedRevision, request.Actor)
}

func (service *Service) GetManagementSecurityPolicy(ctx context.Context, namespaceID string) (ManagementSecurityPolicy, error) {
	if service == nil || !canonicalUUID(namespaceID) {
		return ManagementSecurityPolicy{}, ErrInvalidRequest
	}
	return service.repository.GetManagementSecurityPolicy(ctx, namespaceID)
}

func (service *Service) PatchManagementSecurityPolicy(ctx context.Context, request PatchManagementSecurityPolicyRequest) (MutationResult, error) {
	if service == nil || !canonicalUUID(request.NamespaceID) || request.ExpectedRevision == 0 ||
		validateActor(request.Actor) != nil || validateRequirements(request.ActionRequirements) != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	current, err := service.repository.GetManagementSecurityPolicy(ctx, request.NamespaceID)
	if err != nil {
		return MutationResult{}, err
	}
	if securityPolicyLoosens(current.ActionRequirements, request.ActionRequirements) {
		requirement, found := current.ActionRequirements[ActionSecurityPolicyLoosen]
		now := service.now().UTC()
		if request.Session.ValidateAt(now) != nil || !found || !requirement.Allows(request.Session, now) {
			return MutationResult{}, ErrAssurance
		}
	}
	current.ActionRequirements = cloneRequirements(request.ActionRequirements)
	current.SeedVersion = SecurityPolicySeedVersion
	return service.repository.PatchManagementSecurityPolicy(ctx, current, request.ExpectedRevision, request.Actor)
}

func (service *Service) GetRoutingClaimSchema(ctx context.Context, namespaceID string) (RoutingClaimSchema, error) {
	if service == nil || !canonicalUUID(namespaceID) {
		return RoutingClaimSchema{}, ErrInvalidRequest
	}
	return service.repository.GetRoutingClaimSchema(ctx, namespaceID)
}

func (service *Service) PatchRoutingClaimSchema(ctx context.Context, request PatchRoutingClaimSchemaRequest) (MutationResult, error) {
	if service == nil || !canonicalUUID(request.NamespaceID) || request.ExpectedRevision == 0 ||
		validateActor(request.Actor) != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	schema := accessmanagement.RoutingClaimSchema{Revision: request.ExpectedRevision, Definitions: request.Definitions}
	if accessmanagement.ValidateSchema(schema) != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	return service.repository.PatchRoutingClaimSchema(ctx, RoutingClaimSchema{
		NamespaceID: request.NamespaceID,
		Definitions: cloneDefinitions(request.Definitions), Revision: request.ExpectedRevision,
	}, request.ExpectedRevision, request.Actor)
}

func (scope ResultScope) Canonical() (ResultScope, error) {
	result := ResultScope{All: scope.All, NamespaceIDs: slices.Clone(scope.NamespaceIDs)}
	if result.All {
		result.NamespaceIDs = nil
		return result, nil
	}
	slices.Sort(result.NamespaceIDs)
	result.NamespaceIDs = slices.Compact(result.NamespaceIDs)
	for _, id := range result.NamespaceIDs {
		if !canonicalUUID(id) {
			return ResultScope{}, ErrInvalidRequest
		}
	}
	return result, nil
}

func (scope ResultScope) Digest() (string, error) {
	canonical, err := scope.Canonical()
	if err != nil {
		return "", err
	}
	encoded, err := json.Marshal(canonical)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(encoded)
	return hex.EncodeToString(digest[:]), nil
}

func restrictiveSecurityRequirements() map[string]managementauth.ActionRequirement {
	strong := managementauth.ActionRequirement{AnyOf: []managementauth.AuthenticationRequirement{
		{Kind: managementauth.RequirementHuman, Human: &managementauth.HumanRequirement{
			MinimumAAL: "aal2", AcceptedAMR: []string{}, MaxAuthenticationAgeSeconds: 900,
		}},
		{Kind: managementauth.RequirementWorkload, Workload: &managementauth.WorkloadRequirement{
			MinimumWorkloadClass: "workload_strong", MaxSourceAgeSeconds: 30 * 24 * 60 * 60,
		}},
	}}
	result := make(map[string]managementauth.ActionRequirement, 5)
	for _, action := range []string{
		ActionUnknownUsageFenceWaive, ActionSecurityPolicyLoosen,
		ActionSecretDeliver, ActionSecretReveal, ActionRoleDelegate,
	} {
		result[action] = strong
	}
	return result
}

func validateSelfService(policy SelfServicePolicy) error {
	if !canonicalUUID(policy.NamespaceID) || policy.MaxKeysPerUser < 0 || policy.MaxKeysPerUser > 1000 ||
		policy.MaxDelegatedSessions < 0 || policy.MaxDelegatedSessions > 10000 ||
		policy.DelegatedSessionTTL < time.Minute || policy.DelegatedSessionTTL > 24*time.Hour || policy.Revision == 0 {
		return ErrInvalidRequest
	}
	if _, err := accesscontrol.TeamEntitlementOptionsFromPolicy(policy.AllowTeamKeyDelegation, policy.TeamAdminCapabilities); err != nil {
		return ErrInvalidRequest
	}
	for _, id := range []string{policy.DefaultAccessPolicyID, policy.DefaultRateLimitPolicyID} {
		if id != "" && !canonicalUUID(id) {
			return ErrInvalidRequest
		}
	}
	return nil
}

func validateRequirements(requirements map[string]managementauth.ActionRequirement) error {
	policy := managementauth.SessionPolicy{
		AccessTokenTTL: time.Minute, SessionTTL: time.Hour,
		MaxActiveSessions: 1, ActionRequirements: requirements,
		SeedVersion: managementauth.SupportedSessionPolicySeedVersion, Revision: 1, UpdatedAt: time.Now().UTC(),
	}
	return policy.Validate()
}

func securityPolicyLoosens(current, target map[string]managementauth.ActionRequirement) bool {
	for action, targetRequirement := range target {
		currentRequirement, found := current[action]
		if !found || requirementLoosens(currentRequirement, targetRequirement) {
			return true
		}
	}
	return false
}

func requirementLoosens(current, target managementauth.ActionRequirement) bool {
	for _, targetBranch := range target.AnyOf {
		covered := false
		for _, currentBranch := range current.AnyOf {
			if branchSubset(targetBranch, currentBranch) {
				covered = true
				break
			}
		}
		if !covered {
			return true
		}
	}
	return false
}

// branchSubset reports whether every session accepted by candidate is also
// accepted by baseline; then replacing baseline with candidate is not wider.
func branchSubset(candidate, baseline managementauth.AuthenticationRequirement) bool {
	if candidate.Kind != baseline.Kind {
		return false
	}
	switch candidate.Kind {
	case managementauth.RequirementHuman:
		if candidate.Human == nil || baseline.Human == nil {
			return false
		}
		return aalRank(candidate.Human.MinimumAAL) >= aalRank(baseline.Human.MinimumAAL) &&
			candidate.Human.MaxAuthenticationAgeSeconds <= baseline.Human.MaxAuthenticationAgeSeconds &&
			stringSetSubset(candidate.Human.AcceptedAMR, baseline.Human.AcceptedAMR)
	case managementauth.RequirementWorkload:
		if candidate.Workload == nil || baseline.Workload == nil {
			return false
		}
		return workloadRank(candidate.Workload.MinimumWorkloadClass) >= workloadRank(baseline.Workload.MinimumWorkloadClass) &&
			candidate.Workload.MaxSourceAgeSeconds <= baseline.Workload.MaxSourceAgeSeconds
	default:
		return false
	}
}

func stringSetSubset(candidate, baseline []string) bool {
	if len(baseline) == 0 {
		return true
	}
	if len(candidate) == 0 {
		return false
	}
	for _, value := range candidate {
		if !slices.Contains(baseline, value) {
			return false
		}
	}
	return true
}

func aalRank(value string) int {
	switch value {
	case "aal1":
		return 1
	case "aal2":
		return 2
	case "aal3":
		return 3
	}
	return 0
}

func workloadRank(value string) int {
	switch value {
	case "workload_standard":
		return 1
	case "workload_strong":
		return 2
	}
	return 0
}

func cloneRequirements(source map[string]managementauth.ActionRequirement) map[string]managementauth.ActionRequirement {
	result := make(map[string]managementauth.ActionRequirement, len(source))
	for action, requirement := range source {
		copyRequirement := requirement
		copyRequirement.AnyOf = slices.Clone(requirement.AnyOf)
		for index := range copyRequirement.AnyOf {
			if copyRequirement.AnyOf[index].Human != nil {
				human := *copyRequirement.AnyOf[index].Human
				human.AcceptedAMR = slices.Clone(human.AcceptedAMR)
				copyRequirement.AnyOf[index].Human = &human
			}
			if copyRequirement.AnyOf[index].Workload != nil {
				workload := *copyRequirement.AnyOf[index].Workload
				copyRequirement.AnyOf[index].Workload = &workload
			}
		}
		result[action] = copyRequirement
	}
	return result
}

func cloneDefinitions(source map[string]accessmanagement.ClaimDefinition) map[string]accessmanagement.ClaimDefinition {
	result := make(map[string]accessmanagement.ClaimDefinition, len(source))
	for name, definition := range source {
		result[name] = definition
	}
	return result
}

func validateActor(actor Actor) error {
	if !canonicalUUID(actor.PrincipalID) || strings.TrimSpace(actor.RequestID) == "" ||
		strings.TrimSpace(actor.Reason) == "" || strings.TrimSpace(actor.Reason) != actor.Reason ||
		len(actor.Reason) > 512 {
		return ErrInvalidRequest
	}
	for _, id := range actor.ActorChain {
		if !canonicalUUID(id) {
			return ErrInvalidRequest
		}
	}
	if actor.SourceIP.IsValid() && actor.SourceIP != actor.SourceIP.Unmap() {
		return ErrInvalidRequest
	}
	return nil
}

func validateName(value string) error {
	if value == "" || len(value) > 200 {
		return ErrInvalidRequest
	}
	for _, character := range value {
		if unicode.IsControl(character) {
			return ErrInvalidRequest
		}
	}
	return nil
}

func validCurrency(value string) bool {
	if len(value) != 3 || strings.ToUpper(value) != value {
		return false
	}
	for _, character := range value {
		if character < 'A' || character > 'Z' {
			return false
		}
	}
	return true
}

func canonicalUUID(value string) bool {
	parsed, err := uuid.Parse(value)
	return err == nil && parsed.String() == value
}
