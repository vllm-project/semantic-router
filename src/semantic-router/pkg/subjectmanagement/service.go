package subjectmanagement

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
	"unicode"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"
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
	if options.Repository == nil || options.CommandCodec == nil {
		return nil, ErrUnavailable
	}
	cursors, err := newCursorCodec(options.CursorKeyring)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
	if options.IdempotencyTTL < time.Minute || options.IdempotencyTTL > 7*24*time.Hour {
		cursors.close()
		return nil, fmt.Errorf("%w: idempotency TTL must be between 1m and 7d", ErrUnavailable)
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

func (service *Service) ResolveTeamDefaults(ctx context.Context, namespaceID string) (TeamDefaults, error) {
	if service == nil || !canonicalUUID(namespaceID) {
		return TeamDefaults{}, ErrInvalidRequest
	}
	return service.repository.ResolveTeamDefaults(ctx, namespaceID)
}

func (service *Service) GetUser(ctx context.Context, namespaceID, userID string) (User, error) {
	if service == nil || !canonicalUUID(namespaceID) || !canonicalUUID(userID) {
		return User{}, ErrInvalidRequest
	}
	return service.repository.GetUser(ctx, namespaceID, userID)
}

func (service *Service) ListUsers(ctx context.Context, request ListRequest) (Page[User], error) {
	pageSize, listUsersErr := validateListRequest(request, validUserStatus)
	if service == nil || listUsersErr != nil {
		return Page[User]{}, ErrInvalidRequest
	}
	search, listUsersErr := managementsearch.Normalize(request.Search)
	if listUsersErr != nil {
		return Page[User]{}, ErrInvalidRequest
	}
	request.Search = search
	scopeDigest, listUsersErr := request.Scope.Digest()
	if listUsersErr != nil || request.Scope.NamespaceID != accesscontrol.NamespaceID(request.NamespaceID) {
		return Page[User]{}, ErrInvalidRequest
	}
	query := UserQuery{
		NamespaceID: request.NamespaceID, Status: request.Status, Search: search,
		Scope: request.Scope, Limit: pageSize,
	}
	if request.Cursor != "" {
		cursor, err := service.cursors.decode(request.Cursor)
		if err != nil || cursor.Kind != "users" || cursor.NamespaceID != request.NamespaceID ||
			cursor.Status != request.Status || cursor.Search != search || cursor.ScopeDigest != scopeDigest ||
			!canonicalUUID(cursor.ID) || cursor.CreatedAt.IsZero() {
			return Page[User]{}, ErrInvalidRequest
		}
		query.After = &UserCursor{CreatedAt: cursor.CreatedAt, ID: cursor.ID}
	}
	if !request.Scope.All && len(request.Scope.UserIDs) == 0 {
		return Page[User]{Items: []User{}, PageSize: pageSize}, nil
	}
	page, listUsersErr := service.repository.ListUsers(ctx, query)
	if listUsersErr != nil {
		return Page[User]{}, listUsersErr
	}
	return service.userPage(request, page, pageSize)
}

func (service *Service) CreateUser(ctx context.Context, request CreateUserRequest) (MutationResult, error) {
	request.Email = accesscontrol.NormalizeEmail(request.Email)
	request.DisplayName = strings.TrimSpace(request.DisplayName)
	if service == nil || validateActor(request.NamespaceID, request.Actor) != nil ||
		validateEmail(request.Email) != nil || validateText("display name", request.DisplayName, 200) != nil {
		return MutationResult{}, ErrInvalidRequest
	}
	command, createUserErr := service.bindCommand(request.NamespaceID, request.Actor.PrincipalID,
		"/management/v1/users", request.IdempotencyKey, struct {
			Email       string `json:"email"`
			DisplayName string `json:"displayName"`
		}{request.Email, request.DisplayName})
	if createUserErr != nil {
		return MutationResult{}, createUserErr
	}
	if replay, found, err := service.repository.Replay(ctx, command); err != nil || found {
		return replay, err
	}
	id, createUserErr := service.nextID()
	if createUserErr != nil {
		return MutationResult{}, createUserErr
	}
	now := service.now().UTC()
	return service.repository.CreateUser(ctx, CreateUserMutation{User: User{
		ID: id, NamespaceID: request.NamespaceID, Email: request.Email, DisplayName: request.DisplayName,
		Status: accesscontrol.UserStatusActive, Revision: 1, CreatedAt: now, UpdatedAt: now,
	}, Command: command, Actor: request.Actor})
}

func (service *Service) UpdateUser(ctx context.Context, request UpdateUserRequest) (MutationResult, error) {
	if service == nil || validateActor(request.NamespaceID, request.Actor) != nil ||
		!canonicalUUID(request.UserID) || request.ExpectedRevision == 0 ||
		(request.Email == nil && request.DisplayName == nil && request.Status == nil) {
		return MutationResult{}, ErrInvalidRequest
	}
	current, err := service.repository.GetUser(ctx, request.NamespaceID, request.UserID)
	if err != nil {
		return MutationResult{}, err
	}
	if request.Email != nil {
		current.Email = accesscontrol.NormalizeEmail(*request.Email)
		if validateEmail(current.Email) != nil {
			return MutationResult{}, ErrInvalidRequest
		}
	}
	if request.DisplayName != nil {
		current.DisplayName = strings.TrimSpace(*request.DisplayName)
		if validateText("display name", current.DisplayName, 200) != nil {
			return MutationResult{}, ErrInvalidRequest
		}
	}
	if request.Status != nil {
		if *request.Status != accesscontrol.UserStatusActive && *request.Status != accesscontrol.UserStatusDisabled {
			return MutationResult{}, ErrInvalidRequest
		}
		current.Status = *request.Status
	}
	return service.repository.UpdateUser(ctx, current, request.ExpectedRevision, request.Actor)
}

func (service *Service) DeleteUser(ctx context.Context, request DeleteUserRequest) (MutationResult, error) {
	if service == nil || validateActor(request.NamespaceID, request.Actor) != nil ||
		!canonicalUUID(request.UserID) || request.ExpectedRevision == 0 {
		return MutationResult{}, ErrInvalidRequest
	}
	return service.repository.DeleteUser(ctx, request.NamespaceID, request.UserID, request.ExpectedRevision, request.Actor)
}

func (service *Service) GetTeam(ctx context.Context, namespaceID, teamID string) (Team, error) {
	if service == nil || !canonicalUUID(namespaceID) || !canonicalUUID(teamID) {
		return Team{}, ErrInvalidRequest
	}
	return service.repository.GetTeam(ctx, namespaceID, teamID)
}

func (service *Service) ListTeams(ctx context.Context, request ListRequest) (Page[Team], error) {
	pageSize, listTeamsErr := validateListRequest(request, validTeamStatus)
	if service == nil || listTeamsErr != nil {
		return Page[Team]{}, ErrInvalidRequest
	}
	search, listTeamsErr := managementsearch.Normalize(request.Search)
	if listTeamsErr != nil {
		return Page[Team]{}, ErrInvalidRequest
	}
	request.Search = search
	scopeDigest, listTeamsErr := request.Scope.Digest()
	if listTeamsErr != nil || request.Scope.NamespaceID != accesscontrol.NamespaceID(request.NamespaceID) {
		return Page[Team]{}, ErrInvalidRequest
	}
	query := TeamQuery{
		NamespaceID: request.NamespaceID, Status: request.Status, Search: search,
		Scope: request.Scope, Limit: pageSize,
	}
	if request.Cursor != "" {
		cursor, err := service.cursors.decode(request.Cursor)
		if err != nil || cursor.Kind != "teams" || cursor.NamespaceID != request.NamespaceID ||
			cursor.Status != request.Status || cursor.Search != search || cursor.ScopeDigest != scopeDigest ||
			!canonicalUUID(cursor.ID) || cursor.CreatedAt.IsZero() {
			return Page[Team]{}, ErrInvalidRequest
		}
		query.After = &TeamCursor{CreatedAt: cursor.CreatedAt, ID: cursor.ID}
	}
	if !request.Scope.All && len(request.Scope.TeamIDs) == 0 {
		return Page[Team]{Items: []Team{}, PageSize: pageSize}, nil
	}
	page, listTeamsErr := service.repository.ListTeams(ctx, query)
	if listTeamsErr != nil {
		return Page[Team]{}, listTeamsErr
	}
	return service.teamPage(request, page, pageSize)
}

func (service *Service) CreateTeam(ctx context.Context, request CreateTeamRequest) (MutationResult, error) {
	request.Name, request.Description = strings.TrimSpace(request.Name), strings.TrimSpace(request.Description)
	accessPolicyIDs, selectionValid := canonicalTeamAccessPolicyIDs(request.AccessPolicyIDs)
	if service == nil || validateActor(request.NamespaceID, request.Actor) != nil ||
		validateText("team name", request.Name, 200) != nil || validateOptionalText(request.Description, 1000) != nil ||
		!selectionValid || !canonicalUUID(request.RateLimitPolicyID) || !validTeamDefaultsSelection(request) {
		return MutationResult{}, ErrInvalidRequest
	}
	request.AccessPolicyIDs = accessPolicyIDs
	command, createTeamErr := service.bindCommand(request.NamespaceID, request.Actor.PrincipalID,
		"/management/v1/teams", request.IdempotencyKey, struct {
			Name              string   `json:"name"`
			Description       string   `json:"description"`
			AccessPolicyIDs   []string `json:"accessPolicyIds"`
			RateLimitPolicyID string   `json:"rateLimitPolicyId"`
		}{request.Name, request.Description, request.AccessPolicyIDs, request.RateLimitPolicyID})
	if createTeamErr != nil {
		return MutationResult{}, createTeamErr
	}
	if replay, found, err := service.repository.Replay(ctx, command); err != nil || found {
		return replay, err
	}
	teamID, createTeamErr := service.nextID()
	if createTeamErr != nil {
		return MutationResult{}, createTeamErr
	}
	accessBindings := make([]TeamAccessPolicyBinding, 0, len(request.AccessPolicyIDs))
	for _, policyID := range request.AccessPolicyIDs {
		bindingID, err := service.nextID()
		if err != nil {
			return MutationResult{}, err
		}
		accessBindings = append(accessBindings, TeamAccessPolicyBinding{ID: bindingID, PolicyID: policyID})
	}
	rateBindingID, createTeamErr := service.nextID()
	if createTeamErr != nil {
		return MutationResult{}, createTeamErr
	}
	now := service.now().UTC()
	return service.repository.CreateTeam(ctx, CreateTeamMutation{
		Team: Team{
			ID: teamID, NamespaceID: request.NamespaceID, Name: request.Name,
			Description: request.Description, Status: accesscontrol.TeamStatusActive,
			Revision: 1, CreatedAt: now, UpdatedAt: now,
		},
		AccessPolicyBindings: accessBindings,
		RateLimitAllocation:  TeamRateLimitAllocation{ID: rateBindingID, PolicyID: request.RateLimitPolicyID},
		NamespaceDefaults:    request.NamespaceDefaults, UseDefaultAccessPolicy: request.UseDefaultAccessPolicy,
		UseDefaultRateLimitPolicy: request.UseDefaultRateLimitPolicy,
		Command:                   command, Actor: request.Actor,
	})
}

func (service *Service) UpdateTeam(ctx context.Context, request UpdateTeamRequest) (MutationResult, error) {
	if service == nil || validateActor(request.NamespaceID, request.Actor) != nil ||
		!canonicalUUID(request.TeamID) || request.ExpectedRevision == 0 ||
		(request.Name == nil && request.Description == nil && request.Status == nil) {
		return MutationResult{}, ErrInvalidRequest
	}
	current, err := service.repository.GetTeam(ctx, request.NamespaceID, request.TeamID)
	if err != nil {
		return MutationResult{}, err
	}
	if request.Name != nil {
		current.Name = strings.TrimSpace(*request.Name)
		if validateText("team name", current.Name, 200) != nil {
			return MutationResult{}, ErrInvalidRequest
		}
	}
	if request.Description != nil {
		current.Description = strings.TrimSpace(*request.Description)
		if validateOptionalText(current.Description, 1000) != nil {
			return MutationResult{}, ErrInvalidRequest
		}
	}
	if request.Status != nil {
		if *request.Status != accesscontrol.TeamStatusActive && *request.Status != accesscontrol.TeamStatusDisabled {
			return MutationResult{}, ErrInvalidRequest
		}
		current.Status = *request.Status
	}
	return service.repository.UpdateTeam(ctx, current, request.ExpectedRevision, request.Actor)
}

func (service *Service) DeleteTeam(ctx context.Context, request DeleteTeamRequest) (MutationResult, error) {
	if service == nil || validateActor(request.NamespaceID, request.Actor) != nil ||
		!canonicalUUID(request.TeamID) || request.ExpectedRevision == 0 {
		return MutationResult{}, ErrInvalidRequest
	}
	return service.repository.DeleteTeam(ctx, request.NamespaceID, request.TeamID, request.ExpectedRevision, request.Actor)
}

func (service *Service) GetMembership(ctx context.Context, namespaceID, teamID, userID string) (Membership, error) {
	if service == nil || !canonicalUUID(namespaceID) || !canonicalUUID(teamID) || !canonicalUUID(userID) {
		return Membership{}, ErrInvalidRequest
	}
	return service.repository.GetMembership(ctx, namespaceID, teamID, userID)
}

func (service *Service) ListUserMemberships(ctx context.Context, request MembershipListRequest) (Page[UserMembership], error) {
	pageSize, query, err := service.membershipQuery(request, "user_memberships")
	if err != nil || request.UserID == "" || request.TeamID != "" {
		return Page[UserMembership]{}, ErrInvalidRequest
	}
	if !request.Scope.All && len(request.Scope.TeamIDs) == 0 {
		return Page[UserMembership]{Items: []UserMembership{}, PageSize: pageSize}, nil
	}
	page, err := service.repository.ListUserMemberships(ctx, query)
	if err != nil {
		return Page[UserMembership]{}, err
	}
	result := Page[UserMembership]{Items: page.Items, HasMore: page.HasMore, PageSize: pageSize}
	if page.HasMore {
		if len(page.Items) == 0 {
			return Page[UserMembership]{}, ErrUnavailable
		}
		last := page.Items[len(page.Items)-1]
		result.NextCursor, err = service.membershipCursor("user_memberships", request, last.CreatedAt, last.TeamID)
	}
	return result, err
}

func (service *Service) ListTeamMembers(ctx context.Context, request MembershipListRequest) (Page[TeamMember], error) {
	pageSize, query, err := service.membershipQuery(request, "team_members")
	if err != nil || request.TeamID == "" || request.UserID != "" {
		return Page[TeamMember]{}, ErrInvalidRequest
	}
	if !request.Scope.All && len(request.Scope.UserIDs) == 0 {
		return Page[TeamMember]{Items: []TeamMember{}, PageSize: pageSize}, nil
	}
	page, err := service.repository.ListTeamMembers(ctx, query)
	if err != nil {
		return Page[TeamMember]{}, err
	}
	result := Page[TeamMember]{Items: page.Items, HasMore: page.HasMore, PageSize: pageSize}
	if page.HasMore {
		if len(page.Items) == 0 {
			return Page[TeamMember]{}, ErrUnavailable
		}
		last := page.Items[len(page.Items)-1]
		result.NextCursor, err = service.membershipCursor("team_members", request, last.CreatedAt, last.UserID)
	}
	return result, err
}

func (service *Service) PutMembership(ctx context.Context, request PutMembershipRequest) (MutationResult, error) {
	if service == nil || validateActor(request.NamespaceID, request.Actor) != nil ||
		!canonicalUUID(request.TeamID) || !canonicalUUID(request.UserID) || !request.Role.Valid() {
		return MutationResult{}, ErrInvalidRequest
	}
	command, err := service.bindCommand(request.NamespaceID, request.Actor.PrincipalID,
		"/management/v1/teams/"+request.TeamID+"/members/"+request.UserID,
		request.IdempotencyKey, struct {
			Role accesscontrol.TeamRole `json:"role"`
		}{request.Role})
	if err != nil {
		return MutationResult{}, err
	}
	if replay, found, err := service.repository.Replay(ctx, command); err != nil || found {
		return replay, err
	}
	now := service.now().UTC()
	return service.repository.PutMembership(ctx, PutMembershipMutation{Membership: Membership{
		NamespaceID: request.NamespaceID, TeamID: request.TeamID, UserID: request.UserID,
		Role: request.Role, Status: accesscontrol.MembershipStatusActive,
		Revision: 1, CreatedAt: now, UpdatedAt: now,
	}, Command: command, Actor: request.Actor})
}

func (service *Service) UpdateMembership(ctx context.Context, request UpdateMembershipRequest) (MutationResult, error) {
	if service == nil || validateActor(request.NamespaceID, request.Actor) != nil ||
		!canonicalUUID(request.TeamID) || !canonicalUUID(request.UserID) || request.ExpectedRevision == 0 ||
		(request.Role == nil && request.Status == nil) {
		return MutationResult{}, ErrInvalidRequest
	}
	current, err := service.repository.GetMembership(ctx, request.NamespaceID, request.TeamID, request.UserID)
	if err != nil {
		return MutationResult{}, err
	}
	if request.Role != nil {
		if !request.Role.Valid() {
			return MutationResult{}, ErrInvalidRequest
		}
		current.Role = *request.Role
	}
	if request.Status != nil {
		if !request.Status.Valid() {
			return MutationResult{}, ErrInvalidRequest
		}
		current.Status = *request.Status
	}
	return service.repository.UpdateMembership(ctx, current, request.ExpectedRevision, request.Actor)
}

func (service *Service) DeleteMembership(ctx context.Context, request DeleteMembershipRequest) (MutationResult, error) {
	if service == nil || validateActor(request.NamespaceID, request.Actor) != nil ||
		!canonicalUUID(request.TeamID) || !canonicalUUID(request.UserID) || request.ExpectedRevision == 0 {
		return MutationResult{}, ErrInvalidRequest
	}
	return service.repository.DeleteMembership(ctx, request.NamespaceID, request.TeamID, request.UserID, request.ExpectedRevision, request.Actor)
}

func (service *Service) bindCommand(namespaceID, principalID, endpoint, key string, body any) (managementcommand.Command, error) {
	canonical, err := json.Marshal(body)
	if err != nil {
		return managementcommand.Command{}, ErrInvalidRequest
	}
	now := service.now().UTC()
	command, err := service.commands.Bind(managementcommand.NamespaceCommandScope(namespaceID), principalID,
		endpoint, key, canonical, now, now.Add(service.idempotencyTTL))
	if err != nil {
		return managementcommand.Command{}, ErrInvalidRequest
	}
	return command, nil
}

func (service *Service) nextID() (string, error) {
	value := service.newID()
	if !canonicalUUID(value) {
		return "", ErrUnavailable
	}
	return value, nil
}

func (service *Service) userPage(request ListRequest, page RepositoryPage[User], pageSize int) (Page[User], error) {
	result := Page[User]{Items: page.Items, HasMore: page.HasMore, PageSize: pageSize}
	if page.HasMore {
		if len(page.Items) == 0 {
			return Page[User]{}, ErrUnavailable
		}
		last := page.Items[len(page.Items)-1]
		scopeDigest, err := request.Scope.Digest()
		if err != nil {
			return Page[User]{}, ErrInvalidRequest
		}
		cursor, err := service.cursors.encode(cursorPayload{
			Kind: "users", NamespaceID: request.NamespaceID,
			Status: request.Status, Search: request.Search, ScopeDigest: scopeDigest,
			CreatedAt: last.CreatedAt, ID: last.ID,
		})
		result.NextCursor = cursor
		return result, err
	}
	return result, nil
}

func (service *Service) teamPage(request ListRequest, page RepositoryPage[Team], pageSize int) (Page[Team], error) {
	result := Page[Team]{Items: page.Items, HasMore: page.HasMore, PageSize: pageSize}
	if page.HasMore {
		if len(page.Items) == 0 {
			return Page[Team]{}, ErrUnavailable
		}
		last := page.Items[len(page.Items)-1]
		scopeDigest, err := request.Scope.Digest()
		if err != nil {
			return Page[Team]{}, ErrInvalidRequest
		}
		cursor, err := service.cursors.encode(cursorPayload{
			Kind: "teams", NamespaceID: request.NamespaceID,
			Status: request.Status, Search: request.Search, ScopeDigest: scopeDigest,
			CreatedAt: last.CreatedAt, ID: last.ID,
		})
		result.NextCursor = cursor
		return result, err
	}
	return result, nil
}

func (service *Service) membershipQuery(request MembershipListRequest, kind string) (int, MembershipQuery, error) {
	if service == nil || !canonicalUUID(request.NamespaceID) ||
		(request.UserID != "" && !canonicalUUID(request.UserID)) ||
		(request.TeamID != "" && !canonicalUUID(request.TeamID)) ||
		(request.Status != "" && !request.Status.Valid()) {
		return 0, MembershipQuery{}, ErrInvalidRequest
	}
	pageSize := request.PageSize
	if pageSize == 0 {
		pageSize = defaultPageSize
	}
	if pageSize < 1 || pageSize > maximumPageSize {
		return 0, MembershipQuery{}, ErrInvalidRequest
	}
	scopeDigest, err := request.Scope.Digest()
	if err != nil || request.Scope.NamespaceID != accesscontrol.NamespaceID(request.NamespaceID) {
		return 0, MembershipQuery{}, ErrInvalidRequest
	}
	query := MembershipQuery{
		NamespaceID: request.NamespaceID, UserID: request.UserID,
		TeamID: request.TeamID, Status: string(request.Status), Scope: request.Scope, Limit: pageSize,
	}
	if request.Cursor != "" {
		cursor, err := service.cursors.decode(request.Cursor)
		ownerID := request.UserID
		if request.TeamID != "" {
			ownerID = request.TeamID
		}
		if err != nil || cursor.Kind != kind || cursor.NamespaceID != request.NamespaceID ||
			cursor.OwnerID != ownerID || cursor.Status != string(request.Status) ||
			cursor.ScopeDigest != scopeDigest ||
			!canonicalUUID(cursor.ID) || cursor.CreatedAt.IsZero() {
			return 0, MembershipQuery{}, ErrInvalidRequest
		}
		query.After = &MembershipCursor{CreatedAt: cursor.CreatedAt, ID: cursor.ID}
	}
	return pageSize, query, nil
}

func (service *Service) membershipCursor(kind string, request MembershipListRequest, createdAt time.Time, id string) (string, error) {
	ownerID := request.UserID
	if request.TeamID != "" {
		ownerID = request.TeamID
	}
	scopeDigest, err := request.Scope.Digest()
	if err != nil {
		return "", ErrInvalidRequest
	}
	return service.cursors.encode(cursorPayload{
		Kind: kind, NamespaceID: request.NamespaceID,
		OwnerID: ownerID, Status: string(request.Status), ScopeDigest: scopeDigest, CreatedAt: createdAt, ID: id,
	})
}

func validateListRequest(request ListRequest, validStatus func(string) bool) (int, error) {
	if !canonicalUUID(request.NamespaceID) || (request.Status != "" && !validStatus(request.Status)) {
		return 0, ErrInvalidRequest
	}
	pageSize := request.PageSize
	if pageSize == 0 {
		pageSize = defaultPageSize
	}
	if pageSize < 1 || pageSize > maximumPageSize {
		return 0, ErrInvalidRequest
	}
	return pageSize, nil
}

func validUserStatus(value string) bool {
	return value == string(accesscontrol.UserStatusActive) || value == string(accesscontrol.UserStatusDisabled) ||
		value == string(accesscontrol.UserStatusDeleted)
}

func validTeamStatus(value string) bool {
	return value == string(accesscontrol.TeamStatusActive) || value == string(accesscontrol.TeamStatusDisabled)
}

func validDefaults(namespaceID string, defaults TeamDefaults) bool {
	return defaults.NamespaceID == namespaceID && defaults.SelfServiceRevision > 0 &&
		defaults.AccessPolicyRevision > 0 && defaults.RateLimitPolicyRevision > 0 &&
		canonicalUUID(defaults.AccessPolicyID) && canonicalUUID(defaults.RateLimitPolicyID)
}

func validTeamDefaultsSelection(request CreateTeamRequest) bool {
	usesDefaults := request.UseDefaultAccessPolicy || request.UseDefaultRateLimitPolicy
	if !usesDefaults {
		return request.NamespaceDefaults == nil
	}
	if request.NamespaceDefaults == nil || !validDefaults(request.NamespaceID, *request.NamespaceDefaults) {
		return false
	}
	if request.UseDefaultAccessPolicy &&
		(len(request.AccessPolicyIDs) != 1 || request.AccessPolicyIDs[0] != request.NamespaceDefaults.AccessPolicyID) {
		return false
	}
	return !request.UseDefaultRateLimitPolicy || request.RateLimitPolicyID == request.NamespaceDefaults.RateLimitPolicyID
}

func canonicalTeamAccessPolicyIDs(values []string) ([]string, bool) {
	if len(values) == 0 {
		return nil, false
	}
	canonical := append([]string(nil), values...)
	for _, value := range canonical {
		if !canonicalUUID(value) {
			return nil, false
		}
	}
	sort.Strings(canonical)
	for index := 1; index < len(canonical); index++ {
		if canonical[index] == canonical[index-1] {
			return nil, false
		}
	}
	return canonical, true
}

func validateActor(namespaceID string, actor Actor) error {
	if !canonicalUUID(namespaceID) || !canonicalUUID(actor.PrincipalID) || strings.TrimSpace(actor.RequestID) == "" {
		return ErrInvalidRequest
	}
	for _, principalID := range actor.ActorChain {
		if !canonicalUUID(principalID) {
			return ErrInvalidRequest
		}
	}
	if actor.SourceIP.IsValid() && actor.SourceIP != actor.SourceIP.Unmap() {
		return ErrInvalidRequest
	}
	return nil
}

func validateEmail(value string) error {
	if strings.Count(value, "@") != 1 || strings.ContainsAny(value, "\r\n\x00") || len(value) > 320 {
		return ErrInvalidRequest
	}
	// Reuse normalization semantics without coupling validation to UUID persistence.
	if accesscontrol.NormalizeEmail(value) != value {
		return ErrInvalidRequest
	}
	return nil
}

func validateText(field, value string, maximum int) error {
	if value == "" || len(value) > maximum {
		return fmt.Errorf("%s is invalid", field)
	}
	for _, character := range value {
		if unicode.IsControl(character) {
			return fmt.Errorf("%s is invalid", field)
		}
	}
	return nil
}

func validateOptionalText(value string, maximum int) error {
	if value == "" {
		return nil
	}
	return validateText("description", value, maximum)
}

func canonicalUUID(value string) bool {
	parsed, err := uuid.Parse(value)
	return err == nil && parsed.String() == value
}
