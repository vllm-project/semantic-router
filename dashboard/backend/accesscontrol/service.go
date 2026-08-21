package accesscontrol

import (
	"context"
	"errors"
	"net/mail"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

type Service struct {
	store     *Store
	quota     *QuotaManager
	keySecret string
}

type Actor struct {
	ID    string
	Email string
}

func NewService(store *Store, quota *QuotaManager, keySecret string) (*Service, error) {
	if store == nil || quota == nil {
		return nil, errors.New("access-control durable and quota stores are required")
	}
	if len(strings.TrimSpace(keySecret)) < 32 {
		return nil, errors.New("ACCESS_CONTROL_KEY_SECRET must contain at least 32 characters")
	}
	return &Service{store: store, quota: quota, keySecret: keySecret}, nil
}

func (s *Service) Close() {
	s.quota.Close()
	s.store.Close()
}

func (s *Service) Store() *Store                                  { return s.store }
func (s *Service) Quota() *QuotaManager                           { return s.quota }
func (s *Service) Overview(ctx context.Context) (Overview, error) { return s.store.Overview(ctx) }

func (s *Service) OverviewForUser(ctx context.Context, userID string) (Overview, error) {
	return s.store.OverviewForUser(ctx, userID)
}

func (s *Service) ListUsers(ctx context.Context, filter ListFilter) ([]User, int64, error) {
	return s.store.ListUsers(ctx, filter)
}

func (s *Service) GetUser(ctx context.Context, id string) (User, error) {
	return s.store.GetUser(ctx, id)
}

func (s *Service) ListTeams(ctx context.Context, filter ListFilter) ([]Team, int64, error) {
	return s.store.ListTeams(ctx, filter)
}

func (s *Service) GetTeam(ctx context.Context, id string) (Team, error) {
	return s.store.GetTeam(ctx, id)
}

// EnsureModelUser keeps the model-serving identity aligned with its Dashboard
// account. Dashboard roles never participate in inference authorization.
func (s *Service) EnsureModelUser(ctx context.Context, id, email, name string) error {
	item := User{
		ID: id, Email: email, Name: name, Status: StatusActive,
	}
	if existing, err := s.store.GetUser(ctx, id); err == nil {
		item.AccessGroupIDs = existing.AccessGroupIDs
		item.BudgetID = existing.BudgetID
	}
	_, err := s.SaveUser(ctx, Actor{}, item)
	return err
}

// AssignModelUserTeam adds the invited user to the selected Team.
func (s *Service) AssignModelUserTeam(ctx context.Context, id, teamID, role string) error {
	return s.store.SetUserTeamMembership(ctx, id, strings.TrimSpace(teamID), role)
}

func (s *Service) ModelTeamName(ctx context.Context, teamID string) (string, error) {
	team, err := s.GetTeam(ctx, teamID)
	return team.Name, err
}

func (s *Service) RemoveModelUser(ctx context.Context, id string) error {
	err := s.store.DeleteUser(ctx, id)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil
	}
	return err
}

func (s *Service) ListAPIKeys(ctx context.Context, filter ListFilter) ([]APIKey, int64, error) {
	items, total, err := s.store.ListAPIKeys(ctx, filter)
	if err != nil {
		return nil, 0, err
	}
	items, err = s.store.ResolveAPIKeyPolicies(ctx, items)
	return items, total, err
}

func (s *Service) GetAPIKey(ctx context.Context, id string) (APIKey, error) {
	item, err := s.store.GetAPIKey(ctx, id)
	if err != nil {
		return APIKey{}, err
	}
	if err = s.resolveAPIKeyPolicy(ctx, &item); err != nil {
		return APIKey{}, err
	}
	return item, nil
}

func (s *Service) resolveAPIKeyPolicy(ctx context.Context, item *APIKey) error {
	items, err := s.store.ResolveAPIKeyPolicies(ctx, []APIKey{*item})
	if err != nil {
		return err
	}
	*item = items[0]
	return nil
}

func (s *Service) UpdateUser(ctx context.Context, actor Actor, item User) (User, error) {
	if strings.TrimSpace(item.ID) == "" {
		return User{}, validationError("user id is required")
	}
	if _, err := s.store.GetUser(ctx, item.ID); err != nil {
		return User{}, err
	}
	return s.SaveUser(ctx, actor, item)
}

func (s *Service) ListAccessGroups(ctx context.Context, filter ListFilter) ([]AccessGroup, int64, error) {
	return s.store.ListAccessGroups(ctx, filter)
}

func (s *Service) GetAccessGroup(ctx context.Context, id string) (AccessGroup, error) {
	return s.store.GetAccessGroup(ctx, id)
}

func (s *Service) ListBudgets(ctx context.Context, filter ListFilter) ([]Budget, int64, error) {
	return s.store.ListBudgets(ctx, filter)
}

func (s *Service) GetBudget(ctx context.Context, id string) (Budget, error) {
	return s.store.GetBudget(ctx, id)
}

func (s *Service) SaveUser(ctx context.Context, actor Actor, item User) (User, error) {
	item.Email = strings.ToLower(strings.TrimSpace(item.Email))
	item.Name = strings.TrimSpace(item.Name)
	if _, err := mail.ParseAddress(item.Email); err != nil {
		return User{}, validationError("a valid email is required")
	}
	if item.Name == "" {
		return User{}, validationError("name is required")
	}
	if item.Status == "" {
		item.Status = StatusActive
	}
	if item.Status != StatusActive && item.Status != StatusDisabled {
		return User{}, validationError("status must be active or disabled")
	}
	item.AccessGroupIDs = uniqueStrings(item.AccessGroupIDs)
	item.BudgetID = strings.TrimSpace(item.BudgetID)
	if item.BudgetID != "" {
		if err := s.validateAssignableBudget(ctx, item.BudgetID); err != nil {
			return User{}, err
		}
	}
	created := item.ID == ""
	if created {
		item.ID = uuid.NewString()
	}
	result, err := s.store.SaveUser(ctx, item)
	if err == nil {
		s.audit(ctx, actor, choose(created, "user.created", "user.updated"), "user", result.ID, map[string]any{"email": result.Email, "status": result.Status, "accessGroupIds": result.AccessGroupIDs, "budgetId": result.BudgetID})
	}
	return result, err
}

func (s *Service) DeleteUser(ctx context.Context, actor Actor, id string) error {
	err := s.store.DeleteUser(ctx, id)
	if err == nil {
		s.audit(ctx, actor, "user.deleted", "user", id, nil)
	}
	return err
}

func (s *Service) SaveTeam(ctx context.Context, actor Actor, item Team) (Team, error) {
	item.Name = strings.TrimSpace(item.Name)
	if item.Name == "" {
		return Team{}, validationError("name is required")
	}
	if item.Status == "" {
		item.Status = StatusActive
	}
	if item.Status != StatusActive && item.Status != StatusDisabled {
		return Team{}, validationError("status must be active or disabled")
	}
	var err error
	item.Members, err = normalizeTeamMemberships(item.ID, item.Members)
	if err != nil {
		return Team{}, err
	}
	if len(item.Members) > 0 {
		hasAdmin := false
		for _, membership := range item.Members {
			if membership.Role == TeamRoleAdmin {
				hasAdmin = true
				break
			}
		}
		if !hasAdmin {
			return Team{}, validationError("select at least one Team admin")
		}
	}
	item.AccessGroupIDs = uniqueStrings(item.AccessGroupIDs)
	if len(item.AccessGroupIDs) == 0 {
		return Team{}, validationError("at least one access group is required")
	}
	item.BudgetID = strings.TrimSpace(item.BudgetID)
	if item.BudgetID == "" {
		return Team{}, validationError("a budget is required")
	}
	if err = s.validateAssignableBudget(ctx, item.BudgetID); err != nil {
		return Team{}, err
	}
	created := item.ID == ""
	if created {
		item.ID = uuid.NewString()
	}
	result, err := s.store.SaveTeam(ctx, item)
	if err == nil {
		s.audit(ctx, actor, choose(created, "team.created", "team.updated"), "team", result.ID, map[string]any{
			"members": len(result.Members), "status": result.Status,
			"accessGroupIds": result.AccessGroupIDs, "budgetId": result.BudgetID,
		})
	}
	return result, err
}

func (s *Service) DeleteTeam(ctx context.Context, actor Actor, id string) error {
	err := s.store.DeleteTeam(ctx, id)
	if err == nil {
		s.audit(ctx, actor, "team.deleted", "team", id, nil)
	}
	return err
}

func (s *Service) CreateAPIKey(ctx context.Context, actor Actor, item APIKey) (CreatedAPIKey, error) {
	item.Name = strings.TrimSpace(item.Name)
	if item.Name == "" {
		return CreatedAPIKey{}, validationError("name is required")
	}
	if err := s.normalizeAndValidateKeyOwner(ctx, &item); err != nil {
		return CreatedAPIKey{}, err
	}
	item.AccessGroupIDs = uniqueStrings(item.AccessGroupIDs)
	if item.BudgetID != "" {
		if err := s.validateAssignableBudget(ctx, item.BudgetID); err != nil {
			return CreatedAPIKey{}, err
		}
	}
	secret, prefix, err := NewSecret()
	if err != nil {
		return CreatedAPIKey{}, err
	}
	digest, err := DigestKey(secret, s.keySecret)
	if err != nil {
		return CreatedAPIKey{}, err
	}
	item.ID = uuid.NewString()
	ciphertext, err := EncryptKeySecret(secret, s.keySecret, item.ID)
	if err != nil {
		return CreatedAPIKey{}, err
	}
	item.Prefix = prefix
	item.Status = StatusActive
	created, err := s.store.CreateAPIKey(ctx, item, digest, ciphertext)
	if err != nil {
		return CreatedAPIKey{}, err
	}
	s.audit(ctx, actor, "api_key.created", "api_key", created.ID, map[string]any{"prefix": created.Prefix, "ownerType": created.OwnerType, "ownerId": created.OwnerID, "contextTeamId": created.ContextTeamID, "accessGroupIds": created.AccessGroupIDs, "budgetId": created.BudgetID})
	return CreatedAPIKey{APIKey: created, Secret: secret}, nil
}

func (s *Service) RevealAPIKey(ctx context.Context, actor Actor, id string) (string, error) {
	ciphertext, err := s.store.APIKeyCiphertext(ctx, id)
	if err != nil {
		return "", err
	}
	secret, err := DecryptKeySecret(ciphertext, s.keySecret, id)
	if err == nil {
		s.audit(ctx, actor, "api_key.revealed", "api_key", id, nil)
	}
	return secret, err
}

func (s *Service) RotateAPIKey(ctx context.Context, actor Actor, id string) (CreatedAPIKey, error) {
	secret, prefix, err := NewSecret()
	if err != nil {
		return CreatedAPIKey{}, err
	}
	digest, err := DigestKey(secret, s.keySecret)
	if err != nil {
		return CreatedAPIKey{}, err
	}
	ciphertext, err := EncryptKeySecret(secret, s.keySecret, id)
	if err != nil {
		return CreatedAPIKey{}, err
	}
	item, err := s.store.RotateAPIKeySecret(ctx, id, prefix, digest, ciphertext)
	if err != nil {
		return CreatedAPIKey{}, err
	}
	s.audit(ctx, actor, "api_key.rotated", "api_key", id, map[string]any{"prefix": prefix})
	return CreatedAPIKey{APIKey: item, Secret: secret}, nil
}

func (s *Service) SetAPIKeyStatus(ctx context.Context, actor Actor, id, status string) (APIKey, error) {
	if status != StatusActive && status != StatusDisabled {
		return APIKey{}, validationError("status must be active or disabled")
	}
	item, err := s.store.SetAPIKeyStatus(ctx, id, status)
	if err == nil {
		s.audit(ctx, actor, "api_key.status_changed", "api_key", id, map[string]any{"status": status})
	}
	return item, err
}

func (s *Service) UpdateAPIKey(ctx context.Context, actor Actor, item APIKey) (APIKey, error) {
	item.Name = strings.TrimSpace(item.Name)
	item.BudgetID = strings.TrimSpace(item.BudgetID)
	if item.ID == "" || item.Name == "" {
		return APIKey{}, validationError("id and name are required")
	}
	if err := s.normalizeAndValidateKeyOwner(ctx, &item); err != nil {
		return APIKey{}, err
	}
	if item.Status != StatusActive && item.Status != StatusDisabled {
		return APIKey{}, validationError("status must be active or disabled")
	}
	if item.BudgetID != "" {
		if err := s.validateAssignableBudget(ctx, item.BudgetID); err != nil {
			return APIKey{}, err
		}
	}
	item.AccessGroupIDs = uniqueStrings(item.AccessGroupIDs)
	result, err := s.store.UpdateAPIKey(ctx, item)
	if err == nil {
		s.audit(ctx, actor, "api_key.updated", "api_key", item.ID, map[string]any{"ownerType": item.OwnerType, "ownerId": item.OwnerID, "contextTeamId": item.ContextTeamID, "accessGroupIds": item.AccessGroupIDs, "budgetId": item.BudgetID})
	}
	return result, err
}

func (s *Service) SaveAccessGroup(ctx context.Context, actor Actor, item AccessGroup) (AccessGroup, error) {
	item.Name = strings.TrimSpace(item.Name)
	if item.Name == "" {
		return AccessGroup{}, validationError("name is required")
	}
	item.ModelPatterns = uniqueStrings(item.ModelPatterns)
	if len(item.ModelPatterns) == 0 {
		return AccessGroup{}, validationError("at least one model pattern is required")
	}
	created := item.ID == ""
	if created {
		item.ID = uuid.NewString()
	}
	result, err := s.store.SaveAccessGroup(ctx, item)
	if err == nil {
		s.audit(ctx, actor, choose(created, "access_group.created", "access_group.updated"), "access_group", result.ID, map[string]any{"modelPatterns": result.ModelPatterns})
	}
	return result, err
}

func (s *Service) DeleteAccessGroup(ctx context.Context, actor Actor, id string) error {
	err := s.store.DeleteAccessGroup(ctx, id)
	if err == nil {
		s.audit(ctx, actor, "access_group.deleted", "access_group", id, nil)
	}
	return err
}

func (s *Service) SaveBudget(ctx context.Context, actor Actor, item Budget) (Budget, error) {
	item.Name = strings.TrimSpace(item.Name)
	item.Description = strings.TrimSpace(item.Description)
	if item.Name == "" {
		return Budget{}, validationError("name is required")
	}
	if item.RPM < 0 || item.TPM < 0 || item.DailyTokens < 0 {
		return Budget{}, validationError("quota limits cannot be negative")
	}
	if item.RPM == 0 && item.TPM == 0 && item.DailyTokens == 0 {
		return Budget{}, validationError("at least one quota limit is required")
	}
	if err := s.validateBudgetMutation(ctx, item); err != nil {
		return Budget{}, err
	}
	created := item.ID == ""
	if created {
		item.ID = uuid.NewString()
	}
	result, err := s.store.SaveBudget(ctx, item)
	if err == nil {
		s.audit(ctx, actor, choose(created, "budget.created", "budget.updated"), "budget", result.ID, map[string]any{"rpm": result.RPM, "tpm": result.TPM, "dailyTokens": result.DailyTokens})
	}
	return result, err
}

func (s *Service) normalizeAndValidateKeyOwner(ctx context.Context, item *APIKey) error {
	item.OwnerType = strings.TrimSpace(item.OwnerType)
	item.OwnerID = strings.TrimSpace(item.OwnerID)
	item.ContextTeamID = strings.TrimSpace(item.ContextTeamID)
	if item.OwnerType == "" {
		if item.UserID != "" {
			item.OwnerType, item.OwnerID = "user", item.UserID
		} else if item.TeamID != "" {
			item.OwnerType, item.OwnerID = "team", item.TeamID
		}
	}
	switch item.OwnerType {
	case "user":
		item.UserID, item.TeamID = item.OwnerID, ""
		if item.UserID == "" {
			return validationError("select a user")
		}
		if _, err := s.store.GetUser(ctx, item.UserID); err != nil {
			return validationError("selected user does not exist")
		}
		if item.ContextTeamID != "" {
			var member bool
			if err := s.store.pool.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM access_team_members WHERE team_id=$1 AND user_id=$2)`, item.ContextTeamID, item.UserID).Scan(&member); err != nil || !member {
				return validationError("the user must be a member of the selected Team")
			}
		}
	case "team":
		item.TeamID, item.UserID, item.ContextTeamID = item.OwnerID, "", item.OwnerID
		if item.TeamID == "" {
			return validationError("select a Team")
		}
		if _, err := s.store.GetTeam(ctx, item.TeamID); err != nil {
			return validationError("selected Team does not exist")
		}
	default:
		return validationError("owned by must be Personal or Team")
	}
	return nil
}

func (s *Service) DeleteBudget(ctx context.Context, actor Actor, id string) error {
	err := s.store.DeleteBudget(ctx, id)
	if err == nil {
		s.audit(ctx, actor, "budget.deleted", "budget", id, nil)
	}
	return err
}

func (s *Service) Authenticate(ctx context.Context, secret string) (*Principal, error) {
	digest, err := DigestKey(secret, s.keySecret)
	if err != nil {
		return nil, pgx.ErrNoRows
	}
	return s.store.PrincipalByDigest(ctx, digest)
}

// PrincipalForDashboardUser resolves the signed-in user's self-service key so
// Playground and public API calls share one authorization and usage contract.
func (s *Service) PrincipalForDashboardUser(ctx context.Context, userID string) (*Principal, error) {
	return s.store.PrincipalForDashboardUser(ctx, strings.TrimSpace(userID))
}

func (s *Service) RecordUsage(ctx context.Context, event UsageEvent) error {
	if event.ID == "" {
		event.ID = uuid.NewString()
	}
	if event.CreatedAt.IsZero() {
		event.CreatedAt = time.Now().UTC()
	}
	return s.store.InsertUsage(ctx, event)
}

func (s *Service) audit(ctx context.Context, actor Actor, action, resourceType, resourceID string, details map[string]any) {
	_ = s.store.InsertAudit(ctx, AuditEvent{
		ID: uuid.NewString(), ActorID: actor.ID, ActorEmail: actor.Email, Action: action,
		ResourceType: resourceType, ResourceID: resourceID, Details: details, CreatedAt: time.Now().UTC(),
	})
}

func choose[T any](condition bool, yes, no T) T {
	if condition {
		return yes
	}
	return no
}

func PublicError(err error) (int, string) {
	if errors.Is(err, pgx.ErrNoRows) {
		return 404, "resource not found"
	}
	if errors.Is(err, ErrSelfAPIKeyExists) {
		return 409, ErrSelfAPIKeyExists.Error()
	}
	var validation *ValidationError
	if errors.As(err, &validation) {
		return 400, validation.Error()
	}
	var databaseError *pgconn.PgError
	if errors.As(err, &databaseError) {
		switch databaseError.Code {
		case "23505":
			return 409, "a resource with these details already exists"
		case "23503":
			return 409, "this resource is still in use"
		case "23514":
			return 400, "the supplied values are not valid"
		}
	}
	return 500, "access-control operation failed"
}
