package accesscontrol

import (
	"context"
	"errors"
	"net/mail"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
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
	_, err := s.SaveUser(ctx, Actor{}, User{
		ID: id, Email: email, Name: name, Status: StatusActive,
	})
	return err
}

// AssignModelUserTeam replaces the optional Team selected for a Dashboard user.
func (s *Service) AssignModelUserTeam(ctx context.Context, id, teamID string) error {
	return s.store.SetUserTeam(ctx, id, strings.TrimSpace(teamID))
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
	return s.store.ListAPIKeys(ctx, filter)
}

func (s *Service) GetAPIKey(ctx context.Context, id string) (APIKey, error) {
	item, err := s.store.GetAPIKey(ctx, id)
	if err != nil {
		return APIKey{}, err
	}
	item.ModelPatterns, err = s.store.ModelPatternsForKey(ctx, item)
	return item, err
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
		return User{}, errors.New("a valid email is required")
	}
	if item.Name == "" {
		return User{}, errors.New("name is required")
	}
	if item.Status == "" {
		item.Status = StatusActive
	}
	if item.Status != StatusActive && item.Status != StatusDisabled {
		return User{}, errors.New("status must be active or disabled")
	}
	created := item.ID == ""
	if created {
		item.ID = uuid.NewString()
	}
	result, err := s.store.SaveUser(ctx, item)
	if err == nil {
		s.audit(ctx, actor, choose(created, "user.created", "user.updated"), "user", result.ID, map[string]any{"email": result.Email, "status": result.Status})
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
		return Team{}, errors.New("name is required")
	}
	if item.Status == "" {
		item.Status = StatusActive
	}
	if item.Status != StatusActive && item.Status != StatusDisabled {
		return Team{}, errors.New("status must be active or disabled")
	}
	created := item.ID == ""
	if created {
		item.ID = uuid.NewString()
	}
	result, err := s.store.SaveTeam(ctx, item)
	if err == nil {
		s.audit(ctx, actor, choose(created, "team.created", "team.updated"), "team", result.ID, map[string]any{"members": len(result.UserIDs), "status": result.Status})
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
		return CreatedAPIKey{}, errors.New("name is required")
	}
	if (item.UserID == "") == (item.TeamID == "") {
		return CreatedAPIKey{}, errors.New("exactly one of userId or teamId is required")
	}
	item.AccessGroupIDs = uniqueStrings(item.AccessGroupIDs)
	if item.BudgetID != "" {
		if _, err := s.store.GetBudget(ctx, item.BudgetID); err != nil {
			return CreatedAPIKey{}, err
		}
	}
	if item.Budget != nil {
		if item.Budget.RPM < 0 || item.Budget.TPM < 0 || item.Budget.DailyTokens < 0 {
			return CreatedAPIKey{}, errors.New("quota limits cannot be negative")
		}
		if item.Budget.RPM == 0 && item.Budget.TPM == 0 && item.Budget.DailyTokens == 0 {
			item.Budget = nil
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
	s.audit(ctx, actor, "api_key.created", "api_key", created.ID, map[string]any{"prefix": created.Prefix, "userId": created.UserID, "teamId": created.TeamID, "accessGroupIds": created.AccessGroupIDs, "budgetId": created.BudgetID, "hasKeyBudget": created.Budget != nil})
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
		return APIKey{}, errors.New("status must be active or disabled")
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
		return APIKey{}, errors.New("id and name are required")
	}
	if (item.UserID == "") == (item.TeamID == "") {
		return APIKey{}, errors.New("exactly one of userId or teamId is required")
	}
	if item.Status != StatusActive && item.Status != StatusDisabled {
		return APIKey{}, errors.New("status must be active or disabled")
	}
	if item.BudgetID != "" {
		if _, err := s.store.GetBudget(ctx, item.BudgetID); err != nil {
			return APIKey{}, err
		}
	}
	item.AccessGroupIDs = uniqueStrings(item.AccessGroupIDs)
	if item.Budget != nil {
		if item.Budget.RPM < 0 || item.Budget.TPM < 0 || item.Budget.DailyTokens < 0 {
			return APIKey{}, errors.New("quota limits cannot be negative")
		}
		if item.Budget.RPM == 0 && item.Budget.TPM == 0 && item.Budget.DailyTokens == 0 {
			item.Budget = nil
		}
	}
	result, err := s.store.UpdateAPIKey(ctx, item)
	if err == nil {
		s.audit(ctx, actor, "api_key.updated", "api_key", item.ID, map[string]any{"accessGroupIds": item.AccessGroupIDs, "budgetId": item.BudgetID, "hasKeyBudget": item.Budget != nil})
	}
	return result, err
}

func (s *Service) SaveAccessGroup(ctx context.Context, actor Actor, item AccessGroup) (AccessGroup, error) {
	item.Name = strings.TrimSpace(item.Name)
	if item.Name == "" {
		return AccessGroup{}, errors.New("name is required")
	}
	item.ModelPatterns = uniqueStrings(item.ModelPatterns)
	if len(item.ModelPatterns) == 0 {
		return AccessGroup{}, errors.New("at least one model pattern is required")
	}
	created := item.ID == ""
	if created {
		item.ID = uuid.NewString()
	}
	result, err := s.store.SaveAccessGroup(ctx, item)
	if err == nil {
		s.audit(ctx, actor, choose(created, "access_group.created", "access_group.updated"), "access_group", result.ID, map[string]any{"modelPatterns": result.ModelPatterns, "bindings": len(result.Bindings)})
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
	item.ScopeType = strings.TrimSpace(item.ScopeType)
	item.ScopeID = strings.TrimSpace(item.ScopeID)
	if item.Name == "" {
		return Budget{}, errors.New("name is required")
	}
	if item.ScopeType != "global" && item.ScopeType != "user" && item.ScopeType != "team" && item.ScopeType != "key" {
		return Budget{}, errors.New("scopeType must be global, user, team, or key")
	}
	if item.ScopeType == "global" {
		item.ScopeID = ""
	} else if item.ScopeID == "" {
		return Budget{}, errors.New("scopeId is required for non-global budgets")
	}
	if item.RPM < 0 || item.TPM < 0 || item.DailyTokens < 0 {
		return Budget{}, errors.New("quota limits cannot be negative")
	}
	if item.RPM == 0 && item.TPM == 0 && item.DailyTokens == 0 {
		return Budget{}, errors.New("at least one quota limit is required")
	}
	created := item.ID == ""
	if created {
		item.ID = uuid.NewString()
	}
	result, err := s.store.SaveBudget(ctx, item)
	if err == nil {
		s.audit(ctx, actor, choose(created, "budget.created", "budget.updated"), "budget", result.ID, map[string]any{"scopeType": result.ScopeType, "scopeId": result.ScopeID, "rpm": result.RPM, "tpm": result.TPM, "dailyTokens": result.DailyTokens})
	}
	return result, err
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
	message := err.Error()
	for _, marker := range []string{"required", "must be", "cannot be", "valid email", "quota limit"} {
		if strings.Contains(message, marker) {
			return 400, message
		}
	}
	return 500, "access-control operation failed"
}
