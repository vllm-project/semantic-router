package accesscontrol

import (
	"context"
	"errors"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
)

var ErrSelfAPIKeyExists = errors.New("you already have an API key")

func (s *Service) ListSelfAPIKeys(ctx context.Context, userID string) ([]APIKey, error) {
	items, err := s.store.ListAPIKeysForUser(ctx, userID)
	if err != nil {
		return nil, err
	}
	// A user can own one personal key and administer shared keys for their Teams.
	// Resolve effective visibility without exposing policy definitions outside
	// those explicit relationships.
	return s.store.ResolveAPIKeyPolicies(ctx, items)
}

func (s *Service) GetSelfAPIKey(ctx context.Context, userID, id string) (APIKey, error) {
	item, err := s.GetAPIKey(ctx, id)
	if err != nil {
		return APIKey{}, err
	}
	if item.UserID == userID {
		return item, nil
	}
	allowed, allowedErr := s.store.IsTeamAdmin(ctx, userID, item.TeamID)
	if allowedErr != nil {
		return APIKey{}, allowedErr
	}
	if !allowed {
		return APIKey{}, pgx.ErrNoRows
	}
	return item, nil
}

func (s *Service) RevealSelfAPIKey(ctx context.Context, actor Actor, id string) (string, error) {
	if _, err := s.GetSelfAPIKey(ctx, actor.ID, id); err != nil {
		return "", err
	}
	return s.RevealAPIKey(ctx, actor, id)
}

func (s *Service) RotateSelfAPIKey(ctx context.Context, actor Actor, id string) (CreatedAPIKey, error) {
	if _, err := s.GetSelfAPIKey(ctx, actor.ID, id); err != nil {
		return CreatedAPIKey{}, err
	}
	return s.RotateAPIKey(ctx, actor, id)
}

func (s *Service) CreateSelfAPIKey(
	ctx context.Context,
	actor Actor,
	name, ownerType, ownerID, contextTeamID string,
) (CreatedAPIKey, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		name = "My API key"
	}
	ownerType = strings.ToLower(strings.TrimSpace(ownerType))
	if ownerType == "" {
		ownerType = "user"
	}
	if ownerType == "team" {
		ownerID = strings.TrimSpace(ownerID)
		allowed, err := s.store.IsTeamAdmin(ctx, actor.ID, ownerID)
		if err != nil {
			return CreatedAPIKey{}, err
		}
		if !allowed {
			return CreatedAPIKey{}, validationError("Team admin role is required to create a Team key")
		}
		return s.CreateAPIKey(ctx, actor, APIKey{
			Name: name, OwnerType: "team", OwnerID: ownerID, Status: StatusActive,
		})
	}
	if ownerType != "user" || (strings.TrimSpace(ownerID) != "" && ownerID != actor.ID) {
		return CreatedAPIKey{}, validationError("personal keys must be owned by the signed-in user")
	}
	secret, prefix, err := NewSecret()
	if err != nil {
		return CreatedAPIKey{}, err
	}
	digest, err := DigestKey(secret, s.keySecret)
	if err != nil {
		return CreatedAPIKey{}, err
	}
	item := APIKey{ID: uuid.NewString(), Name: name, UserID: actor.ID, OwnerType: "user", OwnerID: actor.ID, ContextTeamID: strings.TrimSpace(contextTeamID), Prefix: prefix, Status: StatusActive}
	if err = s.normalizeAndValidateKeyOwner(ctx, &item); err != nil {
		return CreatedAPIKey{}, err
	}
	ciphertext, err := EncryptKeySecret(secret, s.keySecret, item.ID)
	if err != nil {
		return CreatedAPIKey{}, err
	}
	created, err := s.store.CreateSelfAPIKey(ctx, item, digest, ciphertext)
	if err != nil {
		return CreatedAPIKey{}, err
	}
	s.audit(ctx, actor, "api_key.self_created", "api_key", created.ID, map[string]any{
		"prefix": created.Prefix, "userId": actor.ID,
	})
	return CreatedAPIKey{APIKey: created, Secret: secret}, nil
}

func (s *Service) SetSelfAPIKeyStatus(ctx context.Context, actor Actor, id, status string) (APIKey, error) {
	if status != StatusActive && status != StatusDisabled {
		return APIKey{}, validationError("status must be active or disabled")
	}
	item, err := s.GetAPIKey(ctx, id)
	if err != nil || item.UserID != actor.ID {
		return APIKey{}, pgx.ErrNoRows
	}
	item, err = s.store.SetAPIKeyStatus(ctx, id, status)
	if err == nil {
		s.audit(ctx, actor, "api_key.self_status_changed", "api_key", id, map[string]any{"status": status})
	}
	return item, err
}

func (s *Service) SelfOverview(ctx context.Context, userID string) (Overview, error) {
	return s.OverviewForUser(ctx, userID)
}

func (s *Service) SelfTeams(ctx context.Context, userID string) ([]Team, error) {
	return s.store.ListTeamsForUser(ctx, userID)
}

func (s *Service) SelfTeamCatalog(ctx context.Context, userID string) (SelfTeamCatalog, error) {
	teams, err := s.store.ListTeamsForUser(ctx, userID)
	if err != nil {
		return SelfTeamCatalog{}, err
	}
	members, err := s.store.ListUsersSharingTeam(ctx, userID)
	if err != nil {
		return SelfTeamCatalog{}, err
	}
	groups, budgets, err := s.store.ListPoliciesForUserTeams(ctx, userID)
	if err != nil {
		return SelfTeamCatalog{}, err
	}
	return SelfTeamCatalog{
		Teams: teams, Members: members, AccessGroups: groups, Budgets: budgets,
	}, nil
}

func (s *Service) GetSelfTeam(ctx context.Context, userID, teamID string) (Team, error) {
	return s.store.GetTeamForUser(ctx, userID, teamID)
}

// SaveSelfTeam lets Team admins manage identity fields and membership roles.
// Model grants, quota, status, and deletion remain platform-admin controls.
func (s *Service) SaveSelfTeam(ctx context.Context, actor Actor, item Team) (Team, error) {
	var err error
	item.Members, err = normalizeTeamMemberships(item.ID, item.Members)
	if err != nil {
		return Team{}, err
	}
	current, err := s.store.GetTeamForUser(ctx, actor.ID, item.ID)
	if err != nil {
		return Team{}, err
	}
	allowed, err := s.store.IsTeamAdmin(ctx, actor.ID, item.ID)
	if err != nil {
		return Team{}, err
	}
	if !allowed {
		return Team{}, validationError("Team admin role is required")
	}
	visibleMembers, err := s.store.ListUsersSharingTeam(ctx, actor.ID)
	if err != nil {
		return Team{}, err
	}
	visibleMemberIDs := make(map[string]struct{}, len(visibleMembers))
	for _, member := range visibleMembers {
		visibleMemberIDs[member.ID] = struct{}{}
	}
	hasActorAdmin, hasAdmin := false, false
	for _, membership := range item.Members {
		if _, visible := visibleMemberIDs[membership.UserID]; !visible {
			return Team{}, validationError("Team members must already share a Team with you")
		}
		if membership.Role != TeamRoleAdmin && membership.Role != TeamRoleMember {
			return Team{}, validationError("Team role must be admin or member")
		}
		if membership.Role == TeamRoleAdmin {
			hasAdmin = true
			hasActorAdmin = hasActorAdmin || membership.UserID == actor.ID
		}
	}
	if !hasAdmin || !hasActorAdmin {
		return Team{}, validationError("you must remain a Team admin")
	}
	current.Name = strings.TrimSpace(item.Name)
	current.Description = strings.TrimSpace(item.Description)
	current.Members = item.Members
	return s.SaveTeam(ctx, actor, current)
}

func (s *Service) SelfUsage(ctx context.Context, userID string, filter ListFilter) (UsageSummary, error) {
	if filter.KeyID != "" {
		if _, err := s.GetSelfAPIKey(ctx, userID, filter.KeyID); err != nil {
			return UsageSummary{}, err
		}
		filter.UserID, filter.TeamID = "", ""
		return s.store.UsageSummary(ctx, filter)
	}
	filter.UserID, filter.TeamID = userID, ""
	return s.store.UsageSummary(ctx, filter)
}

func (s *Service) SelfRequestLogs(ctx context.Context, userID string, filter ListFilter) ([]UsageEvent, int64, error) {
	if filter.KeyID != "" {
		if _, err := s.GetSelfAPIKey(ctx, userID, filter.KeyID); err != nil {
			return nil, 0, err
		}
		filter.UserID, filter.TeamID = "", ""
	} else {
		filter.UserID, filter.TeamID = userID, ""
	}
	items, err := s.store.ListUsage(ctx, filter)
	if err != nil {
		return nil, 0, err
	}
	total, err := s.store.CountUsage(ctx, filter)
	return items, total, err
}

func (s *Service) SelfRequestLog(ctx context.Context, userID, id string) (UsageEvent, error) {
	item, err := s.store.GetUsage(ctx, id)
	if err != nil {
		return UsageEvent{}, err
	}
	if item.UserID == userID {
		return item, nil
	}
	if _, err = s.GetSelfAPIKey(ctx, userID, item.KeyID); err != nil {
		return UsageEvent{}, pgx.ErrNoRows
	}
	return item, nil
}
