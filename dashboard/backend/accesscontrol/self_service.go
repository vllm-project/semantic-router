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
	items, _, err := s.ListAPIKeys(ctx, ListFilter{UserID: userID, Limit: 100})
	if err != nil {
		return nil, err
	}
	// Self-service users can own at most one key. Resolve its effective model
	// visibility so the page reflects Team inheritance without exposing policy
	// definitions from other Teams.
	for index := range items {
		items[index].ModelPatterns, err = s.store.ModelPatternsForKey(ctx, items[index])
		if err != nil {
			return nil, err
		}
	}
	return items, nil
}

func (s *Service) GetSelfAPIKey(ctx context.Context, userID, id string) (APIKey, error) {
	item, err := s.GetAPIKey(ctx, id)
	if err != nil || item.UserID != userID {
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

func (s *Service) CreateSelfAPIKey(ctx context.Context, actor Actor, name string) (CreatedAPIKey, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		name = "My API key"
	}
	secret, prefix, err := NewSecret()
	if err != nil {
		return CreatedAPIKey{}, err
	}
	digest, err := DigestKey(secret, s.keySecret)
	if err != nil {
		return CreatedAPIKey{}, err
	}
	item := APIKey{ID: uuid.NewString(), Name: name, UserID: actor.ID, Prefix: prefix, Status: StatusActive}
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
		return APIKey{}, errors.New("status must be active or disabled")
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

func (s *Service) SelfUsage(ctx context.Context, userID string, filter ListFilter) (UsageSummary, error) {
	filter.UserID, filter.TeamID = userID, ""
	return s.store.UsageSummary(ctx, filter)
}

func (s *Service) SelfRequestLogs(ctx context.Context, userID string, filter ListFilter) ([]UsageEvent, int64, error) {
	filter.UserID, filter.TeamID = userID, ""
	items, err := s.store.ListUsage(ctx, filter)
	if err != nil {
		return nil, 0, err
	}
	total, err := s.store.CountUsage(ctx, filter)
	return items, total, err
}

func (s *Service) SelfRequestLog(ctx context.Context, userID, id string) (UsageEvent, error) {
	item, err := s.store.GetUsage(ctx, id)
	if err != nil || item.UserID != userID {
		return UsageEvent{}, pgx.ErrNoRows
	}
	return item, nil
}
