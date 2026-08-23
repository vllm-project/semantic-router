package managementstatistics

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

type Options struct {
	Repository     Repository
	Now            func() time.Time
	ExpiringWindow time.Duration
}

type Service struct {
	repository     Repository
	now            func() time.Time
	expiringWindow time.Duration
}

func NewService(options Options) (*Service, error) {
	if options.Repository == nil {
		return nil, ErrUnavailable
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	window := options.ExpiringWindow
	if window == 0 {
		window = DefaultExpiringWindow
	}
	if window < time.Hour || window > 366*24*time.Hour {
		return nil, ErrInvalidRequest
	}
	return &Service{repository: options.Repository, now: now, expiringWindow: window}, nil
}

func (service *Service) Ready(ctx context.Context) error {
	if service == nil || service.repository == nil {
		return ErrUnavailable
	}
	if err := service.repository.Ready(ctx); err != nil {
		return fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
	return nil
}

func (service *Service) Snapshot(ctx context.Context, request Request) (Snapshot, error) {
	if service == nil || service.repository == nil || request.NamespaceID == "" {
		return Snapshot{}, ErrInvalidRequest
	}
	scopes, err := canonicalScopes(request.NamespaceID, request.Scopes)
	if err != nil {
		return Snapshot{}, err
	}
	asOf := service.now().UTC()
	snapshot, err := service.repository.Snapshot(ctx, Query{
		NamespaceID: request.NamespaceID, AsOf: asOf,
		ExpiringBefore: asOf.Add(service.expiringWindow), Scopes: scopes,
	})
	if err != nil {
		return Snapshot{}, err
	}
	if err := validateSnapshot(snapshot, asOf, asOf.Add(service.expiringWindow)); err != nil {
		return Snapshot{}, fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
	return snapshot, nil
}

func canonicalScopes(namespaceID string, scopes Scopes) (Scopes, error) {
	result := Scopes{}
	fields := []struct {
		source **accesscontrol.ResultScope
		target **accesscontrol.ResultScope
	}{
		{&scopes.Users, &result.Users},
		{&scopes.Teams, &result.Teams},
		{&scopes.APIKeys, &result.APIKeys},
		{&scopes.AccessPolicies, &result.AccessPolicies},
		{&scopes.RatePolicies, &result.RatePolicies},
	}
	for _, field := range fields {
		if *field.source == nil {
			continue
		}
		canonical, err := (*field.source).Canonical()
		if err != nil || canonical.NamespaceID != accesscontrol.NamespaceID(namespaceID) {
			return Scopes{}, ErrInvalidRequest
		}
		*field.target = &canonical
	}
	return result, nil
}

func validateSnapshot(snapshot Snapshot, asOf, expiringBefore time.Time) error {
	if !snapshot.AsOf.Equal(asOf) || !snapshot.ExpiringBefore.Equal(expiringBefore) ||
		!snapshot.AsOf.Before(snapshot.ExpiringBefore) {
		return ErrInvalidRequest
	}
	for _, count := range []*Count{
		snapshot.Users, snapshot.Teams, snapshot.ActiveAPIKeys, snapshot.ExpiringAPIKeys,
		snapshot.AccessPolicies, snapshot.ActiveRatePolicies,
	} {
		if count != nil && !count.Valid() {
			return ErrInvalidRequest
		}
	}
	return nil
}
