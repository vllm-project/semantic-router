package managementstatistics

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

func TestServiceCanonicalizesAuthorizedScopesAndOwnsExpiryWindow(t *testing.T) {
	now := time.Date(2026, 8, 23, 12, 0, 0, 0, time.FixedZone("test", 8*60*60))
	repository := &repositoryStub{}
	service, err := NewService(Options{Repository: repository, Now: func() time.Time { return now }})
	if err != nil {
		t.Fatal(err)
	}
	namespaceID := "10000000-0000-4000-8000-000000000001"
	userScope := accesscontrol.ResultScope{
		NamespaceID: accesscontrol.NamespaceID(namespaceID),
		UserIDs:     []accesscontrol.UserID{"20000000-0000-4000-8000-000000000002", "20000000-0000-4000-8000-000000000001", "20000000-0000-4000-8000-000000000001"},
	}
	result, err := service.Snapshot(context.Background(), Request{
		NamespaceID: namespaceID,
		Scopes:      Scopes{Users: &userScope},
	})
	if err != nil {
		t.Fatal(err)
	}
	if repository.query.AsOf.Location() != time.UTC ||
		repository.query.ExpiringBefore.Sub(repository.query.AsOf) != DefaultExpiringWindow {
		t.Fatalf("query time contract = %#v", repository.query)
	}
	if got := repository.query.Scopes.Users.UserIDs; len(got) != 2 || got[0] >= got[1] {
		t.Fatalf("canonical user scope = %#v", got)
	}
	if result.Users == nil || *result.Users != "2" || result.Teams != nil {
		t.Fatalf("snapshot = %#v", result)
	}
}

func TestServiceRejectsCrossNamespaceAndInvalidRepositoryResults(t *testing.T) {
	namespaceID := "10000000-0000-4000-8000-000000000001"
	other := accesscontrol.ResultScope{NamespaceID: "10000000-0000-4000-8000-000000000002", All: true}
	service, _ := NewService(Options{Repository: &repositoryStub{}})
	if _, err := service.Snapshot(context.Background(), Request{
		NamespaceID: namespaceID,
		Scopes:      Scopes{Users: &other},
	}); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("cross-namespace error = %v", err)
	}

	service.repository = &repositoryStub{invalid: true}
	all := accesscontrol.ResultScope{NamespaceID: accesscontrol.NamespaceID(namespaceID), All: true}
	if _, err := service.Snapshot(context.Background(), Request{
		NamespaceID: namespaceID,
		Scopes:      Scopes{Users: &all},
	}); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("invalid repository error = %v", err)
	}
}

type repositoryStub struct {
	query   Query
	invalid bool
}

func (repository *repositoryStub) Ready(context.Context) error { return nil }

func (repository *repositoryStub) Snapshot(_ context.Context, query Query) (Snapshot, error) {
	repository.query = query
	count := Count("2")
	if repository.invalid {
		count = "02"
	}
	return Snapshot{AsOf: query.AsOf, ExpiringBefore: query.ExpiringBefore, Users: &count}, nil
}
