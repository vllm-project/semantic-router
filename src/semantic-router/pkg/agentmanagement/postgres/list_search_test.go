package postgres

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

func TestResourceListsBindScopeLiteralPrefixAndKeyset(t *testing.T) {
	namespaceID := uuid.NewString()
	resourceID := uuid.NewString()
	after := time.Date(2026, time.August, 23, 12, 30, 0, 0, time.UTC)
	afterID := uuid.NewString()
	queryFor := func(resourceType accesscontrol.ScopeResourceType) agentmanagement.ListQuery {
		return agentmanagement.ListQuery{
			Limit:  21,
			After:  &agentmanagement.Seek{Timestamp: after, ID: afterID},
			Search: `alpha_%`,
			Scope: accesscontrol.ResultScope{
				NamespaceID: accesscontrol.NamespaceID(namespaceID),
				ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
					resourceType: {accesscontrol.ResourceID(resourceID)},
				},
			},
		}
	}
	wantErr := errors.New("query stopped after contract match")

	t.Run("profiles", func(t *testing.T) {
		store, mock := newListSearchStore(t)
		mock.ExpectQuery(`WHERE p\.namespace_id=\$1 AND p\.status<>'deleted'.*lower\(p\.name\) LIKE \$4 ESCAPE.*lower\(p\.description\) LIKE \$4 ESCAPE.*\(p\.created_at,p\.id\)<\(\$5,\$6::uuid\).*ORDER BY p\.created_at DESC,p\.id DESC LIMIT \$7`).
			WithArgs(namespaceID, false, sqlmock.AnyArg(), `alpha\_\%%`, after, afterID, 21).
			WillReturnError(wantErr)
		_, err := store.ListProfiles(context.Background(), namespaceID,
			queryFor(accesscontrol.ScopeResourceAgentProfile))
		assertListSearchError(t, mock, err, wantErr)
	})

	t.Run("skills", func(t *testing.T) {
		store, mock := newListSearchStore(t)
		mock.ExpectQuery(`WHERE \(s\.namespace_id=\$1 OR s\.namespace_id IS NULL\) AND s\.status<>'deleted'.*\(s\.builtin OR \$2 OR s\.id=ANY\(\$3::uuid\[\]\)\).*lower\(s\.name\) LIKE \$4 ESCAPE.*lower\(s\.description\) LIKE \$4 ESCAPE.*\(s\.created_at,s\.id\)<\(\$5,\$6::uuid\).*ORDER BY s\.created_at DESC,s\.id DESC LIMIT \$7`).
			WithArgs(namespaceID, false, sqlmock.AnyArg(), `alpha\_\%%`, after, afterID, 21).
			WillReturnError(wantErr)
		_, err := store.ListSkills(context.Background(), namespaceID,
			queryFor(accesscontrol.ScopeResourceAgentSkill))
		assertListSearchError(t, mock, err, wantErr)
	})

	t.Run("credentials", func(t *testing.T) {
		store, mock := newListSearchStore(t)
		mock.ExpectQuery(`WHERE namespace_id=\$1 AND status<>'deleted' AND \(\$2 OR id=ANY\(\$3::uuid\[\]\)\).*lower\(name\) LIKE \$4 ESCAPE.*\(created_at,id\)<\(\$5,\$6::uuid\).*ORDER BY created_at DESC,id DESC LIMIT \$7`).
			WithArgs(namespaceID, false, sqlmock.AnyArg(), `alpha\_\%%`, after, afterID, 21).
			WillReturnError(wantErr)
		_, err := store.ListToolCredentials(context.Background(), namespaceID,
			queryFor(accesscontrol.ScopeResourceAgentToolCredential))
		assertListSearchError(t, mock, err, wantErr)
	})

	t.Run("sources", func(t *testing.T) {
		store, mock := newListSearchStore(t)
		mock.ExpectQuery(`WHERE s\.namespace_id=\$1 AND s\.status<>'deleted' AND \(\$2 OR s\.id=ANY\(\$3::uuid\[\]\)\).*lower\(s\.name\) LIKE \$4 ESCAPE.*lower\(s\.description\) LIKE \$4 ESCAPE.*\(s\.created_at,s\.id\)<\(\$5,\$6::uuid\).*ORDER BY s\.created_at DESC,s\.id DESC LIMIT \$7`).
			WithArgs(namespaceID, false, sqlmock.AnyArg(), `alpha\_\%%`, after, afterID, 21).
			WillReturnError(wantErr)
		_, err := store.ListToolSources(context.Background(), namespaceID,
			queryFor(accesscontrol.ScopeResourceAgentToolSource))
		assertListSearchError(t, mock, err, wantErr)
	})

	t.Run("sessions", func(t *testing.T) {
		store, mock := newListSearchStore(t)
		ownerID := uuid.NewString()
		query := queryFor(accesscontrol.ScopeResourceAgentSession)
		query.OwnerPrincipalID = ownerID
		query.Scope.TeamIDs = []accesscontrol.TeamID{accesscontrol.TeamID(uuid.NewString())}
		query.Scope.UserIDs = []accesscontrol.UserID{accesscontrol.UserID(uuid.NewString())}
		mock.ExpectQuery(`WHERE session\.namespace_id=\$1 AND session\.status<>'deleted'.*session\.owner_principal_id=\$3.*session\.id=ANY\(\$4::uuid\[\]\).*session\.effective_team_id=ANY\(\$5::uuid\[\]\).*session\.effective_user_id=ANY\(\$6::uuid\[\]\).*lower\(session\.title\) LIKE \$7 ESCAPE.*\(session\.updated_at,session\.id\)<\(\$8,\$9::uuid\).*ORDER BY session\.updated_at DESC,session\.id DESC LIMIT \$10`).
			WithArgs(namespaceID, false, ownerID, sqlmock.AnyArg(), sqlmock.AnyArg(), sqlmock.AnyArg(),
				`alpha\_\%%`, after, afterID, 21).
			WillReturnError(wantErr)
		_, err := store.ListSessions(context.Background(), namespaceID, query)
		assertListSearchError(t, mock, err, wantErr)
	})
}

func newListSearchStore(t *testing.T) (*Store, sqlmock.Sqlmock) {
	t.Helper()
	database, mock, err := sqlmock.New()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = database.Close() })
	return &Store{db: database}, mock
}

func assertListSearchError(t *testing.T, mock sqlmock.Sqlmock, got, want error) {
	t.Helper()
	if !errors.Is(got, want) {
		t.Fatalf("list error = %v, want %v", got, want)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}
