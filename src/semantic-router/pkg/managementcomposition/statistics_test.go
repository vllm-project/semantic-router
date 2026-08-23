package managementcomposition

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managedruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementserver"
)

func TestComposeStatisticsWiresBorrowedDatabaseAndAuthorizationRuntime(t *testing.T) {
	database, mock, err := sqlmock.New(sqlmock.MonitorPingsOption(true))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = database.Close() })
	mock.ExpectPing()
	routes, err := composeStatistics(
		managedruntime.ManagementDependencies{Database: database},
		managementauthorization.Runtime{Loader: statisticsSnapshotLoader{}},
		managementserver.NamespaceResolverFunc(func(context.Context, *http.Request) (string, error) {
			return "11111111-1111-4111-8111-111111111111", nil
		}),
		statisticsSessionAuthenticator{},
		time.Now,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := routes.Ready(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

func TestComposeStatisticsRejectsIncompleteDependencies(t *testing.T) {
	if _, err := composeStatistics(
		managedruntime.ManagementDependencies{}, managementauthorization.Runtime{}, nil, nil, nil,
	); err == nil {
		t.Fatal("composeStatistics accepted incomplete dependencies")
	}
}

type statisticsSnapshotLoader struct{}

func (statisticsSnapshotLoader) Load(
	context.Context,
	accesscontrol.ManagementPrincipalID,
	accesscontrol.NamespaceID,
) (managementauthorization.Snapshot, error) {
	return managementauthorization.Snapshot{}, nil
}

type statisticsSessionAuthenticator struct{}

func (statisticsSessionAuthenticator) Authenticate(
	context.Context,
	string,
	string,
	time.Time,
) (managementauth.AuthenticatedSession, error) {
	return managementauth.AuthenticatedSession{}, nil
}

var (
	_ managementauthorization.SnapshotLoader = statisticsSnapshotLoader{}
	_ managementserver.SessionAuthenticator  = statisticsSessionAuthenticator{}
)
