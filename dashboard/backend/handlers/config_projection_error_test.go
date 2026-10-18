package handlers

import (
	"bytes"
	"database/sql"
	"log"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/configprojection"
)

func TestConfigProjectionHandlersHideStoreErrors(t *testing.T) {
	for _, endpoint := range []struct {
		name, path, table, message string
		handler                    http.HandlerFunc
	}{
		{"list", "/api/router/config/deployments", "config_deployments", "failed to list deployments", ConfigDeploymentsHandler()},
		{"detail", "/api/router/config/deployments/version", "config_deployments", "failed to load deployment", ConfigDeploymentDetailHandler()},
		{"active", "/api/router/config/active-projection", "config_projection_active", "failed to load active projection", ActiveConfigProjectionHandler()},
	} {
		t.Run(endpoint.name, func(t *testing.T) {
			for _, failure := range []string{"closed database", "schema drift"} {
				t.Run(failure, func(t *testing.T) {
					path := filepath.Join(t.TempDir(), "projection.sqlite")
					store, err := configprojection.Open(path)
					if err != nil {
						t.Fatal(err)
					}
					previous := configProjectionStore
					SetConfigProjectionStore(store)
					t.Cleanup(func() { SetConfigProjectionStore(previous); _ = store.Close() })
					storeError := "sql: database is closed"
					if failure == "closed database" {
						if err := store.Close(); err != nil {
							t.Fatal(err)
						}
					} else {
						db, err := sql.Open("sqlite3", path)
						if err != nil {
							t.Fatal(err)
						}
						defer db.Close()
						if _, err := db.Exec("DROP TABLE " + endpoint.table); err != nil {
							t.Fatal(err)
						}
						storeError = "no such table: " + endpoint.table
					}
					var logs bytes.Buffer
					previousLog := log.Writer()
					log.SetOutput(&logs)
					defer log.SetOutput(previousLog)
					response := httptest.NewRecorder()
					endpoint.handler(response, httptest.NewRequest(http.MethodGet, endpoint.path, nil))
					if response.Code != http.StatusInternalServerError || response.Body.String() != endpoint.message+"\n" {
						t.Errorf("status = %d, body = %q; want 500 with %q", response.Code, response.Body.String(), endpoint.message)
					}
					if strings.Contains(response.Body.String(), storeError) {
						t.Errorf("body leaks store error: %q", response.Body.String())
					}
					if !strings.Contains(logs.String(), storeError) {
						t.Errorf("log = %q, want store error %q", logs.String(), storeError)
					}
				})
			}
		})
	}
}

func TestConfigDeploymentDetailNotFoundHidesRequestedVersion(t *testing.T) {
	setupTestConfigProjectionStore(t)
	response := httptest.NewRecorder()
	ConfigDeploymentDetailHandler()(response, httptest.NewRequest(http.MethodGet, "/api/router/config/deployments/private-version", nil))
	if response.Code != http.StatusNotFound || response.Body.String() != "deployment not found\n" {
		t.Fatalf("status = %d, body = %q; want 404 with fixed message", response.Code, response.Body.String())
	}
}
