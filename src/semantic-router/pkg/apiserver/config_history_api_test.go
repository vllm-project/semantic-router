//go:build !windows

package apiserver

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

// newHistoryAPIServer runs the API beside the Router's lifecycle, with the
// history persisted in the directory this API has kept its backups in.
func newHistoryAPIServer(t *testing.T, reject error) (*ClassificationAPIServer, *configsnapshot.Manager, string) {
	t.Helper()
	path := writeDeployTestBaseConfig(t)
	startup, err := config.Parse(path)
	if err != nil {
		t.Fatal(err)
	}
	document, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	registry := routerruntime.NewRegistry(startup)
	history, err := configsnapshot.NewHistory(configsnapshot.DefaultHistoryLimit,
		configsnapshot.NewDirStore(configBackupDir(resolveConfigPersistencePaths(path).sourcePath)))
	if err != nil {
		t.Fatal(err)
	}
	manager := configsnapshot.NewManager(configsnapshot.Options{
		Runtime: publishingRuntime{registry: registry, reject: reject}, Reporter: registry, History: history,
	})
	registry.SetConfigLifecycle(manager)
	server := &ClassificationAPIServer{configPath: path, runtimeRegistry: registry}
	registry.OnConfigAttempt(server.recordConfigAudit)
	snapshot, err := manager.Install(context.Background(), configsnapshot.Update{
		Origin: configsnapshot.Origin{Source: configsnapshot.SourceStartup}, Config: startup, Document: document,
	})
	if err != nil {
		t.Fatal(err)
	}
	registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{Config: startup, ConfigSnapshot: snapshot})
	return server, manager, path
}

func listVersions(t *testing.T, server *ClassificationAPIServer) []RouterConfigVersionEntry {
	t.Helper()
	recorder := httptest.NewRecorder()
	server.setupRoutes().ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, apiConfigVersionsPath, nil))
	var entries []RouterConfigVersionEntry
	if err := json.Unmarshal(recorder.Body.Bytes(), &entries); err != nil {
		t.Fatalf("decode %s: %v", recorder.Body.String(), err)
	}
	return entries
}

func rollbackTo(t *testing.T, server *ClassificationAPIServer, path string, payload any) (*httptest.ResponseRecorder, RouterConfigUpdateResponse) {
	t.Helper()
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, apiConfigRollbackPath, bytes.NewReader(body))
	setConfigPrecondition(t, request, path)
	recorder := httptest.NewRecorder()
	server.setupRoutes().ServeHTTP(recorder, request)
	var result RouterConfigUpdateResponse
	_ = json.Unmarshal(recorder.Body.Bytes(), &result)
	return recorder, result
}

func TestVersionsListTheHistoryOnce(t *testing.T) {
	server, manager, path := newHistoryAPIServer(t, nil)
	watchAndApply(t, manager, path)
	if recorder, result := putConfig(t, server, path); recorder.Code != http.StatusOK || result.ConfigVersion != 2 {
		t.Fatalf("PUT: %d %s", recorder.Code, recorder.Body.String())
	}

	entries := listVersions(t, server)
	if len(entries) != 2 {
		t.Fatalf("versions = %+v, want the startup version and the PUT, the replaced document recorded once", entries)
	}
	put, startup := entries[0], entries[1]
	if put.ConfigVersion != 2 || !put.Active || put.Source != configVersionSourceAPI || put.Filename == "" ||
		put.Timestamp == "" || put.Hash == "" {
		t.Fatalf("newest entry = %+v", put)
	}
	if startup.ConfigVersion != 1 || startup.Active || startup.Source != configVersionSourceStartup {
		t.Fatalf("oldest entry = %+v", startup)
	}
	documents, _ := filepath.Glob(filepath.Join(configBackupDir(resolveConfigPersistencePaths(path).sourcePath), "config.*.yaml"))
	if len(documents) != 2 {
		t.Fatalf("the history directory holds %d documents, want 2", len(documents))
	}
}

func TestRollbackActivatesARecordedVersionAsANewOne(t *testing.T) {
	for _, payload := range []any{
		routerConfigRollbackRequest{Version: "1"},
		routerConfigRollbackRequest{ConfigVersion: 1},
	} {
		server, manager, path := newHistoryAPIServer(t, nil)
		startupDocument, _ := os.ReadFile(path)
		watchAndApply(t, manager, path)
		if recorder, _ := putConfig(t, server, path); recorder.Code != http.StatusOK {
			t.Fatalf("PUT: %d %s", recorder.Code, recorder.Body.String())
		}

		watchAndApply(t, manager, path)
		recorder, result := rollbackTo(t, server, path, payload)
		if recorder.Code != http.StatusOK || result.ConfigVersion != 3 || result.RollbackOf != 1 {
			t.Fatalf("rollback %+v: %d %s", payload, recorder.Code, recorder.Body.String())
		}
		if restored, _ := os.ReadFile(path); !bytes.Equal(restored, startupDocument) {
			t.Fatal("the rollback did not persist the recorded document")
		}
		active := manager.Active()
		if active.Version() != 3 || active.Origin().Source != configsnapshot.SourceRollback || active.Origin().RollbackOf != 1 {
			t.Fatalf("active snapshot = v%d %+v", active.Version(), active.Origin())
		}
		entries := listVersions(t, server)
		if len(entries) != 3 || entries[0].RollbackOf != 1 || entries[0].Source != configVersionSourceRollback {
			t.Fatalf("versions after rollback = %+v", entries)
		}

		putDocument, _ := manager.History().ByVersion(2)
		watchAndApply(t, manager, path)
		recorder, result = rollbackTo(t, server, path, routerConfigRollbackRequest{ConfigVersion: 2})
		if recorder.Code != http.StatusOK || result.ConfigVersion != 4 || result.RollbackOf != 2 {
			t.Fatalf("roll forward: %d %s", recorder.Code, recorder.Body.String())
		}
		if restored, _ := os.ReadFile(path); !bytes.Equal(restored, putDocument.Document) {
			t.Fatal("rolling forward did not restore version 2's document")
		}
		if entries := listVersions(t, server); len(entries) != 4 || !entries[0].Active || entries[0].ConfigVersion != 4 {
			t.Fatalf("versions after rolling forward = %+v", entries)
		}
	}
}

// rejectedFileEdits are edits of the watched file that the Router rejects at
// parse: the file keeps them, and the version before them keeps serving.
var rejectedFileEdits = map[string]string{
	"YAML that does not parse":     "version: v0.3\nrouting: {decisions: [\n",
	"a document out of its layout": "version: v0.3\nrouting: [not, a, map]\n",
}

// rejectFileEdit writes document to the watched file and rejects it as the
// Router's file watcher does.
func rejectFileEdit(t *testing.T, manager *configsnapshot.Manager, path, document string) {
	t.Helper()
	serving := manager.Active().Version()
	if err := os.WriteFile(path, []byte(document), 0o644); err != nil {
		t.Fatal(err)
	}
	_, parseErr := config.Parse(path)
	if parseErr == nil {
		t.Fatalf("the edit %q parses", document)
	}
	_ = manager.Reject(configsnapshot.Update{
		Origin: configsnapshot.Origin{Source: configsnapshot.SourceFile}, Document: []byte(document),
	}, configsnapshot.Reject(configsnapshot.StageParse, configsnapshot.CodeInvalidDocument, parseErr))
	if manager.Active().Version() != serving {
		t.Fatal("the rejected edit replaced the serving version")
	}
}

// The restored document is checked against the version that serves, not
// against the rejected document the file still holds.
func TestRollbackReplacesARejectedDocument(t *testing.T) {
	for name, document := range rejectedFileEdits {
		t.Run(name, func(t *testing.T) {
			server, manager, path := newHistoryAPIServer(t, nil)
			startupDocument, _ := os.ReadFile(path)
			watchAndApply(t, manager, path)
			if recorder, _ := putConfig(t, server, path); recorder.Code != http.StatusOK {
				t.Fatalf("PUT: %d %s", recorder.Code, recorder.Body.String())
			}
			rejectFileEdit(t, manager, path, document)

			watchAndApply(t, manager, path)
			recorder, result := rollbackTo(t, server, path, routerConfigRollbackRequest{ConfigVersion: 1})
			if recorder.Code != http.StatusOK || result.ConfigVersion != 3 || result.RollbackOf != 1 {
				t.Fatalf("rollback: %d %s", recorder.Code, recorder.Body.String())
			}
			if restored, _ := os.ReadFile(path); !bytes.Equal(restored, startupDocument) {
				t.Fatal("the rollback did not persist version 1's document")
			}
		})
	}
}

// A replacement corrects a rejected document, even one that does not parse.
func TestReplacingARejectedDocumentActivatesTheNextVersion(t *testing.T) {
	for name, document := range rejectedFileEdits {
		t.Run(name, func(t *testing.T) {
			server, manager, path := newHistoryAPIServer(t, nil)
			rejectFileEdit(t, manager, path, document)

			watchAndApply(t, manager, path)
			if recorder, result := putConfig(t, server, path); recorder.Code != http.StatusOK || result.ConfigVersion != 2 {
				t.Fatalf("PUT: %d %s", recorder.Code, recorder.Body.String())
			}
		})
	}
}

func TestRollbackRefusesAVersionWithoutADocument(t *testing.T) {
	server, manager, path := newHistoryAPIServer(t, nil)
	cfg := &config.RouterConfig{DocumentHash: "crd-document"}
	if _, err := manager.Apply(context.Background(), configsnapshot.Update{
		Origin: configsnapshot.Origin{Source: configsnapshot.SourceKubernetes}, Config: cfg,
	}); err != nil {
		t.Fatal(err)
	}
	recorder, _ := rollbackTo(t, server, path, routerConfigRollbackRequest{ConfigVersion: 2})
	if recorder.Code != http.StatusConflict {
		t.Fatalf("rollback to a Kubernetes version: %d %s", recorder.Code, recorder.Body.String())
	}
	recorder, _ = rollbackTo(t, server, path, routerConfigRollbackRequest{ConfigVersion: 9})
	if recorder.Code != http.StatusNotFound {
		t.Fatalf("rollback to an unknown version: %d %s", recorder.Code, recorder.Body.String())
	}
	recorder, _ = rollbackTo(t, server, path, routerConfigRollbackRequest{Version: "1", ConfigVersion: 1})
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("rollback naming two versions: %d %s", recorder.Code, recorder.Body.String())
	}
}

func readAudit(t *testing.T, server *ClassificationAPIServer, action RouteAuditAction) []managementAuditEntry {
	t.Helper()
	return server.managementAuditPage(0, 100, action).Entries
}

func TestConfigAuditRecordsWhoChangedWhatAndHowItEnded(t *testing.T) {
	server, manager, path := newHistoryAPIServer(t, nil)
	watchAndApply(t, manager, path)
	recorder, _ := putConfig(t, server, path)
	if recorder.Code != http.StatusOK {
		t.Fatalf("PUT: %d %s", recorder.Code, recorder.Body.String())
	}
	requestID := recorder.Header().Get(managementRequestIDHeader)

	activations := readAudit(t, server, AuditActionConfigActivate)
	if len(activations) != 2 {
		t.Fatalf("activation audit = %+v, want startup and the PUT", activations)
	}
	startup, put := activations[0].Config, activations[1]
	if startup.Source != "startup" || startup.Version != 1 || startup.Result != "active" {
		t.Fatalf("startup audit = %+v", startup)
	}
	if put.Config.Version != 2 || put.Config.Source != "api" || put.RequestID != requestID || put.Method != "" || put.Status != 0 {
		t.Fatalf("PUT activation audit = %+v %+v", put, put.Config)
	}
	if requests := readAudit(t, server, AuditActionConfigPut); len(requests) != 1 || requests[0].RequestID != requestID ||
		requests[0].Status != http.StatusOK {
		t.Fatalf("PUT request audit = %+v", requests)
	}
	all := server.managementAuditPage(0, 100, "").Entries
	for i := 1; i < len(all); i++ {
		if all[i].PreviousHash != all[i-1].Hash {
			t.Fatalf("audit chain broken at %d: %+v", i, all[i])
		}
	}
}

func TestConfigAuditRecordsARejection(t *testing.T) {
	rejection := configsnapshot.Reject(configsnapshot.StageWarm, configsnapshot.CodeModelUnavailable, errors.New("model missing"))
	server, manager, path := newHistoryAPIServer(t, rejection)
	watchAndApply(t, manager, path)
	if recorder, _ := putConfig(t, server, path); recorder.Code != http.StatusServiceUnavailable {
		t.Fatalf("PUT: %d %s", recorder.Code, recorder.Body.String())
	}
	rejections := readAudit(t, server, AuditActionConfigReject)
	if len(rejections) != 1 {
		t.Fatalf("rejection audit = %+v", rejections)
	}
	event := rejections[0].Config
	if event.Result != "failed" || event.Stage != "warm" || len(event.Codes) != 1 || event.Codes[0] != "model_unavailable" ||
		event.Version != 0 || event.Source != "api" {
		t.Fatalf("rejection event = %+v", event)
	}
	if entries := listVersions(t, server); len(entries) != 1 {
		t.Fatalf("a rejected update entered the history: %+v", entries)
	}
}
