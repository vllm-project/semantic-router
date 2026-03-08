package configlifecycle

import (
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

func TestCurrentRevisionReturnsPersistedDetail(t *testing.T) {
	tempDir := t.TempDir()
	configPath := writePersistenceTestConfig(t, tempDir)
	stores := openPersistenceTestStores(t, tempDir)
	service := NewWithStores(configPath, tempDir, stores)

	activeRevisionID := saveRevisionFixture(t, stores, console.ConfigRevisionStatusActive, "revision_current", persistedActiveConfig, time.Now().UTC())
	saveDeployFixture(t, stores, activeRevisionID, console.DeployEventStatusSucceeded, "workspace-files")

	revision, err := service.CurrentRevision()
	if err != nil {
		t.Fatalf("CurrentRevision() error = %v", err)
	}
	if revision == nil {
		t.Fatal("expected current revision")
	}
	if revision.ID != activeRevisionID {
		t.Fatalf("expected revision %q, got %q", activeRevisionID, revision.ID)
	}
	if revision.LastDeployStatus != console.DeployEventStatusSucceeded {
		t.Fatalf("expected succeeded deploy status, got %q", revision.LastDeployStatus)
	}
	if revision.RuntimeTarget != "workspace-files" {
		t.Fatalf("expected runtime target workspace-files, got %q", revision.RuntimeTarget)
	}
	if revision.Document == nil {
		t.Fatal("expected revision document")
	}
}

func TestListRevisionsIncludesPersistedHistory(t *testing.T) {
	tempDir := t.TempDir()
	configPath := writePersistenceTestConfig(t, tempDir)
	stores := openPersistenceTestStores(t, tempDir)
	service := NewWithStores(configPath, tempDir, stores)

	older := time.Now().UTC().Add(-1 * time.Minute)
	saveRevisionFixture(t, stores, console.ConfigRevisionStatusSuperseded, "revision_old", persistenceTestConfig, older)
	activeRevisionID := saveRevisionFixture(t, stores, console.ConfigRevisionStatusActive, "revision_active", persistedActiveConfig, time.Now().UTC())
	saveDeployFixture(t, stores, activeRevisionID, console.DeployEventStatusSucceeded, "workspace-files")

	revisions, err := service.ListRevisions(10)
	if err != nil {
		t.Fatalf("ListRevisions() error = %v", err)
	}
	if len(revisions) != 2 {
		t.Fatalf("expected 2 revisions, got %d", len(revisions))
	}
	if revisions[0].ID != activeRevisionID {
		t.Fatalf("expected newest active revision first, got %q", revisions[0].ID)
	}
	if revisions[0].LastDeployStatus != console.DeployEventStatusSucceeded {
		t.Fatalf("expected deploy status on latest revision, got %q", revisions[0].LastDeployStatus)
	}
	if revisions[1].Status != console.ConfigRevisionStatusSuperseded {
		t.Fatalf("expected superseded older revision, got %q", revisions[1].Status)
	}
}

func saveRevisionFixture(
	t *testing.T,
	stores *console.Stores,
	status console.ConfigRevisionStatus,
	source string,
	yamlConfig string,
	createdAt time.Time,
) string {
	t.Helper()

	documentJSON, err := marshalRevisionDocument([]byte(yamlConfig))
	if err != nil {
		t.Fatalf("marshalRevisionDocument() error = %v", err)
	}

	revision := &console.ConfigRevision{
		Status:            status,
		Source:            source,
		Summary:           "test revision fixture",
		DocumentJSON:      documentJSON,
		RuntimeConfigYAML: yamlConfig,
		CreatedBy:         "test",
		Metadata: map[string]interface{}{
			"runtime_target": "workspace-files",
		},
		CreatedAt: createdAt,
		UpdatedAt: createdAt,
	}
	if status == console.ConfigRevisionStatusActive {
		revision.ActivatedAt = &createdAt
	}

	if err := stores.Revisions.SaveConfigRevision(context.Background(), revision); err != nil {
		t.Fatalf("SaveConfigRevision() error = %v", err)
	}
	return revision.ID
}

func saveDeployFixture(
	t *testing.T,
	stores *console.Stores,
	revisionID string,
	status console.DeployEventStatus,
	runtimeTarget string,
) {
	t.Helper()

	now := time.Now().UTC()
	if err := stores.Deployments.SaveDeployEvent(context.Background(), &console.DeployEvent{
		RevisionID:    revisionID,
		Status:        status,
		TriggerSource: "test",
		Message:       "test deploy fixture",
		RuntimeTarget: runtimeTarget,
		StartedAt:     &now,
		CompletedAt:   &now,
		CreatedAt:     now,
		UpdatedAt:     now,
	}); err != nil {
		t.Fatalf("SaveDeployEvent() error = %v", err)
	}
}
