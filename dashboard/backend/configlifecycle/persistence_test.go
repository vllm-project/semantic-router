package configlifecycle

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

const persistenceTestConfig = `
bert_model:
  model_id: models/all-MiniLM-L12-v2
  threshold: 0.6
  use_cpu: true

classifier:
  category_model:
    model_id: models/lora_intent_classifier_bert-base-uncased_model
    threshold: 0.6
    use_cpu: true
    category_mapping_path: models/lora_intent_classifier_bert-base-uncased_model/category_mapping.json
  pii_model:
    model_id: models/lora_pii_detector_bert-base-uncased_model
    threshold: 0.9
    use_cpu: true
    pii_mapping_path: models/pii_classifier_modernbert-base_presidio_token_model/pii_type_mapping.json

categories:
  - name: business
    description: Business and management related queries

vllm_endpoints:
  - name: endpoint1
    address: 127.0.0.1
    port: 8000
    weight: 1

default_model: test-model

model_config:
  test-model:
    reasoning_family: qwen3
`

func TestUpdateConfigPersistsCompatibilityRevision(t *testing.T) {
	tempDir := t.TempDir()
	configPath := writePersistenceTestConfig(t, tempDir)
	stores := openPersistenceTestStores(t, tempDir)

	service := NewWithStores(configPath, tempDir, stores)
	if err := service.UpdateConfig(map[string]interface{}{
		"ui_test_flag": true,
	}); err != nil {
		t.Fatalf("UpdateConfig() error = %v", err)
	}

	ctx := context.Background()
	revisions, err := stores.Revisions.ListConfigRevisions(ctx, console.ConfigRevisionFilter{Limit: 10})
	if err != nil {
		t.Fatalf("ListConfigRevisions() error = %v", err)
	}
	if len(revisions) != 1 {
		t.Fatalf("expected 1 config revision, got %d", len(revisions))
	}
	revision := revisions[0]
	if revision.Status != console.ConfigRevisionStatusActive {
		t.Fatalf("expected active revision status, got %q", revision.Status)
	}
	if revision.Source != "compat_config_update" {
		t.Fatalf("expected compat_config_update source, got %q", revision.Source)
	}
	if revision.CreatedBy != compatibilityActorID {
		t.Fatalf("expected compat actor %q, got %q", compatibilityActorID, revision.CreatedBy)
	}
	if revision.RuntimeConfigYAML == "" {
		t.Fatal("expected runtime_config_yaml to be persisted")
	}
	if stringMetadata(revision.Metadata, "trigger_source") != "config_api" {
		t.Fatalf("expected config_api trigger source, got %#v", revision.Metadata["trigger_source"])
	}

	deployEvents, err := stores.Deployments.ListDeployEvents(ctx, console.DeployEventFilter{
		RevisionID: revision.ID,
		Limit:      10,
	})
	if err != nil {
		t.Fatalf("ListDeployEvents() error = %v", err)
	}
	if len(deployEvents) != 1 {
		t.Fatalf("expected 1 deploy event, got %d", len(deployEvents))
	}
	if deployEvents[0].Status != console.DeployEventStatusSucceeded {
		t.Fatalf("expected deploy event status succeeded, got %q", deployEvents[0].Status)
	}

	auditEvents, err := stores.Audit.ListAuditEvents(ctx, console.AuditEventFilter{
		Action: "config.update",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListAuditEvents() error = %v", err)
	}
	if len(auditEvents) != 1 {
		t.Fatalf("expected 1 audit event, got %d", len(auditEvents))
	}
	assertRevisionAuditCount(t, stores, "config.save_draft", console.AuditOutcomeSuccess, 1)
	assertRevisionAuditCount(t, stores, "config.validate_revision", console.AuditOutcomeSuccess, 1)
	assertRevisionAuditCount(t, stores, "config.activate_revision", console.AuditOutcomeSuccess, 1)
}

func TestDeployAndRollbackPersistRevisionHistory(t *testing.T) {
	tempDir := t.TempDir()
	configPath := writePersistenceTestConfig(t, tempDir)
	stores := openPersistenceTestStores(t, tempDir)

	service := NewWithStores(configPath, tempDir, stores)
	deployResult, err := service.Deploy(DeployRequest{
		YAML: `
default_model: test-model
model_config:
  test-model:
    reasoning_family: qwen3
    temperature: 0.25
`,
	})
	if err != nil {
		t.Fatalf("Deploy() error = %v", err)
	}

	ctx := context.Background()
	revisions, err := stores.Revisions.ListConfigRevisions(ctx, console.ConfigRevisionFilter{Limit: 10})
	if err != nil {
		t.Fatalf("ListConfigRevisions() after deploy error = %v", err)
	}
	if len(revisions) != 1 {
		t.Fatalf("expected 1 config revision after deploy, got %d", len(revisions))
	}
	deployRevision := revisions[0]
	if deployRevision.Status != console.ConfigRevisionStatusActive {
		t.Fatalf("expected deploy revision to be active, got %q", revisions[0].Status)
	}
	if deployRevision.CreatedBy != compatibilityActorID {
		t.Fatalf("expected compat actor %q, got %q", compatibilityActorID, deployRevision.CreatedBy)
	}
	if stringMetadata(deployRevision.Metadata, "trigger_source") != "deploy_api" {
		t.Fatalf("expected deploy_api trigger source, got %#v", deployRevision.Metadata["trigger_source"])
	}
	if stringMetadata(deployRevision.Metadata, "deploy_version") != deployResult.Version {
		t.Fatalf("expected deploy version metadata %q, got %#v", deployResult.Version, deployRevision.Metadata["deploy_version"])
	}
	assertRevisionAuditCount(t, stores, "config.save_draft", console.AuditOutcomeSuccess, 1)
	assertRevisionAuditCount(t, stores, "config.validate_revision", console.AuditOutcomeSuccess, 1)
	assertRevisionAuditCount(t, stores, "config.activate_revision", console.AuditOutcomeSuccess, 1)

	if _, rollbackErr := service.Rollback(deployResult.Version); rollbackErr != nil {
		t.Fatalf("Rollback() error = %v", rollbackErr)
	}
	assertRollbackPersistence(t, ctx, stores, deployRevision.ID)
}

func TestDeployPreviewPrefersPersistedActiveRevision(t *testing.T) {
	tempDir := t.TempDir()
	configPath := writePersistenceTestConfig(t, tempDir)
	stores := openPersistenceTestStores(t, tempDir)
	service := NewWithStores(configPath, tempDir, stores)

	saveActiveRevisionForReadModel(t, stores, []byte(persistedActiveConfig))

	preview, err := service.DeployPreview(DeployRequest{
		YAML: `
decisions:
  - name: staged-decision
    description: staged
    priority: 200
`,
	})
	if err != nil {
		t.Fatalf("DeployPreview() error = %v", err)
	}
	if !strings.Contains(preview.Current, "persisted-model") {
		t.Fatalf("expected preview to use persisted active revision, got:\n%s", preview.Current)
	}
	if strings.Contains(preview.Current, "bert_model:") {
		t.Fatalf("expected preview current config to avoid legacy file-backed config, got:\n%s", preview.Current)
	}
}

func writePersistenceTestConfig(t *testing.T, dir string) string {
	t.Helper()

	configPath := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(configPath, []byte(persistenceTestConfig), 0o644); err != nil {
		t.Fatalf("failed to write test config: %v", err)
	}
	return configPath
}

func openPersistenceTestStores(t *testing.T, dir string) *console.Stores {
	t.Helper()

	store, err := console.NewSQLiteStore(filepath.Join(dir, "console.db"))
	if err != nil {
		t.Fatalf("NewSQLiteStore() error = %v", err)
	}
	t.Cleanup(func() {
		_ = store.Close()
	})
	return console.NewStores(store)
}

func assertRollbackPersistence(t *testing.T, ctx context.Context, stores *console.Stores, deployRevisionID string) {
	t.Helper()

	revisions, err := stores.Revisions.ListConfigRevisions(ctx, console.ConfigRevisionFilter{Limit: 10})
	if err != nil {
		t.Fatalf("ListConfigRevisions() after rollback error = %v", err)
	}
	if len(revisions) != 2 {
		t.Fatalf("expected 2 config revisions after rollback, got %d", len(revisions))
	}

	activeRevision, rolledBackRevision := findRollbackRevisions(revisions, deployRevisionID)
	if activeRevision == nil {
		t.Fatal("expected an active revision after rollback")
	}
	if activeRevision.Source != "compat_config_rollback" {
		t.Fatalf("expected rollback revision source, got %q", activeRevision.Source)
	}
	if activeRevision.CreatedBy != compatibilityActorID {
		t.Fatalf("expected rollback revision actor %q, got %q", compatibilityActorID, activeRevision.CreatedBy)
	}
	if stringMetadata(activeRevision.Metadata, "trigger_source") != "rollback_api" {
		t.Fatalf("expected rollback_api trigger source, got %#v", activeRevision.Metadata["trigger_source"])
	}
	if rolledBackRevision == nil {
		t.Fatal("expected to find the original deploy revision")
	}
	if rolledBackRevision.Status != console.ConfigRevisionStatusRolledBack {
		t.Fatalf("expected original deploy revision to be rolled_back, got %q", rolledBackRevision.Status)
	}

	rollbackEvents, err := stores.Deployments.ListDeployEvents(ctx, console.DeployEventFilter{
		RevisionID: activeRevision.ID,
		Limit:      10,
	})
	if err != nil {
		t.Fatalf("ListDeployEvents() for rollback revision error = %v", err)
	}
	if len(rollbackEvents) != 1 {
		t.Fatalf("expected 1 rollback deploy event, got %d", len(rollbackEvents))
	}
	if rollbackEvents[0].Status != console.DeployEventStatusRolledBack {
		t.Fatalf("expected rollback deploy event status rolled_back, got %q", rollbackEvents[0].Status)
	}
	if rollbackEvents[0].RollbackRevisionID != deployRevisionID {
		t.Fatalf("expected rollback deploy event to point at %q, got %q", deployRevisionID, rollbackEvents[0].RollbackRevisionID)
	}
	if rollbackEvents[0].TriggerSource != "rollback_api" {
		t.Fatalf("expected rollback trigger source rollback_api, got %q", rollbackEvents[0].TriggerSource)
	}

	auditEvents, err := stores.Audit.ListAuditEvents(ctx, console.AuditEventFilter{
		Action: "config.rollback",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListAuditEvents() error = %v", err)
	}
	if len(auditEvents) != 1 {
		t.Fatalf("expected 1 rollback audit event, got %d", len(auditEvents))
	}
	assertRevisionAuditCount(t, stores, "config.save_draft", console.AuditOutcomeSuccess, 2)
	assertRevisionAuditCount(t, stores, "config.validate_revision", console.AuditOutcomeSuccess, 2)
	assertRevisionAuditCount(t, stores, "config.activate_revision", console.AuditOutcomeSuccess, 2)
}

func findRollbackRevisions(revisions []console.ConfigRevision, deployRevisionID string) (*console.ConfigRevision, *console.ConfigRevision) {
	var activeRevision *console.ConfigRevision
	var rolledBackRevision *console.ConfigRevision

	for i := range revisions {
		revision := &revisions[i]
		switch {
		case revision.Status == console.ConfigRevisionStatusActive:
			activeRevision = revision
		case revision.ID == deployRevisionID:
			rolledBackRevision = revision
		}
	}
	return activeRevision, rolledBackRevision
}
