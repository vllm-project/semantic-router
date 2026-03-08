package configlifecycle

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

func TestActivateRevisionPromotesPersistedRevision(t *testing.T) {
	configPath, currentRevisionID, targetRevisionID, service, stores := newActivationFixture(t)

	result, err := service.ActivateRevision(targetRevisionID)
	if err != nil {
		t.Fatalf("ActivateRevision() error = %v", err)
	}
	assertActivationResult(t, result, targetRevisionID)
	assertActivatedCurrentState(t, service, targetRevisionID)
	assertActivatedPersistence(t, stores, currentRevisionID, targetRevisionID)
	assertActivatedConfigWritten(t, configPath)
}

func newActivationFixture(t *testing.T) (string, string, string, *Service, *console.Stores) {
	t.Helper()

	tempDir := t.TempDir()
	configPath := writePersistenceTestConfig(t, tempDir)
	stores := openPersistenceTestStores(t, tempDir)
	service := NewWithStores(configPath, tempDir, stores)

	now := time.Now().UTC()
	currentRevisionID := saveRevisionFixture(t, stores, console.ConfigRevisionStatusActive, "revision_current", persistenceTestConfig, now.Add(-1*time.Minute))
	saveDeployFixture(t, stores, currentRevisionID, console.DeployEventStatusSucceeded, "workspace-files")

	targetConfig := strings.Replace(persistenceTestConfig, "threshold: 0.6", "threshold: 0.75", 1)
	targetRevisionID := saveRevisionFixture(t, stores, console.ConfigRevisionStatusValidated, "revision_validated", targetConfig, now.Add(-2*time.Minute))

	return configPath, currentRevisionID, targetRevisionID, service, stores
}

func assertActivationResult(t *testing.T, result *RevisionActivationResult, targetRevisionID string) {
	t.Helper()

	if result == nil {
		t.Fatal("expected activation result")
	}
	if result.ID != targetRevisionID {
		t.Fatalf("expected activated revision %q, got %q", targetRevisionID, result.ID)
	}
	if result.Status != console.ConfigRevisionStatusActive {
		t.Fatalf("expected activated revision status active, got %q", result.Status)
	}
}

func assertActivatedCurrentState(t *testing.T, service *Service, targetRevisionID string) {
	t.Helper()

	currentRevision, err := service.CurrentRevision()
	if err != nil {
		t.Fatalf("CurrentRevision() error = %v", err)
	}
	if currentRevision == nil || currentRevision.ID != targetRevisionID {
		t.Fatalf("expected current revision %q after activation, got %#v", targetRevisionID, currentRevision)
	}

	revisions, err := service.ListRevisions(10)
	if err != nil {
		t.Fatalf("ListRevisions() error = %v", err)
	}
	if len(revisions) != 2 {
		t.Fatalf("expected 2 revisions, got %d", len(revisions))
	}
	if revisions[0].ID != targetRevisionID {
		t.Fatalf("expected activated revision to be listed first, got %q", revisions[0].ID)
	}
}

func assertActivatedPersistence(t *testing.T, stores *console.Stores, currentRevisionID, targetRevisionID string) {
	t.Helper()

	ctx := context.Background()
	targetRevision, err := stores.Revisions.GetConfigRevision(ctx, targetRevisionID)
	if err != nil {
		t.Fatalf("GetConfigRevision(target) error = %v", err)
	}
	if targetRevision == nil || targetRevision.Status != console.ConfigRevisionStatusActive {
		t.Fatalf("expected target revision to be active, got %#v", targetRevision)
	}
	if targetRevision.ActivatedAt == nil {
		t.Fatal("expected target revision activated_at to be populated")
	}

	previousRevision, err := stores.Revisions.GetConfigRevision(ctx, currentRevisionID)
	if err != nil {
		t.Fatalf("GetConfigRevision(previous) error = %v", err)
	}
	if previousRevision == nil || previousRevision.Status != console.ConfigRevisionStatusSuperseded {
		t.Fatalf("expected previous revision to be superseded, got %#v", previousRevision)
	}

	assertActivationEvents(t, ctx, stores, targetRevisionID)
}

func assertActivationEvents(t *testing.T, ctx context.Context, stores *console.Stores, targetRevisionID string) {
	t.Helper()

	deployEvents, err := stores.Deployments.ListDeployEvents(ctx, console.DeployEventFilter{
		RevisionID: targetRevisionID,
		Limit:      10,
	})
	if err != nil {
		t.Fatalf("ListDeployEvents() error = %v", err)
	}
	if len(deployEvents) != 1 {
		t.Fatalf("expected 1 deploy event for activated revision, got %d", len(deployEvents))
	}
	if deployEvents[0].TriggerSource != "revision_api" {
		t.Fatalf("expected revision_api trigger source, got %q", deployEvents[0].TriggerSource)
	}
	if deployEvents[0].Status != console.DeployEventStatusSucceeded {
		t.Fatalf("expected succeeded deploy event, got %q", deployEvents[0].Status)
	}

	auditEvents, err := stores.Audit.ListAuditEvents(ctx, console.AuditEventFilter{
		Action: "config.activate_revision",
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListAuditEvents() error = %v", err)
	}
	if len(auditEvents) != 1 {
		t.Fatalf("expected 1 activation audit event, got %d", len(auditEvents))
	}
	if auditEvents[0].ActorID != consoleServiceActorID {
		t.Fatalf("expected audit actor %q, got %q", consoleServiceActorID, auditEvents[0].ActorID)
	}
}

func assertActivatedConfigWritten(t *testing.T, configPath string) {
	t.Helper()

	configData, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("ReadFile(configPath) error = %v", err)
	}
	if !strings.Contains(string(configData), "threshold: 0.75") {
		t.Fatalf("expected activated config to be written to disk, got:\n%s", string(configData))
	}
}
