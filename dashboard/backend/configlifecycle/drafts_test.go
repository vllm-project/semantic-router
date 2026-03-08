package configlifecycle

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

func TestSaveDraftRevisionCreatesParentedDraft(t *testing.T) {
	service, stores, activeRevisionID := newDraftServiceFixture(t)

	result, err := service.SaveDraftRevision(RevisionDraftInput{
		Summary:           "draft from test",
		RuntimeConfigYAML: persistenceTestConfig,
	})
	if err != nil {
		t.Fatalf("SaveDraftRevision() error = %v", err)
	}
	if result == nil || result.ID == "" {
		t.Fatalf("expected saved draft result with ID, got %#v", result)
	}
	if result.Status != console.ConfigRevisionStatusDraft {
		t.Fatalf("expected draft status, got %q", result.Status)
	}
	if result.ParentRevisionID != activeRevisionID {
		t.Fatalf("expected parent revision %q, got %q", activeRevisionID, result.ParentRevisionID)
	}

	assertRevisionAuditCount(t, stores, "config.save_draft", console.AuditOutcomeSuccess, 1)
}

func TestValidateRevisionPromotesDraftToValidated(t *testing.T) {
	service, stores, _ := newDraftServiceFixture(t)
	draft := saveDraftForValidation(t, service, persistenceTestConfig)

	result, err := service.ValidateRevision(draft.ID)
	if err != nil {
		t.Fatalf("ValidateRevision() error = %v", err)
	}
	if result == nil || result.Status != console.ConfigRevisionStatusValidated {
		t.Fatalf("expected validated result, got %#v", result)
	}

	revision, err := stores.Revisions.GetConfigRevision(context.Background(), draft.ID)
	if err != nil {
		t.Fatalf("GetConfigRevision() error = %v", err)
	}
	if revision == nil || revision.Status != console.ConfigRevisionStatusValidated {
		t.Fatalf("expected validated revision in store, got %#v", revision)
	}
	assertRevisionAuditCount(t, stores, "config.validate_revision", console.AuditOutcomeSuccess, 1)
}

func TestValidateRevisionFailureAppendsAuditFailure(t *testing.T) {
	service, stores, _ := newDraftServiceFixture(t)
	invalidDraft := saveDraftForValidation(t, service, invalidValidationConfig())

	_, err := service.ValidateRevision(invalidDraft.ID)
	assertRevisionValidationError(t, err)

	revision, loadErr := stores.Revisions.GetConfigRevision(context.Background(), invalidDraft.ID)
	if loadErr != nil {
		t.Fatalf("GetConfigRevision() error = %v", loadErr)
	}
	if revision == nil || revision.Status != console.ConfigRevisionStatusDraft {
		t.Fatalf("expected failed validation to keep draft status, got %#v", revision)
	}
	assertRevisionAuditCount(t, stores, "config.validate_revision", console.AuditOutcomeFailure, 1)
}

func newDraftServiceFixture(t *testing.T) (*Service, *console.Stores, string) {
	t.Helper()

	tempDir := t.TempDir()
	configPath := writePersistenceTestConfig(t, tempDir)
	stores := openPersistenceTestStores(t, tempDir)
	service := NewWithStores(configPath, tempDir, stores)

	activeRevisionID := saveRevisionFixture(
		t,
		stores,
		console.ConfigRevisionStatusActive,
		"revision_active",
		persistedActiveConfig,
		time.Now().UTC().Add(-1*time.Minute),
	)
	return service, stores, activeRevisionID
}

func saveDraftForValidation(t *testing.T, service *Service, runtimeConfigYAML string) *RevisionSaveResult {
	t.Helper()

	result, err := service.SaveDraftRevision(RevisionDraftInput{
		Summary:           "draft to validate",
		RuntimeConfigYAML: runtimeConfigYAML,
	})
	if err != nil {
		t.Fatalf("SaveDraftRevision() error = %v", err)
	}
	return result
}

func invalidValidationConfig() string {
	return strings.Replace(persistenceTestConfig, "address: 127.0.0.1", "address: http://bad-endpoint", 1)
}

func assertRevisionValidationError(t *testing.T, err error) {
	t.Helper()

	var lifecycleErr *Error
	if !errors.As(err, &lifecycleErr) {
		t.Fatalf("expected lifecycle error, got %T", err)
	}
	if lifecycleErr.StatusCode != 400 || lifecycleErr.Code != "revision_invalid" {
		t.Fatalf("expected 400 revision_invalid, got %#v", lifecycleErr)
	}
}

func assertRevisionAuditCount(
	t *testing.T,
	stores *console.Stores,
	action string,
	outcome console.AuditOutcome,
	expected int,
) {
	t.Helper()

	events, err := stores.Audit.ListAuditEvents(context.Background(), console.AuditEventFilter{
		Action: action,
		Limit:  10,
	})
	if err != nil {
		t.Fatalf("ListAuditEvents() error = %v", err)
	}

	count := 0
	for _, event := range events {
		if event.Outcome == outcome {
			count++
		}
	}
	if count != expected {
		t.Fatalf("expected %d %s audit events for %s, got %d", expected, outcome, action, count)
	}
}
