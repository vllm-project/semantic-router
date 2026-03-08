package configlifecycle

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

const consoleServiceActorID = "dashboard-console"

// RevisionActivationResult describes the activated revision plus rollout feedback.
type RevisionActivationResult struct {
	RevisionDetail
	Message string
}

func (s *Service) ActivateRevision(revisionID string) (*RevisionActivationResult, error) {
	if err := ensureRevisionActivationReady(s, revisionID); err != nil {
		return nil, err
	}
	if !deployMu.TryLock() {
		return nil, &Error{
			StatusCode: http.StatusConflict,
			Code:       "deploy_in_progress",
			Message:    "Another config operation is in progress. Please try again.",
		}
	}
	defer deployMu.Unlock()

	ctx := context.Background()
	revision, yamlData, err := s.loadRevisionForActivation(ctx, revisionID)
	if err != nil {
		return nil, err
	}
	version, err := s.applyActivatedConfig(yamlData)
	if err != nil {
		return nil, err
	}

	message := fmt.Sprintf("Activated config revision %s. Router and Envoy have been updated.", revisionID)
	if persistErr := s.persistRevisionActivation(ctx, revision, version, message); persistErr != nil {
		return nil, persistErr
	}

	detail, detailErr := s.revisionDetail(ctx, *revision)
	if detailErr != nil {
		return nil, detailErr
	}
	return &RevisionActivationResult{
		RevisionDetail: *detail,
		Message:        message,
	}, nil
}

func ensureRevisionActivationReady(s *Service, revisionID string) error {
	if s == nil || s.Stores == nil || s.Stores.Revisions == nil {
		return &Error{
			StatusCode: http.StatusServiceUnavailable,
			Code:       "console_store_unavailable",
			Message:    "Console revision store is not configured.",
		}
	}
	if revisionID == "" {
		return &Error{
			StatusCode: http.StatusBadRequest,
			Code:       "revision_id_required",
			Message:    "revision id is required",
		}
	}
	return nil
}

func (s *Service) loadRevisionForActivation(ctx context.Context, revisionID string) (*console.ConfigRevision, []byte, error) {
	revision, err := s.Stores.Revisions.GetConfigRevision(ctx, revisionID)
	if err != nil {
		return nil, nil, err
	}
	if revision == nil {
		return nil, nil, &Error{
			StatusCode: http.StatusNotFound,
			Code:       "revision_not_found",
			Message:    fmt.Sprintf("Config revision %s not found", revisionID),
		}
	}

	yamlData, err := revisionYAMLBytes(*revision)
	if err != nil {
		return nil, nil, &Error{
			StatusCode: http.StatusBadRequest,
			Code:       "revision_invalid",
			Message:    fmt.Sprintf("Config revision %s is not deployable: %v", revisionID, err),
		}
	}
	if validationErr := validateConfigYAML(yamlData); validationErr != nil {
		return nil, nil, &Error{
			StatusCode: http.StatusBadRequest,
			Code:       "revision_invalid",
			Message:    fmt.Sprintf("Config revision %s failed validation: %v", revisionID, validationErr),
		}
	}
	return revision, yamlData, nil
}

func (s *Service) applyActivatedConfig(yamlData []byte) (string, error) {
	existingData, readErr := os.ReadFile(s.ConfigPath)
	if readErr != nil && !os.IsNotExist(readErr) {
		return "", fmt.Errorf("failed to read existing config: %w", readErr)
	}

	version := time.Now().Format("20060102-150405")
	if backupErr := s.backupConfigData(existingData, version); backupErr != nil {
		log.Printf("Warning: failed to create backup before revision activation: %v", backupErr)
	}
	if writeErr := writeConfigAtomically(s.ConfigPath, yamlData); writeErr != nil {
		return "", fmt.Errorf("failed to write config: %w", writeErr)
	}
	if propagateErr := s.propagateConfigToRuntime(); propagateErr != nil {
		if restoreErr := s.restorePreviousRuntimeConfig(existingData); restoreErr != nil {
			return "", fmt.Errorf("failed to activate config revision: %w; failed to restore previous config: %w", propagateErr, restoreErr)
		}
		return "", fmt.Errorf("failed to activate config revision: %w; previous config restored", propagateErr)
	}
	return version, nil
}

func (s *Service) persistRevisionActivation(
	ctx context.Context,
	revision *console.ConfigRevision,
	version string,
	message string,
) error {
	previousRevision, err := s.latestActiveRevision(ctx)
	if err != nil {
		return err
	}

	now := time.Now().UTC()
	metadata := s.baseMetadata(revision.Metadata)
	metadata["operation"] = "activate_revision"
	metadata["deploy_version"] = version
	metadata["trigger_source"] = "revision_api"
	if previousRevision != nil && previousRevision.ID != revision.ID {
		metadata["previous_active_revision_id"] = previousRevision.ID
	}

	revision.Status = console.ConfigRevisionStatusActive
	revision.ActivatedAt = &now
	revision.Metadata = metadata
	if err := s.Stores.Revisions.SaveConfigRevision(ctx, revision); err != nil {
		return fmt.Errorf("failed to persist activated config revision %s: %w", revision.ID, err)
	}

	if previousRevision != nil && previousRevision.ID != revision.ID {
		previousRevision.Status = console.ConfigRevisionStatusSuperseded
		if err := s.Stores.Revisions.SaveConfigRevision(ctx, previousRevision); err != nil {
			return fmt.Errorf("failed to mark previous config revision %s as superseded: %w", previousRevision.ID, err)
		}
	}

	if s.Stores.Deployments != nil {
		event := &console.DeployEvent{
			RevisionID:    revision.ID,
			Status:        console.DeployEventStatusSucceeded,
			TriggerSource: "revision_api",
			Message:       message,
			RuntimeTarget: s.runtimeTarget(),
			Metadata:      metadata,
			StartedAt:     &now,
			CompletedAt:   &now,
		}
		if err := s.Stores.Deployments.SaveDeployEvent(ctx, event); err != nil {
			return fmt.Errorf("failed to save config revision deploy event: %w", err)
		}
	}

	if s.Stores.Audit != nil {
		event := &console.AuditEvent{
			ActorType:  console.PrincipalTypeServiceAccount,
			ActorID:    consoleServiceActorID,
			Action:     "config.activate_revision",
			TargetType: "config_revision",
			TargetID:   revision.ID,
			Outcome:    console.AuditOutcomeSuccess,
			Message:    message,
			Metadata:   metadata,
			OccurredAt: now,
		}
		if err := s.Stores.Audit.AppendAuditEvent(ctx, event); err != nil {
			return fmt.Errorf("failed to append config revision audit event: %w", err)
		}
	}

	return nil
}
