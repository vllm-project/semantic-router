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

// RevisionActivationResult describes the activated revision plus rollout feedback.
type RevisionActivationResult struct {
	RevisionDetail
	Message string
	Version string
}

func (s *Service) ActivateRevision(revisionID string) (*RevisionActivationResult, error) {
	return s.activateRevision(revisionID, activationOptions{
		mutation:       revisionAPIMutationContext(),
		successMessage: fmt.Sprintf("Activated config revision %s. Router and Envoy have been updated.", revisionID),
	})
}

func (s *Service) ActivateRevisionAs(revisionID string, actorID string) (*RevisionActivationResult, error) {
	return s.activateRevision(revisionID, activationOptions{
		mutation:       revisionAPIMutationContextForActor(actorID),
		successMessage: fmt.Sprintf("Activated config revision %s. Router and Envoy have been updated.", revisionID),
	})
}

type activationOptions struct {
	mutation             revisionMutationContext
	successMessage       string
	previousActiveStatus console.ConfigRevisionStatus
	deployStatus         console.DeployEventStatus
	auditAction          string
}

func (s *Service) activateRevision(
	revisionID string,
	options activationOptions,
) (*RevisionActivationResult, error) {
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

	message := options.successMessage
	if message == "" {
		message = fmt.Sprintf("Activated config revision %s. Router and Envoy have been updated.", revisionID)
	}
	if persistErr := s.persistRevisionActivation(ctx, revision, version, message, options); persistErr != nil {
		return nil, persistErr
	}

	detail, detailErr := s.revisionDetail(ctx, *revision)
	if detailErr != nil {
		return nil, detailErr
	}
	return &RevisionActivationResult{
		RevisionDetail: *detail,
		Message:        message,
		Version:        version,
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
	options activationOptions,
) error {
	previousRevision, err := s.latestActiveRevision(ctx)
	if err != nil {
		return err
	}

	now := time.Now().UTC()
	metadata := s.revisionMutationMetadata(revision.Metadata, nil, "activate_revision", options.mutation)
	metadata["deploy_version"] = version
	if previousRevision != nil && previousRevision.ID != revision.ID {
		metadata["previous_active_revision_id"] = previousRevision.ID
	}

	revision.Status = console.ConfigRevisionStatusActive
	revision.ActivatedAt = &now
	revision.Metadata = metadata
	if err := s.Stores.Revisions.SaveConfigRevision(ctx, revision); err != nil {
		return fmt.Errorf("failed to persist activated config revision %s: %w", revision.ID, err)
	}

	previousStatus := options.previousActiveStatus
	if previousStatus == "" {
		previousStatus = console.ConfigRevisionStatusSuperseded
	}
	if err := s.updatePreviousActiveRevisionStatus(ctx, previousRevision, previousStatus); err != nil {
		return err
	}

	deployStatus := options.deployStatus
	if deployStatus == "" {
		deployStatus = console.DeployEventStatusSucceeded
	}
	if err := s.saveActivationDeployEvent(ctx, revision, previousRevision, metadata, message, previousStatus, deployStatus, options.mutation.triggerSource, now); err != nil {
		return err
	}

	auditAction := options.auditAction
	if auditAction == "" {
		auditAction = "config.activate_revision"
	}
	if err := s.appendActivationAuditEvent(ctx, revision.ID, options.mutation.actorID, auditAction, message, metadata, now); err != nil {
		return err
	}

	return nil
}

func (s *Service) updatePreviousActiveRevisionStatus(
	ctx context.Context,
	previousRevision *console.ConfigRevision,
	previousStatus console.ConfigRevisionStatus,
) error {
	if previousRevision == nil || previousRevision.ID == "" || previousRevision.Status == previousStatus {
		return nil
	}
	previousRevision.Status = previousStatus
	if err := s.Stores.Revisions.SaveConfigRevision(ctx, previousRevision); err != nil {
		return fmt.Errorf("failed to update previous config revision %s status: %w", previousRevision.ID, err)
	}
	return nil
}

func (s *Service) saveActivationDeployEvent(
	ctx context.Context,
	revision *console.ConfigRevision,
	previousRevision *console.ConfigRevision,
	metadata map[string]interface{},
	message string,
	previousStatus console.ConfigRevisionStatus,
	deployStatus console.DeployEventStatus,
	triggerSource string,
	now time.Time,
) error {
	if s.Stores.Deployments == nil {
		return nil
	}
	event := &console.DeployEvent{
		RevisionID:    revision.ID,
		Status:        deployStatus,
		TriggerSource: triggerSource,
		Message:       message,
		RuntimeTarget: s.runtimeTarget(),
		Metadata:      metadata,
		StartedAt:     &now,
		CompletedAt:   &now,
	}
	if previousRevision != nil && previousRevision.ID != revision.ID && previousStatus == console.ConfigRevisionStatusRolledBack {
		event.RollbackRevisionID = previousRevision.ID
	}
	if err := s.Stores.Deployments.SaveDeployEvent(ctx, event); err != nil {
		return fmt.Errorf("failed to save config revision deploy event: %w", err)
	}
	return nil
}

func (s *Service) appendActivationAuditEvent(
	ctx context.Context,
	revisionID string,
	actorID string,
	auditAction string,
	message string,
	metadata map[string]interface{},
	now time.Time,
) error {
	if s.Stores.Audit == nil {
		return nil
	}
	event := &console.AuditEvent{
		ActorType:  console.PrincipalTypeServiceAccount,
		ActorID:    actorID,
		Action:     auditAction,
		TargetType: "config_revision",
		TargetID:   revisionID,
		Outcome:    console.AuditOutcomeSuccess,
		Message:    message,
		Metadata:   metadata,
		OccurredAt: now,
	}
	if err := s.Stores.Audit.AppendAuditEvent(ctx, event); err != nil {
		return fmt.Errorf("failed to append config revision audit event: %w", err)
	}
	return nil
}
