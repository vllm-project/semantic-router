package configlifecycle

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

func (s *Service) ValidateRevision(revisionID string) (*RevisionValidationResult, error) {
	return s.validateRevision(revisionID, revisionAPIMutationContext())
}

func (s *Service) ValidateRevisionAs(revisionID string, actorID string) (*RevisionValidationResult, error) {
	return s.validateRevision(revisionID, revisionAPIMutationContextForActor(actorID))
}

func (s *Service) validateRevision(
	revisionID string,
	mutation revisionMutationContext,
) (*RevisionValidationResult, error) {
	if err := ensureRevisionID(ensureRevisionStoreAvailable(s), revisionID); err != nil {
		return nil, err
	}

	ctx := context.Background()
	revision, err := s.loadValidatableRevision(ctx, revisionID)
	if err != nil {
		return nil, err
	}

	yamlData, err := revisionYAMLBytes(*revision)
	if err != nil {
		return nil, s.failRevisionValidation(ctx, revision.ID, revision.Metadata, mutation, fmt.Sprintf("Config revision %s is not deployable: %v", revisionID, err))
	}
	if validationErr := validateConfigYAML(yamlData); validationErr != nil {
		return nil, s.failRevisionValidation(ctx, revision.ID, revision.Metadata, mutation, fmt.Sprintf("Config revision %s failed validation: %v", revisionID, validationErr))
	}

	message := fmt.Sprintf("Validated config revision %s. Ready for activation.", revisionID)
	if err := s.persistValidatedRevision(ctx, revision, yamlData, mutation, message); err != nil {
		return nil, err
	}
	return s.revisionValidationResult(ctx, *revision, message)
}

func (s *Service) loadValidatableRevision(ctx context.Context, revisionID string) (*console.ConfigRevision, error) {
	revision, err := s.Stores.Revisions.GetConfigRevision(ctx, revisionID)
	if err != nil {
		return nil, err
	}
	if revision == nil {
		return nil, &Error{
			StatusCode: 404,
			Code:       "revision_not_found",
			Message:    fmt.Sprintf("Config revision %s not found", revisionID),
		}
	}
	if !isDraftMutableStatus(revision.Status) {
		return nil, &Error{
			StatusCode: 409,
			Code:       "revision_not_mutable",
			Message:    fmt.Sprintf("Config revision %s cannot be validated from its current status", revisionID),
		}
	}
	return revision, nil
}

func (s *Service) failRevisionValidation(
	ctx context.Context,
	revisionID string,
	metadata map[string]interface{},
	mutation revisionMutationContext,
	message string,
) error {
	merged := s.revisionMutationMetadata(metadata, nil, "validate_revision", mutation)
	if auditErr := s.appendRevisionAudit(ctx, mutation.actorID, "config.validate_revision", revisionID, console.AuditOutcomeFailure, message, merged); auditErr != nil {
		return auditErr
	}
	return &Error{
		StatusCode: 400,
		Code:       "revision_invalid",
		Message:    message,
	}
}

func (s *Service) persistValidatedRevision(
	ctx context.Context,
	revision *console.ConfigRevision,
	yamlData []byte,
	mutation revisionMutationContext,
	message string,
) error {
	metadata := s.revisionMutationMetadata(revision.Metadata, nil, "validate_revision", mutation)
	metadata["validated_at"] = time.Now().UTC().Format(time.RFC3339)

	revision.Status = console.ConfigRevisionStatusValidated
	revision.RuntimeConfigYAML = string(yamlData)
	revision.Metadata = metadata
	if err := s.Stores.Revisions.SaveConfigRevision(ctx, revision); err != nil {
		return fmt.Errorf("failed to save validated config revision %s: %w", revision.ID, err)
	}
	if err := s.appendRevisionAudit(ctx, mutation.actorID, "config.validate_revision", revision.ID, console.AuditOutcomeSuccess, message, metadata); err != nil {
		return err
	}
	return nil
}

func (s *Service) revisionValidationResult(ctx context.Context, revision console.ConfigRevision, message string) (*RevisionValidationResult, error) {
	detail, err := s.revisionDetail(ctx, revision)
	if err != nil {
		return nil, err
	}
	return &RevisionValidationResult{RevisionDetail: *detail, Message: message}, nil
}
