package configlifecycle

import (
	"context"

	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

type compatibilityWorkflowOptions struct {
	source         string
	summary        string
	triggerSource  string
	auditAction    string
	successMessage string
	metadata       map[string]interface{}
	activation     activationOptions
}

func (s *Service) hasRevisionStore() bool {
	return s != nil && s.Stores != nil && s.Stores.Revisions != nil
}

func (s *Service) runCompatibilityRevisionWorkflow(
	yamlData []byte,
	options compatibilityWorkflowOptions,
) (*RevisionActivationResult, error) {
	mutation := compatibilityMutationContext(options.triggerSource)
	draftResult, err := s.saveDraftRevision(RevisionDraftInput{
		Source:            options.source,
		Summary:           options.summary,
		RuntimeConfigYAML: string(yamlData),
		Metadata:          options.metadata,
	}, mutation)
	if err != nil {
		return nil, err
	}

	if _, validateErr := s.validateRevision(draftResult.ID, mutation); validateErr != nil {
		return nil, validateErr
	}

	activationResult, err := s.activateRevision(draftResult.ID, activationOptions{
		mutation:             mutation,
		successMessage:       options.successMessage,
		previousActiveStatus: options.activation.previousActiveStatus,
		deployStatus:         options.activation.deployStatus,
		auditAction:          options.activation.auditAction,
	})
	if err != nil {
		return nil, err
	}

	if options.auditAction == "" {
		return activationResult, nil
	}
	if err := s.appendRevisionAudit(
		context.Background(),
		mutation.actorID,
		options.auditAction,
		activationResult.ID,
		console.AuditOutcomeSuccess,
		activationResult.Message,
		activationResult.Metadata,
	); err != nil {
		return nil, err
	}
	return activationResult, nil
}
