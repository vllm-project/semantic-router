package configlifecycle

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

type revisionDraftPayload struct {
	documentJSON      json.RawMessage
	runtimeConfigYAML string
}

func (s *Service) SaveDraftRevision(input RevisionDraftInput) (*RevisionSaveResult, error) {
	if err := ensureRevisionStoreAvailable(s); err != nil {
		return nil, err
	}

	ctx := context.Background()
	existing, err := s.loadMutableDraftRevision(ctx, input.ID)
	if err != nil {
		return nil, err
	}

	payload, err := buildRevisionDraftPayload(input)
	if err != nil {
		return nil, err
	}

	revision, err := s.buildDraftRevision(ctx, input, existing, payload)
	if err != nil {
		return nil, err
	}
	if err := s.Stores.Revisions.SaveConfigRevision(ctx, revision); err != nil {
		return nil, fmt.Errorf("failed to save config draft revision: %w", err)
	}

	message := fmt.Sprintf("Saved config draft revision %s.", revision.ID)
	if err := s.appendRevisionAudit(ctx, "config.save_draft", revision.ID, console.AuditOutcomeSuccess, message, revision.Metadata); err != nil {
		return nil, err
	}
	return s.revisionSaveResult(ctx, *revision, message)
}

func buildRevisionDraftPayload(input RevisionDraftInput) (revisionDraftPayload, error) {
	switch {
	case input.Document != nil:
		return draftPayloadFromDocument(input.Document)
	case strings.TrimSpace(input.RuntimeConfigYAML) != "":
		return draftPayloadFromYAML(input.RuntimeConfigYAML)
	default:
		return revisionDraftPayload{}, &Error{
			StatusCode: 400,
			Code:       "revision_document_required",
			Message:    "draft revision requires either document or runtimeConfigYAML",
		}
	}
}

func draftPayloadFromDocument(document interface{}) (revisionDraftPayload, error) {
	normalized := normalizeYAMLValue(document)
	documentJSON, err := json.Marshal(normalized)
	if err != nil {
		return revisionDraftPayload{}, err
	}
	yamlData, err := yaml.Marshal(normalized)
	if err != nil {
		return revisionDraftPayload{}, err
	}
	return revisionDraftPayload{
		documentJSON:      documentJSON,
		runtimeConfigYAML: string(yamlData),
	}, nil
}

func draftPayloadFromYAML(rawYAML string) (revisionDraftPayload, error) {
	yamlData, err := parseDeployYAML(rawYAML)
	if err != nil {
		return revisionDraftPayload{}, err
	}

	documentJSON, err := marshalRevisionDocument(yamlData)
	if err != nil {
		return revisionDraftPayload{}, err
	}
	return revisionDraftPayload{
		documentJSON:      documentJSON,
		runtimeConfigYAML: CanonicalizeYAMLForDiff(yamlData),
	}, nil
}

func (s *Service) loadMutableDraftRevision(ctx context.Context, revisionID string) (*console.ConfigRevision, error) {
	if revisionID == "" {
		return nil, nil
	}

	revision, err := s.Stores.Revisions.GetConfigRevision(ctx, revisionID)
	if err != nil {
		return nil, err
	}
	if revision == nil {
		return nil, nil
	}
	if !isDraftMutableStatus(revision.Status) {
		return nil, &Error{
			StatusCode: 409,
			Code:       "revision_not_mutable",
			Message:    fmt.Sprintf("Config revision %s is not editable from the draft API", revisionID),
		}
	}
	return revision, nil
}

func isDraftMutableStatus(status console.ConfigRevisionStatus) bool {
	switch status {
	case "", console.ConfigRevisionStatusDraft, console.ConfigRevisionStatusValidated:
		return true
	default:
		return false
	}
}

func (s *Service) buildDraftRevision(
	ctx context.Context,
	input RevisionDraftInput,
	existing *console.ConfigRevision,
	payload revisionDraftPayload,
) (*console.ConfigRevision, error) {
	parentRevisionID, err := s.resolveDraftParentRevisionID(ctx, input.ParentRevisionID, existing)
	if err != nil {
		return nil, err
	}

	revision := &console.ConfigRevision{
		ID:                input.ID,
		ParentRevisionID:  parentRevisionID,
		Status:            console.ConfigRevisionStatusDraft,
		Source:            coalesceString(input.Source, existingSource(existing), "revision_api_draft"),
		Summary:           coalesceString(input.Summary, existingSummary(existing)),
		DocumentJSON:      payload.documentJSON,
		RuntimeConfigYAML: payload.runtimeConfigYAML,
		CreatedBy:         coalesceString(existingCreatedBy(existing), consoleServiceActorID),
		Metadata:          s.baseMetadata(mergeMetadataMaps(existingMetadata(existing), input.Metadata)),
		CreatedAt:         existingCreatedAt(existing),
	}
	revision.Metadata["operation"] = "save_draft"
	revision.Metadata["trigger_source"] = revisionAPITriggerSource
	return revision, nil
}

func (s *Service) resolveDraftParentRevisionID(
	ctx context.Context,
	requestedParentID string,
	existing *console.ConfigRevision,
) (string, error) {
	if requestedParentID != "" {
		return requestedParentID, nil
	}
	if existing != nil && existing.ParentRevisionID != "" {
		return existing.ParentRevisionID, nil
	}

	activeRevision, err := s.latestActiveRevision(ctx)
	if err != nil || activeRevision == nil {
		return "", err
	}
	return activeRevision.ID, nil
}

func (s *Service) revisionSaveResult(ctx context.Context, revision console.ConfigRevision, message string) (*RevisionSaveResult, error) {
	detail, err := s.revisionDetail(ctx, revision)
	if err != nil {
		return nil, err
	}
	return &RevisionSaveResult{RevisionDetail: *detail, Message: message}, nil
}
