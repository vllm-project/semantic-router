package configlifecycle

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

const (
	consoleServiceActorID = "dashboard-console"
	compatibilityActorID  = "dashboard-compat"
)

const revisionAPITriggerSource = "revision_api"

type revisionMutationContext struct {
	actorID              string
	triggerSource        string
	compatibilityAdapter bool
}

func revisionAPIMutationContext() revisionMutationContext {
	return revisionAPIMutationContextForActor("")
}

func revisionAPIMutationContextForActor(actorID string) revisionMutationContext {
	return revisionMutationContext{
		actorID:       coalesceString(actorID, consoleServiceActorID),
		triggerSource: revisionAPITriggerSource,
	}
}

func compatibilityMutationContext(triggerSource string) revisionMutationContext {
	return revisionMutationContext{
		actorID:              compatibilityActorID,
		triggerSource:        triggerSource,
		compatibilityAdapter: true,
	}
}

// RevisionDraftInput captures the persisted revision fields accepted by the draft API.
type RevisionDraftInput struct {
	ID                string
	ParentRevisionID  string
	Source            string
	Summary           string
	DSLSource         string
	Document          interface{}
	RuntimeConfigYAML string
	Metadata          map[string]interface{}
}

// RevisionSaveResult returns the saved draft plus a caller-friendly status message.
type RevisionSaveResult struct {
	RevisionDetail
	Message string
}

// RevisionValidationResult returns the validated revision plus a caller-friendly status message.
type RevisionValidationResult struct {
	RevisionDetail
	Message string
}

func ensureRevisionStoreAvailable(s *Service) error {
	if s == nil || s.Stores == nil || s.Stores.Revisions == nil {
		return &Error{
			StatusCode: http.StatusServiceUnavailable,
			Code:       "console_store_unavailable",
			Message:    "Console revision store is not configured.",
		}
	}
	return nil
}

func ensureRevisionID(err error, revisionID string) error {
	if err != nil {
		return err
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

func (s *Service) revisionMutationMetadata(
	base map[string]interface{},
	override map[string]interface{},
	operation string,
	mutation revisionMutationContext,
) map[string]interface{} {
	metadata := s.baseMetadata(mergeMetadataMaps(base, override))
	metadata["operation"] = operation
	metadata["trigger_source"] = mutation.triggerSource
	if mutation.compatibilityAdapter {
		metadata["compatibility_adapter"] = true
	}
	return metadata
}

func (s *Service) appendRevisionAudit(
	ctx context.Context,
	actorID string,
	action string,
	revisionID string,
	outcome console.AuditOutcome,
	message string,
	metadata map[string]interface{},
) error {
	if s == nil || s.Stores == nil || s.Stores.Audit == nil {
		return nil
	}

	event := &console.AuditEvent{
		ActorType:  console.PrincipalTypeServiceAccount,
		ActorID:    actorID,
		Action:     action,
		TargetType: "config_revision",
		TargetID:   revisionID,
		Outcome:    outcome,
		Message:    message,
		Metadata:   metadata,
		OccurredAt: time.Now().UTC(),
	}
	if err := s.Stores.Audit.AppendAuditEvent(ctx, event); err != nil {
		return fmt.Errorf("failed to append config revision audit event: %w", err)
	}
	return nil
}

func mergeMetadataMaps(base, override map[string]interface{}) map[string]interface{} {
	merged := make(map[string]interface{}, len(base)+len(override))
	for key, value := range base {
		merged[key] = value
	}
	for key, value := range override {
		merged[key] = value
	}
	return merged
}

func coalesceString(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func existingSource(revision *console.ConfigRevision) string {
	if revision == nil {
		return ""
	}
	return revision.Source
}

func existingSummary(revision *console.ConfigRevision) string {
	if revision == nil {
		return ""
	}
	return revision.Summary
}

func existingCreatedBy(revision *console.ConfigRevision) string {
	if revision == nil {
		return ""
	}
	return revision.CreatedBy
}

func existingCreatedAt(revision *console.ConfigRevision) time.Time {
	if revision == nil {
		return time.Time{}
	}
	return revision.CreatedAt
}

func existingMetadata(revision *console.ConfigRevision) map[string]interface{} {
	if revision == nil {
		return nil
	}
	return revision.Metadata
}
