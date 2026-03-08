package configlifecycle

import (
	"context"
	"encoding/json"
	"log"
	"sort"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

const compatibilityActorID = "dashboard-compat"

type revisionPersistenceOptions struct {
	source         string
	summary        string
	action         string
	triggerSource  string
	revisionStatus console.ConfigRevisionStatus
	previousStatus console.ConfigRevisionStatus
	deployStatus   console.DeployEventStatus
	message        string
	metadata       map[string]interface{}
}

func (s *Service) recordSuccessfulCompatibilityChange(yamlData []byte, opts revisionPersistenceOptions) {
	if s == nil || s.Stores == nil || s.Stores.Revisions == nil {
		return
	}

	ctx := context.Background()
	previousRevision, err := s.latestActiveRevision(ctx)
	if err != nil {
		log.Printf("Console persistence: failed to load latest active revision: %v", err)
		return
	}

	documentJSON, err := marshalRevisionDocument(yamlData)
	if err != nil {
		log.Printf("Console persistence: failed to encode revision document: %v", err)
		return
	}

	now := time.Now().UTC()
	metadata := s.enrichedMetadata(opts.metadata)
	revision := &console.ConfigRevision{
		Status:            opts.revisionStatus,
		Source:            opts.source,
		Summary:           opts.summary,
		DocumentJSON:      documentJSON,
		RuntimeConfigYAML: string(yamlData),
		CreatedBy:         compatibilityActorID,
		ActivatedAt:       &now,
		Metadata:          metadata,
	}
	if previousRevision != nil {
		revision.ParentRevisionID = previousRevision.ID
	}

	if err := s.Stores.Revisions.SaveConfigRevision(ctx, revision); err != nil {
		log.Printf("Console persistence: failed to save config revision: %v", err)
		return
	}

	if previousRevision != nil && opts.previousStatus != "" {
		previousRevision.Status = opts.previousStatus
		if err := s.Stores.Revisions.SaveConfigRevision(ctx, previousRevision); err != nil {
			log.Printf("Console persistence: failed to update previous active revision %s: %v", previousRevision.ID, err)
		}
	}

	s.saveCompatibilityDeployEvent(ctx, revision.ID, previousRevision, opts, metadata, now)
	s.appendCompatibilityAuditEvent(ctx, revision.ID, opts, metadata, now)
}

func (s *Service) latestActiveRevision(ctx context.Context) (*console.ConfigRevision, error) {
	if s == nil || s.Stores == nil || s.Stores.Revisions == nil {
		return nil, nil
	}

	revisions, err := s.Stores.Revisions.ListConfigRevisions(ctx, console.ConfigRevisionFilter{
		Status: console.ConfigRevisionStatusActive,
		Limit:  50,
	})
	if err != nil {
		return nil, err
	}
	if len(revisions) == 0 {
		return nil, nil
	}

	sort.SliceStable(revisions, func(i, j int) bool {
		return revisionSortTime(revisions[i].ActivatedAt, revisions[i].CreatedAt).After(
			revisionSortTime(revisions[j].ActivatedAt, revisions[j].CreatedAt),
		)
	})
	return &revisions[0], nil
}

func marshalRevisionDocument(yamlData []byte) (json.RawMessage, error) {
	var document interface{}
	if err := yaml.Unmarshal(yamlData, &document); err != nil {
		return nil, err
	}

	payload, err := json.Marshal(normalizeYAMLValue(document))
	if err != nil {
		return nil, err
	}
	return payload, nil
}

func (s *Service) baseMetadata(metadata map[string]interface{}) map[string]interface{} {
	enriched := map[string]interface{}{
		"config_path":    s.ConfigPath,
		"config_dir":     s.ConfigDir,
		"runtime_target": s.runtimeTarget(),
	}
	for key, value := range metadata {
		enriched[key] = value
	}
	return enriched
}

func (s *Service) enrichedMetadata(metadata map[string]interface{}) map[string]interface{} {
	enriched := s.baseMetadata(metadata)
	enriched["compatibility_adapter"] = true
	return enriched
}

func (s *Service) runtimeTarget() string {
	switch {
	case isRunningInContainer() && isManagedContainerConfigPath(s.ConfigPath):
		return "embedded-dashboard"
	case s.shouldPropagateToManagedContainer():
		return "managed-container"
	default:
		return "workspace-files"
	}
}

func (s *Service) saveCompatibilityDeployEvent(
	ctx context.Context,
	revisionID string,
	previousRevision *console.ConfigRevision,
	opts revisionPersistenceOptions,
	metadata map[string]interface{},
	now time.Time,
) {
	if s.Stores == nil || s.Stores.Deployments == nil {
		return
	}

	deployEvent := &console.DeployEvent{
		RevisionID:    revisionID,
		Status:        opts.deployStatus,
		TriggerSource: opts.triggerSource,
		Message:       opts.message,
		RuntimeTarget: s.runtimeTarget(),
		Metadata:      metadata,
		StartedAt:     &now,
		CompletedAt:   &now,
	}
	if opts.previousStatus == console.ConfigRevisionStatusRolledBack && previousRevision != nil {
		deployEvent.RollbackRevisionID = previousRevision.ID
	}
	if err := s.Stores.Deployments.SaveDeployEvent(ctx, deployEvent); err != nil {
		log.Printf("Console persistence: failed to save deploy event: %v", err)
	}
}

func (s *Service) appendCompatibilityAuditEvent(
	ctx context.Context,
	revisionID string,
	opts revisionPersistenceOptions,
	metadata map[string]interface{},
	now time.Time,
) {
	if s.Stores == nil || s.Stores.Audit == nil {
		return
	}

	auditEvent := &console.AuditEvent{
		ActorType:  console.PrincipalTypeServiceAccount,
		ActorID:    compatibilityActorID,
		Action:     opts.action,
		TargetType: "config_revision",
		TargetID:   revisionID,
		Outcome:    console.AuditOutcomeSuccess,
		Message:    opts.message,
		Metadata:   metadata,
		OccurredAt: now,
	}
	if err := s.Stores.Audit.AppendAuditEvent(ctx, auditEvent); err != nil {
		log.Printf("Console persistence: failed to append audit event: %v", err)
	}
}
