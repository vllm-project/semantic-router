package configlifecycle

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

// RevisionSummary is the revision-native read model for history listings.
type RevisionSummary struct {
	ID                string
	ParentRevisionID  string
	Status            console.ConfigRevisionStatus
	Source            string
	Summary           string
	CreatedBy         string
	RuntimeTarget     string
	LastDeployStatus  console.DeployEventStatus
	LastDeployMessage string
	ActivatedAt       *time.Time
	LastDeployedAt    *time.Time
	CreatedAt         time.Time
	UpdatedAt         time.Time
}

// RevisionDetail expands a revision summary with the persisted document payload.
type RevisionDetail struct {
	RevisionSummary
	Document          interface{}
	RuntimeConfigYAML string
	Metadata          map[string]interface{}
}

func (s *Service) ListRevisions(limit int) ([]RevisionSummary, error) {
	if s == nil || s.Stores == nil || s.Stores.Revisions == nil {
		return []RevisionSummary{}, nil
	}

	ctx := context.Background()
	revisions, err := s.Stores.Revisions.ListConfigRevisions(ctx, console.ConfigRevisionFilter{Limit: limit})
	if err != nil {
		return nil, err
	}

	summaries := make([]RevisionSummary, 0, len(revisions))
	for _, revision := range revisions {
		summary, err := s.revisionSummary(ctx, revision)
		if err != nil {
			return nil, err
		}
		summaries = append(summaries, summary)
	}
	sort.SliceStable(summaries, func(i, j int) bool {
		return revisionSortTime(summaries[i].ActivatedAt, summaries[i].CreatedAt).After(
			revisionSortTime(summaries[j].ActivatedAt, summaries[j].CreatedAt),
		)
	})
	return summaries, nil
}

func (s *Service) CurrentRevision() (*RevisionDetail, error) {
	if s == nil || s.Stores == nil || s.Stores.Revisions == nil {
		return nil, nil
	}

	revision, err := s.latestActiveRevision(context.Background())
	if err != nil || revision == nil {
		return nil, err
	}
	return s.revisionDetail(context.Background(), *revision)
}

func (s *Service) revisionSummary(ctx context.Context, revision console.ConfigRevision) (RevisionSummary, error) {
	summary := RevisionSummary{
		ID:               revision.ID,
		ParentRevisionID: revision.ParentRevisionID,
		Status:           revision.Status,
		Source:           revision.Source,
		Summary:          revision.Summary,
		CreatedBy:        revision.CreatedBy,
		ActivatedAt:      revision.ActivatedAt,
		CreatedAt:        revision.CreatedAt,
		UpdatedAt:        revision.UpdatedAt,
		RuntimeTarget:    stringMetadata(revision.Metadata, "runtime_target"),
	}

	if s == nil || s.Stores == nil || s.Stores.Deployments == nil {
		return summary, nil
	}

	events, err := s.Stores.Deployments.ListDeployEvents(ctx, console.DeployEventFilter{
		RevisionID: revision.ID,
		Limit:      1,
	})
	if err != nil {
		return RevisionSummary{}, err
	}
	if len(events) == 0 {
		return summary, nil
	}

	event := events[0]
	summary.LastDeployStatus = event.Status
	summary.LastDeployMessage = event.Message
	summary.LastDeployedAt = event.CompletedAt
	if summary.RuntimeTarget == "" {
		summary.RuntimeTarget = event.RuntimeTarget
	}
	return summary, nil
}

func (s *Service) revisionDetail(ctx context.Context, revision console.ConfigRevision) (*RevisionDetail, error) {
	summary, err := s.revisionSummary(ctx, revision)
	if err != nil {
		return nil, err
	}

	document, err := decodeRevisionDocument(revision.DocumentJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to decode revision document %s: %w", revision.ID, err)
	}

	return &RevisionDetail{
		RevisionSummary:   summary,
		Document:          document,
		RuntimeConfigYAML: revision.RuntimeConfigYAML,
		Metadata:          revision.Metadata,
	}, nil
}

func decodeRevisionDocument(documentJSON []byte) (interface{}, error) {
	if len(documentJSON) == 0 {
		return map[string]interface{}{}, nil
	}

	var document interface{}
	if err := json.Unmarshal(documentJSON, &document); err != nil {
		return nil, err
	}
	return document, nil
}

func revisionYAMLBytes(revision console.ConfigRevision) ([]byte, error) {
	if revision.RuntimeConfigYAML != "" {
		return []byte(revision.RuntimeConfigYAML), nil
	}
	if len(revision.DocumentJSON) == 0 {
		return nil, fmt.Errorf("revision document is empty")
	}

	document, err := decodeRevisionDocument(revision.DocumentJSON)
	if err != nil {
		return nil, err
	}
	return yaml.Marshal(document)
}

func revisionSortTime(activatedAt *time.Time, createdAt time.Time) time.Time {
	if activatedAt != nil && !activatedAt.IsZero() {
		return activatedAt.UTC()
	}
	return createdAt.UTC()
}

func stringMetadata(metadata map[string]interface{}, key string) string {
	if len(metadata) == 0 {
		return ""
	}
	value, ok := metadata[key]
	if !ok {
		return ""
	}
	stringValue, _ := value.(string)
	return stringValue
}
