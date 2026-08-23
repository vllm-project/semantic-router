package accesspublisher

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"
)

// RuntimeDiagnostics is one read-only observation of publication progress and
// replica acknowledgements for a namespace. It contains no policy documents,
// credentials, Redis keys, or routing topology.
type RuntimeDiagnostics struct {
	NamespaceID                     string               `json:"namespaceId"`
	QuotaPartition                  string               `json:"quotaPartition"`
	AsOf                            time.Time            `json:"asOf"`
	Readiness                       ReadinessDiagnostics `json:"readiness"`
	ActivePublicationID             string               `json:"activePublicationId,omitempty"`
	CandidatePublicationID          string               `json:"candidatePublicationId,omitempty"`
	OpenPublications                int64                `json:"openPublications"`
	ActiveReplicas                  []string             `json:"activeReplicas"`
	RecordedRequiredReplicas        []string             `json:"recordedRequiredReplicas"`
	BarrierAcknowledgementsRequired bool                 `json:"barrierAcknowledgementsRequired"`
	BarrierAcknowledgements         []string             `json:"barrierAcknowledgements"`
	RoutingAcknowledgements         []string             `json:"routingAcknowledgements"`
	MissingBarrierAcks              []string             `json:"missingBarrierAcks"`
	MissingRoutingAcks              []string             `json:"missingRoutingAcks"`
}

// ReadinessDiagnostics is the stable wire view of publication readiness. The
// runtime's internal Readiness value remains free of transport concerns.
type ReadinessDiagnostics struct {
	Ready           bool   `json:"ready"`
	Reason          string `json:"reason"`
	RuntimeEpoch    uint64 `json:"runtimeEpoch"`
	DesiredRevision uint64 `json:"desiredRevision"`
	AppliedRevision uint64 `json:"appliedRevision"`
	AccessGate      string `json:"accessGate,omitempty"`
	RoutingGate     string `json:"routingGate,omitempty"`
	ProjectorLag    uint64 `json:"projectorLag"`
}

// Diagnostics observes active state without mutating replica leases or
// publication lifecycle. Expired replica scores are excluded from the result
// but left for the normal publication coordinator to reap atomically.
func (s *RedisStore) Diagnostics(
	ctx context.Context,
	namespaceID string,
	partition string,
) (RuntimeDiagnostics, error) {
	if s == nil || s.client == nil {
		return RuntimeDiagnostics{}, errors.New("redis publication diagnostics are unavailable")
	}
	keys, err := NewKeyspace(s.keyPrefix, namespaceID, partition)
	if err != nil {
		return RuntimeDiagnostics{}, err
	}
	asOf, err := s.client.Time(ctx).Result()
	if err != nil {
		return RuntimeDiagnostics{}, fmt.Errorf("read publication diagnostics time: %w", err)
	}
	readiness, err := s.Readiness(ctx, namespaceID, partition)
	if err != nil {
		return RuntimeDiagnostics{}, err
	}
	result := RuntimeDiagnostics{
		NamespaceID: namespaceID, QuotaPartition: partition, AsOf: asOf.UTC(),
		Readiness: readinessDiagnostics(readiness), ActivePublicationID: readiness.AccessGate,
	}
	pipeline := s.client.Pipeline()
	activeReplicas := pipeline.ZRangeByScore(ctx, keys.ReplicaIndex(), &redis.ZRangeBy{
		Min: strconv.FormatInt(asOf.UnixMilli()+1, 10), Max: "+inf",
	})
	openPublications := pipeline.ZCard(ctx, keys.OpenPublications())
	candidate := pipeline.HGet(ctx, keys.PendingPublication(), FieldPublicationID)
	var required, barrierAcks, routingAcks *redis.StringSliceCmd
	var publicationPlan *redis.StringCmd
	if result.ActivePublicationID != "" {
		required = pipeline.SMembers(ctx, keys.PublicationRequiredReplicas(result.ActivePublicationID))
		barrierAcks = pipeline.SMembers(ctx, keys.PublicationBarrierAcks(result.ActivePublicationID))
		routingAcks = pipeline.SMembers(ctx, keys.PublicationRoutingAcks(result.ActivePublicationID))
		publicationPlan = pipeline.HGet(ctx, keys.Publication(result.ActivePublicationID), "plan")
	}
	_, err = pipeline.Exec(ctx)
	if err != nil && !errors.Is(err, redis.Nil) {
		return RuntimeDiagnostics{}, fmt.Errorf("read publication diagnostics: %w", err)
	}
	result.ActiveReplicas = sortedUnique(activeReplicas.Val())
	result.OpenPublications = openPublications.Val()
	if candidate.Err() == nil {
		result.CandidatePublicationID = candidate.Val()
	}
	if required != nil {
		barrierRequired, planErr := barrierAcknowledgementsRequired(publicationPlan.Val())
		if planErr != nil {
			return RuntimeDiagnostics{}, fmt.Errorf("read active publication diagnostics plan: %w", planErr)
		}
		result.RecordedRequiredReplicas = sortedUnique(required.Val())
		result.BarrierAcknowledgementsRequired = barrierRequired
		result.BarrierAcknowledgements = sortedUnique(barrierAcks.Val())
		result.RoutingAcknowledgements = sortedUnique(routingAcks.Val())
		if result.BarrierAcknowledgementsRequired {
			result.MissingBarrierAcks = missingStrings(result.ActiveReplicas, result.BarrierAcknowledgements)
		} else {
			result.MissingBarrierAcks = []string{}
		}
		result.MissingRoutingAcks = missingStrings(result.ActiveReplicas, result.RoutingAcknowledgements)
	}
	return result, nil
}

func barrierAcknowledgementsRequired(encodedPlan string) (bool, error) {
	var plan storedPlan
	if err := decodeStrict([]byte(encodedPlan), &plan); err != nil {
		return false, fmt.Errorf("decode immutable publication plan: %w", err)
	}
	return len(plan.Barriers) > 0, nil
}

func readinessDiagnostics(value Readiness) ReadinessDiagnostics {
	return ReadinessDiagnostics(value)
}

func sortedUnique(values []string) []string {
	if len(values) == 0 {
		return []string{}
	}
	result := append([]string(nil), values...)
	sort.Strings(result)
	write := 0
	for _, value := range result {
		if value == "" || (write > 0 && result[write-1] == value) {
			continue
		}
		result[write] = value
		write++
	}
	return result[:write]
}

func missingStrings(required []string, acknowledged []string) []string {
	accepted := make(map[string]struct{}, len(acknowledged))
	for _, value := range acknowledged {
		accepted[value] = struct{}{}
	}
	missing := make([]string, 0)
	for _, value := range required {
		if _, found := accepted[value]; !found {
			missing = append(missing, value)
		}
	}
	return missing
}
