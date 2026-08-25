package application

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	catalogpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog/postgres"
)

const (
	defaultLease         = 45 * time.Second
	defaultRenewInterval = 15 * time.Second
	coldStartTransitions = 8
)

var ErrReplicaNotReady = errors.New("provider catalog replica is not ready")

type Coordinator interface {
	PublicationCoordinator
	SnapshotCoordinator
}

type PublicationCoordinator interface {
	Stage(context.Context, catalogpostgres.StageRequest) (catalogpostgres.State, error)
	Acknowledge(context.Context, catalogpostgres.AcknowledgeRequest) (catalogpostgres.ReplicaAcknowledgement, error)
	Activate(context.Context, catalogpostgres.ActivateRequest) (catalogpostgres.State, error)
}

type SnapshotCoordinator interface {
	State(context.Context) (catalogpostgres.State, error)
	ActiveSnapshot(context.Context) (*providercatalog.Snapshot, error)
	DesiredSnapshot(context.Context) (*providercatalog.Snapshot, error)
}

type ReplicaOptions struct {
	ReplicaID             string
	RolloutGroups         []providercatalog.RolloutGroup
	RequiredRolloutGroups []providercatalog.RolloutGroup
	Lease                 time.Duration
	RenewInterval         time.Duration
	Now                   func() time.Time
}

type ReplicaReadiness struct {
	Ready             bool
	Reason            string
	ActiveRevision    string
	DesiredRevision   string
	DesiredStatus     string
	DesiredReason     string
	DesiredCompatible bool
	LastReconciledAt  time.Time
}

type Replica struct {
	coordinator Coordinator
	registry    *providercatalog.Registry
	options     ReplicaOptions

	mu        sync.RWMutex
	readiness ReplicaReadiness
}

func NewReplica(
	coordinator Coordinator,
	registry *providercatalog.Registry,
	options ReplicaOptions,
) (*Replica, error) {
	if coordinator == nil || registry == nil {
		return nil, fmt.Errorf("provider catalog coordinator and integration registry are required")
	}
	for _, plane := range []providercatalog.CapabilityPlane{
		providercatalog.CapabilityPlaneControl, providercatalog.CapabilityPlaneData,
	} {
		if digest, err := registry.CapabilityDigest(plane); err != nil || len(digest) != 32 {
			return nil, fmt.Errorf("provider Catalog %s-plane capabilities are unavailable", plane)
		}
	}
	if err := validateReplicaID(options.ReplicaID); err != nil {
		return nil, err
	}
	memberships, err := providercatalog.CanonicalRolloutGroups(options.RolloutGroups)
	if err != nil {
		return nil, err
	}
	required, err := providercatalog.CanonicalRolloutGroups(options.RequiredRolloutGroups)
	if err != nil {
		return nil, err
	}
	options.RolloutGroups = memberships
	options.RequiredRolloutGroups = required
	if options.Lease == 0 {
		options.Lease = defaultLease
	}
	if options.RenewInterval == 0 {
		options.RenewInterval = defaultRenewInterval
	}
	if options.Lease < time.Second || options.Lease > 5*time.Minute ||
		options.RenewInterval < time.Second || options.RenewInterval >= options.Lease {
		return nil, fmt.Errorf("provider catalog ACK lease or renewal interval is invalid")
	}
	return &Replica{coordinator: coordinator, registry: registry, options: options}, nil
}

// BootstrapRegistry is the explicit empty-catalog bootstrap operation. The
// caller supplies the generation it observed before the mutation. Replaying
// the same installed Registry revision is idempotent; a different desired
// revision fails the coordinator's compare-and-swap guard.
func (r *Replica) BootstrapRegistry(
	ctx context.Context,
	expectedGeneration uint64,
) (providercatalog.PublicationState, error) {
	return r.coordinator.Stage(ctx, catalogpostgres.StageRequest{
		Snapshot: r.registry.Snapshot(), ExpectedGeneration: expectedGeneration,
		RequiredRolloutGroups: append([]providercatalog.RolloutGroup(nil), r.options.RequiredRolloutGroups...),
	})
}

// EnsureColdStart converges an empty durable catalog onto the
// application-installed Registry. Concurrent replicas may stage, acknowledge,
// and activate the same immutable revision; compare-and-swap conflicts are
// reread instead of failing startup. Any other desired or active revision
// remains exclusively operator controlled.
func (r *Replica) EnsureColdStart(ctx context.Context) error {
	installedRevision := r.registry.Snapshot().Revision()
	for range coldStartTransitions {
		if err := ctx.Err(); err != nil {
			return err
		}
		state, err := r.coordinator.State(ctx)
		if err != nil {
			return fmt.Errorf("read Provider Catalog cold-start state: %w", err)
		}
		switch {
		case state.ActiveRevision == installedRevision:
			if reconcileErr := r.Reconcile(ctx); reconcileErr != nil {
				return fmt.Errorf("reconcile application-installed Provider Catalog: %w", reconcileErr)
			}
			return nil
		case state.ActiveRevision != "":
			return nil
		case state.DesiredRevision == "":
			if _, bootstrapErr := r.BootstrapRegistry(ctx, state.Generation); bootstrapErr != nil {
				if errors.Is(bootstrapErr, providercatalog.ErrPublicationConflict) {
					continue
				}
				return fmt.Errorf("stage application-installed Provider Catalog: %w", bootstrapErr)
			}
			continue
		case state.DesiredRevision != installedRevision:
			return nil
		}

		if reconcileErr := r.Reconcile(ctx); reconcileErr != nil {
			return fmt.Errorf("acknowledge application-installed Provider Catalog: %w", reconcileErr)
		}
		state, err = r.coordinator.State(ctx)
		if err != nil {
			return fmt.Errorf("reread Provider Catalog cold-start state: %w", err)
		}
		if state.ActiveRevision != "" || state.DesiredRevision != installedRevision {
			continue
		}
		if _, err := r.Activate(ctx, installedRevision, state.Generation); err != nil {
			switch {
			case errors.Is(err, providercatalog.ErrPublicationConflict):
				continue
			case errors.Is(err, providercatalog.ErrActivationBlocked):
				// This replica has durably ACKed its rollout groups. Another
				// replica can now complete a distributed gate and activation.
				return nil
			default:
				return fmt.Errorf("activate application-installed Provider Catalog: %w", err)
			}
		}
	}
	return fmt.Errorf("application-installed Provider Catalog did not converge after %d state transitions", coldStartTransitions)
}

// Stage is an explicit operator action for later revisions. The caller must
// provide the exact observed desired revision and generation; configured
// rollout groups cannot be inferred from observed instance ACKs.
func (r *Replica) Stage(
	ctx context.Context,
	snapshot *providercatalog.Snapshot,
	expectedDesiredRevision string,
	expectedGeneration uint64,
) (providercatalog.PublicationState, error) {
	return r.coordinator.Stage(ctx, catalogpostgres.StageRequest{
		Snapshot: snapshot, ExpectedDesiredRevision: expectedDesiredRevision,
		ExpectedGeneration:    expectedGeneration,
		RequiredRolloutGroups: append([]providercatalog.RolloutGroup(nil), r.options.RequiredRolloutGroups...),
	})
}

func (r *Replica) Activate(
	ctx context.Context,
	revision string,
	expectedGeneration uint64,
) (providercatalog.PublicationState, error) {
	return r.coordinator.Activate(ctx, catalogpostgres.ActivateRequest{
		Revision: revision, ExpectedGeneration: expectedGeneration,
	})
}

// Run renews compatibility leases until cancellation. It never compiles or
// stages provider product metadata.
func (r *Replica) Run(ctx context.Context) error {
	_ = r.Reconcile(ctx)
	ticker := time.NewTicker(r.options.RenewInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			_ = r.Reconcile(ctx)
		}
	}
}

func (r *Replica) Reconcile(ctx context.Context) error {
	state, err := r.coordinator.State(ctx)
	if err != nil {
		r.setReadiness(ReplicaReadiness{Reason: "catalog_state_unavailable", LastReconciledAt: r.now()})
		return err
	}
	readiness := ReplicaReadiness{
		Reason: "active_snapshot_unavailable", ActiveRevision: state.ActiveRevision,
		DesiredRevision: state.DesiredRevision, DesiredStatus: "absent",
		LastReconciledAt: r.now(),
	}
	var reconcileErr error
	if state.DesiredRevision != "" {
		readiness.DesiredStatus = "unavailable"
		_, desiredErr := r.coordinator.DesiredSnapshot(ctx)
		status, reason := catalogpostgres.AckCompatible, ""
		readiness.DesiredCompatible = desiredErr == nil
		if desiredErr != nil {
			if !errors.Is(desiredErr, catalogpostgres.ErrCorruptSnapshot) {
				readiness.DesiredReason = "desired_snapshot_unavailable"
				reconcileErr = desiredErr
			} else {
				status, reason = catalogpostgres.AckIncompatible, "desired snapshot fails replica validation"
				readiness.DesiredStatus = string(catalogpostgres.AckIncompatible)
				readiness.DesiredReason = "desired_snapshot_incompatible"
			}
		} else {
			readiness.DesiredStatus = string(catalogpostgres.AckCompatible)
		}
		if desiredErr == nil || errors.Is(desiredErr, catalogpostgres.ErrCorruptSnapshot) {
			if ackErr := r.acknowledgeAll(ctx, state.DesiredRevision, status, reason); ackErr != nil {
				readiness.DesiredStatus = "acknowledgement_failed"
				readiness.DesiredReason = "desired_acknowledgement_failed"
				if reconcileErr == nil {
					reconcileErr = ackErr
				}
			}
		}
	}
	activeCompatible := false
	if state.ActiveRevision != "" {
		_, activeErr := r.coordinator.ActiveSnapshot(ctx)
		status, reason := catalogpostgres.AckCompatible, ""
		activeCompatible = activeErr == nil
		if activeErr != nil {
			if !errors.Is(activeErr, catalogpostgres.ErrCorruptSnapshot) {
				readiness.Reason = "active_snapshot_unavailable"
				r.setReadiness(readiness)
				return activeErr
			}
			status, reason = catalogpostgres.AckIncompatible, "active snapshot fails replica validation"
		}
		if state.ActiveRevision != state.DesiredRevision {
			if ackErr := r.acknowledgeAll(ctx, state.ActiveRevision, status, reason); ackErr != nil && reconcileErr == nil {
				reconcileErr = ackErr
			}
		}
	}
	switch {
	case state.ActiveRevision == "":
		readiness.Reason = "active_snapshot_unavailable"
	case !activeCompatible:
		readiness.Reason = "active_snapshot_incompatible"
	default:
		readiness.Ready = true
		readiness.Reason = "ready"
	}
	r.setReadiness(readiness)
	return reconcileErr
}

func (r *Replica) acknowledgeAll(
	ctx context.Context,
	revision string,
	status catalogpostgres.AckStatus,
	reason string,
) error {
	for _, group := range r.options.RolloutGroups {
		digest, err := r.registry.CapabilityDigest(group.Plane)
		if err != nil {
			return err
		}
		if _, err := r.coordinator.Acknowledge(ctx, catalogpostgres.AcknowledgeRequest{
			Revision: revision, ReplicaID: r.options.ReplicaID, RolloutGroup: group,
			CapabilityDigest: digest, Status: status, Reason: reason, Lease: r.options.Lease,
		}); err != nil {
			return err
		}
	}
	return nil
}

func (r *Replica) Readiness() ReplicaReadiness {
	if r == nil {
		return ReplicaReadiness{Reason: "catalog_runtime_unavailable"}
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.readiness
}

func (r *Replica) Ready(context.Context) error {
	readiness := r.Readiness()
	if !readiness.Ready {
		return fmt.Errorf("%w: %s", ErrReplicaNotReady, readiness.Reason)
	}
	return nil
}

func (r *Replica) setReadiness(readiness ReplicaReadiness) {
	r.mu.Lock()
	r.readiness = readiness
	r.mu.Unlock()
}

func (r *Replica) now() time.Time {
	if r.options.Now != nil {
		return r.options.Now().UTC()
	}
	return time.Now().UTC()
}

func validateReplicaID(value string) error {
	if value == "" || len(value) > 256 || !utf8.ValidString(value) {
		return fmt.Errorf("provider Catalog replica ID is required and bounded")
	}
	if strings.TrimSpace(value) != value {
		return fmt.Errorf("provider Catalog replica ID is not canonical")
	}
	for _, char := range value {
		if char < 0x20 || char == 0x7f {
			return fmt.Errorf("provider Catalog replica ID is not canonical")
		}
	}
	return nil
}
