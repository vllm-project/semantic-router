package managed

import (
	"bytes"
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	catalogpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const (
	testActiveRevision  = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	testDesiredRevision = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
)

type coordinatorStub struct {
	state             catalogpostgres.State
	stateErr          error
	desiredErr        error
	activeErr         error
	stageState        catalogpostgres.State
	stageErr          error
	activateState     catalogpostgres.State
	activateErr       error
	advanceOnStage    bool
	advanceOnActivate bool
	stages            []catalogpostgres.StageRequest
	acks              []catalogpostgres.AcknowledgeRequest
	activations       []catalogpostgres.ActivateRequest
}

func (stub *coordinatorStub) State(context.Context) (catalogpostgres.State, error) {
	return stub.state, stub.stateErr
}

func (stub *coordinatorStub) Stage(
	_ context.Context,
	request catalogpostgres.StageRequest,
) (catalogpostgres.State, error) {
	stub.stages = append(stub.stages, request)
	if stub.stageErr == nil && stub.advanceOnStage {
		stub.state = stub.stageState
	}
	return stub.stageState, stub.stageErr
}

func (stub *coordinatorStub) Acknowledge(
	_ context.Context,
	request catalogpostgres.AcknowledgeRequest,
) (catalogpostgres.ReplicaAcknowledgement, error) {
	stub.acks = append(stub.acks, request)
	return catalogpostgres.ReplicaAcknowledgement{}, nil
}

func (stub *coordinatorStub) Activate(
	_ context.Context,
	request catalogpostgres.ActivateRequest,
) (catalogpostgres.State, error) {
	stub.activations = append(stub.activations, request)
	if stub.activateErr == nil && stub.advanceOnActivate {
		stub.state = stub.activateState
	}
	if stub.activateState.Generation != 0 {
		return stub.activateState, stub.activateErr
	}
	return stub.state, stub.activateErr
}

func (stub *coordinatorStub) ActiveSnapshot(context.Context) (*providercatalog.Snapshot, error) {
	return nil, stub.activeErr
}

func (stub *coordinatorStub) DesiredSnapshot(context.Context) (*providercatalog.Snapshot, error) {
	return nil, stub.desiredErr
}

func TestReplicaBadDesiredBlocksActivationWithoutDrainingActive(t *testing.T) {
	coordinator := &coordinatorStub{
		state: catalogpostgres.State{
			ActiveRevision: testActiveRevision, DesiredRevision: testDesiredRevision, Generation: 3,
		},
		desiredErr: catalogpostgres.ErrCorruptSnapshot,
	}
	replica := testReplica(t, coordinator)
	if err := replica.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	readiness := replica.Readiness()
	if !readiness.Ready || readiness.Reason != "ready" || readiness.DesiredCompatible ||
		readiness.DesiredStatus != string(catalogpostgres.AckIncompatible) ||
		readiness.DesiredReason != "desired_snapshot_incompatible" {
		t.Fatalf("readiness = %+v", readiness)
	}
	if len(coordinator.acks) != 4 {
		t.Fatalf("ack count = %d, want desired and active", len(coordinator.acks))
	}
	if coordinator.acks[0].Revision != testDesiredRevision ||
		coordinator.acks[0].Status != catalogpostgres.AckIncompatible ||
		coordinator.acks[1].Revision != testDesiredRevision ||
		coordinator.acks[1].Status != catalogpostgres.AckIncompatible ||
		coordinator.acks[2].Revision != testActiveRevision ||
		coordinator.acks[2].Status != catalogpostgres.AckCompatible ||
		coordinator.acks[3].Revision != testActiveRevision ||
		coordinator.acks[3].Status != catalogpostgres.AckCompatible {
		t.Fatalf("acknowledgements = %#v", coordinator.acks)
	}
}

func TestReplicaDesiredReadFailureDoesNotHideServingActive(t *testing.T) {
	desiredErr := errors.New("desired read unavailable")
	coordinator := &coordinatorStub{
		state:      catalogpostgres.State{ActiveRevision: testActiveRevision, DesiredRevision: testDesiredRevision},
		desiredErr: desiredErr,
	}
	replica := testReplica(t, coordinator)
	if err := replica.Reconcile(context.Background()); !errors.Is(err, desiredErr) {
		t.Fatalf("Reconcile() error = %v", err)
	}
	readiness := replica.Readiness()
	if !readiness.Ready || readiness.DesiredStatus != "unavailable" ||
		readiness.DesiredReason != "desired_snapshot_unavailable" {
		t.Fatalf("readiness = %+v", readiness)
	}
	if len(coordinator.acks) != 2 || coordinator.acks[0].Revision != testActiveRevision ||
		coordinator.acks[1].Revision != testActiveRevision {
		t.Fatalf("acknowledgements = %#v", coordinator.acks)
	}
}

func TestReplicaFailsReadinessWhenActiveCannotRestore(t *testing.T) {
	coordinator := &coordinatorStub{
		state:     catalogpostgres.State{ActiveRevision: testActiveRevision, DesiredRevision: testActiveRevision},
		activeErr: catalogpostgres.ErrCorruptSnapshot,
	}
	replica := testReplica(t, coordinator)
	if err := replica.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	readiness := replica.Readiness()
	if readiness.Ready || readiness.Reason != "active_snapshot_incompatible" {
		t.Fatalf("readiness = %+v", readiness)
	}
	if err := replica.Ready(context.Background()); !errors.Is(err, ErrReplicaNotReady) {
		t.Fatalf("Ready() error = %v", err)
	}
}

func TestReplicaBootstrapAndStageAreExplicitCASOperations(t *testing.T) {
	coordinator := &coordinatorStub{
		state:      catalogpostgres.State{Generation: 1},
		stageState: catalogpostgres.State{DesiredRevision: testDesiredRevision, Generation: 2},
	}
	replica := testReplica(t, coordinator)
	if err := replica.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(coordinator.stages) != 0 {
		t.Fatal("startup reconciliation staged local catalog state")
	}
	if _, err := replica.BootstrapRegistry(context.Background(), 1); err != nil {
		t.Fatal(err)
	}
	if len(coordinator.stages) != 1 || coordinator.stages[0].ExpectedGeneration != 1 ||
		coordinator.stages[0].ExpectedDesiredRevision != "" ||
		coordinator.stages[0].Snapshot.Revision() != replica.registry.Snapshot().Revision() ||
		!reflect.DeepEqual(coordinator.stages[0].RequiredRolloutGroups, testRolloutGroups()) {
		t.Fatalf("bootstrap stage = %#v", coordinator.stages)
	}
	if _, err := replica.BootstrapRegistry(context.Background(), 1); err != nil {
		t.Fatalf("idempotent bootstrap error = %v", err)
	}
	if len(coordinator.stages) != 2 || coordinator.stages[1].ExpectedGeneration != 1 ||
		coordinator.stages[1].ExpectedDesiredRevision != "" ||
		coordinator.stages[1].Snapshot.Revision() != replica.registry.Snapshot().Revision() {
		t.Fatalf("idempotent bootstrap stage = %#v", coordinator.stages)
	}
	if _, err := replica.Stage(context.Background(), nil, testDesiredRevision, 2); err != nil {
		t.Fatal(err)
	}
	last := coordinator.stages[2]
	if last.ExpectedDesiredRevision != testDesiredRevision || last.ExpectedGeneration != 2 ||
		!reflect.DeepEqual(last.RequiredRolloutGroups, testRolloutGroups()) {
		t.Fatalf("explicit stage = %#v", last)
	}
}

func TestReplicaEnsureColdStartPublishesOnlyEmptyDurableState(t *testing.T) {
	coordinator := &coordinatorStub{state: catalogpostgres.State{Generation: 1}}
	replica := testReplica(t, coordinator)
	revision := replica.registry.Snapshot().Revision()
	coordinator.stageState = catalogpostgres.State{DesiredRevision: revision, Generation: 2}
	coordinator.activateState = catalogpostgres.State{
		DesiredRevision: revision, ActiveRevision: revision, Generation: 2,
	}
	coordinator.advanceOnStage = true
	coordinator.advanceOnActivate = true

	if err := replica.EnsureColdStart(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(coordinator.stages) != 1 || coordinator.stages[0].ExpectedGeneration != 1 ||
		coordinator.stages[0].ExpectedDesiredRevision != "" ||
		coordinator.stages[0].Snapshot.Revision() != revision {
		t.Fatalf("cold-start stage = %#v", coordinator.stages)
	}
	if len(coordinator.acks) != 2*len(testRolloutGroups()) {
		t.Fatalf("cold-start acknowledgements = %#v", coordinator.acks)
	}
	if len(coordinator.activations) != 1 || coordinator.activations[0].Revision != revision ||
		coordinator.activations[0].ExpectedGeneration != 2 {
		t.Fatalf("cold-start activation = %#v", coordinator.activations)
	}

	coordinator.stages = nil
	coordinator.acks = nil
	coordinator.activations = nil
	if err := replica.EnsureColdStart(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(coordinator.stages) != 0 || len(coordinator.acks) != len(testRolloutGroups()) ||
		len(coordinator.activations) != 0 {
		t.Fatalf(
			"restart did not remain idempotent: stages=%#v acks=%#v activations=%#v",
			coordinator.stages,
			coordinator.acks,
			coordinator.activations,
		)
	}

	coordinator.state = catalogpostgres.State{
		DesiredRevision: testDesiredRevision, Generation: 3,
	}
	if err := replica.EnsureColdStart(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(coordinator.stages) != 0 || len(coordinator.activations) != 0 {
		t.Fatalf("existing rollout was crossed: stages=%#v activations=%#v", coordinator.stages, coordinator.activations)
	}
}

func TestReplicaEnsureColdStartRereadsConcurrentStageAndActivation(t *testing.T) {
	coordinator := &coldStartConflictCoordinator{coordinatorStub: coordinatorStub{
		state: catalogpostgres.State{Generation: 1},
	}}
	replica := testReplica(t, coordinator)
	revision := replica.registry.Snapshot().Revision()
	coordinator.stageState = catalogpostgres.State{DesiredRevision: revision, Generation: 2}
	coordinator.activateState = catalogpostgres.State{
		DesiredRevision: revision, ActiveRevision: revision, Generation: 2,
	}

	if err := replica.EnsureColdStart(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !coordinator.stageConflict || !coordinator.activationConflict {
		t.Fatalf(
			"concurrent transitions were not exercised: stage=%t activation=%t",
			coordinator.stageConflict,
			coordinator.activationConflict,
		)
	}
	if coordinator.state.ActiveRevision != revision {
		t.Fatalf("active revision = %q, want %q", coordinator.state.ActiveRevision, revision)
	}
}

func TestReplicaEnsureColdStartLeavesBlockedDistributedGateForPeer(t *testing.T) {
	coordinator := &coordinatorStub{}
	replica := testReplica(t, coordinator)
	revision := replica.registry.Snapshot().Revision()
	coordinator.state = catalogpostgres.State{DesiredRevision: revision, Generation: 2}
	coordinator.activateErr = &providercatalog.ActivationBlockedError{
		Revision: revision,
		Blockers: providercatalog.ActivationBlockers{Missing: []providercatalog.RolloutGroup{
			{Plane: providercatalog.CapabilityPlaneData, ID: "router"},
		}},
	}

	if err := replica.EnsureColdStart(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(coordinator.stages) != 0 || len(coordinator.acks) != len(testRolloutGroups()) ||
		len(coordinator.activations) != 1 {
		t.Fatalf(
			"blocked distributed gate did not retain the local ACK: stages=%#v acks=%#v activations=%#v",
			coordinator.stages,
			coordinator.acks,
			coordinator.activations,
		)
	}
}

type coldStartConflictCoordinator struct {
	coordinatorStub
	stageConflict      bool
	activationConflict bool
}

func (coordinator *coldStartConflictCoordinator) Stage(
	_ context.Context,
	request catalogpostgres.StageRequest,
) (catalogpostgres.State, error) {
	coordinator.stages = append(coordinator.stages, request)
	if !coordinator.stageConflict {
		coordinator.stageConflict = true
		coordinator.state = coordinator.stageState
		return catalogpostgres.State{}, providercatalog.ErrPublicationConflict
	}
	return coordinator.stageState, nil
}

func (coordinator *coldStartConflictCoordinator) Activate(
	_ context.Context,
	request catalogpostgres.ActivateRequest,
) (catalogpostgres.State, error) {
	coordinator.activations = append(coordinator.activations, request)
	if !coordinator.activationConflict {
		coordinator.activationConflict = true
		coordinator.state = coordinator.activateState
		return catalogpostgres.State{}, providercatalog.ErrPublicationConflict
	}
	return coordinator.activateState, nil
}

func TestReplicaAcknowledgementCarriesStableCapabilityDigestAndLease(t *testing.T) {
	coordinator := &coordinatorStub{
		state: catalogpostgres.State{DesiredRevision: testDesiredRevision},
	}
	replica := testReplica(t, coordinator)
	if err := replica.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(coordinator.acks) != 2 {
		t.Fatalf("ack count = %d", len(coordinator.acks))
	}
	for _, ack := range coordinator.acks {
		expectedDigest, err := replica.registry.CapabilityDigest(ack.RolloutGroup.Plane)
		if err != nil || ack.ReplicaID != "router-a" || ack.Lease != 30*time.Second ||
			len(ack.CapabilityDigest) != 32 || !bytes.Equal(ack.CapabilityDigest, expectedDigest) {
			t.Fatalf("acknowledgement = %#v", ack)
		}
	}
}

func testReplica(t *testing.T, coordinator Coordinator) *Replica {
	t.Helper()
	registry, err := providercatalog.NewRegistry(providercatalog.RegistryOptions{
		Integrations:     []providercatalog.Integration{providercatalog.IntegrationFunc(replicaTestDefinition)},
		BackendCompilers: []providercatalog.BackendCompiler{providercatalog.StaticBackendCompiler{}},
		WireFormats:      []string{"openai.chat.v1"}, CredentialAdapterIDs: []string{"bearer"},
		DiscoveryAdapterIDs: []string{"openai.models.v1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	replica, err := NewReplica(coordinator, registry, ReplicaOptions{
		ReplicaID: "router-a", RolloutGroups: testRolloutGroups(),
		RequiredRolloutGroups: []providercatalog.RolloutGroup{
			{Plane: providercatalog.CapabilityPlaneData, ID: "router"},
			{Plane: providercatalog.CapabilityPlaneControl, ID: "management"},
		},
		Lease: 30 * time.Second, RenewInterval: 10 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	return replica
}

func replicaTestDefinition() providercatalog.Definition {
	return providercatalog.Definition{
		ID: "provider", Order: 1,
		Display: providercatalog.Display{
			Name: "Provider", Description: "A provider.", Category: "Model APIs",
			Icon: providercatalog.Icon{Source: "lobe", Value: "provider", Color: false},
		},
		Interfaces: []providercatalog.Interface{{
			ID: "chat", Label: "Chat Completions", Default: true,
			WireFormat: llmprotocol.OpenAIChatV1,
			Compiler:   providercatalog.Compiler{AdapterID: providercatalog.StaticBackendCompilerID, Config: map[string]any{"path": "/chat/completions"}},
		}},
		Credential: providercatalog.Credential{Mode: providercatalog.CredentialRequired, AdapterID: "bearer", Label: "API key"},
		Origin:     providercatalog.Origin{Mode: providercatalog.OriginFixed, DefaultURL: "https://api.example.com/v1"},
		Discovery:  &providercatalog.Discovery{AdapterID: "openai.models.v1", Path: "/models"},
	}
}

func testRolloutGroups() []providercatalog.RolloutGroup {
	return []providercatalog.RolloutGroup{
		{Plane: providercatalog.CapabilityPlaneControl, ID: "management"},
		{Plane: providercatalog.CapabilityPlaneData, ID: "router"},
	}
}
