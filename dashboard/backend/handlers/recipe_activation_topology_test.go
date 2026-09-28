package handlers

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
)

type fakeRuntimeTopology struct {
	mu           sync.Mutex
	inventory    runtimeTopologyInventory
	inventoryErr error
	applyErr     error
	rollbackErr  error
	commitErr    error
	applied      []recipe.ActivationTopologyState
	rolledBack   []recipe.ActivationTopologyState
	committed    []recipe.ActivationTopologyState
	credentials  []string
}

func (f *fakeRuntimeTopology) Inventory(context.Context) (runtimeTopologyInventory, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.inventoryErr != nil {
		return runtimeTopologyInventory{}, f.inventoryErr
	}
	return runtimeTopologyInventory{
		Storage:          append([]string(nil), f.inventory.Storage...),
		ContainerRunning: cloneRunningInventory(f.inventory.ContainerRunning),
	}, nil
}

func TestManagedStoragePortInspectionFailsClosed(t *testing.T) {
	tests := []struct {
		name    string
		payload string
		wantErr string
	}{
		{
			name: "loopback bindings",
			payload: `{"network_mode":"default","publish_all_ports":false,` +
				`"configured":{"6379/tcp":[{"HostIp":"127.0.0.1","HostPort":"6379"}]},` +
				`"actual":{"6379/tcp":[{"HostIp":"::1","HostPort":"6379"}]}}`,
		},
		{
			name:    "host network",
			payload: `{"network_mode":"host","publish_all_ports":false,"configured":{},"actual":{}}`,
			wantErr: "host network mode",
		},
		{
			name: "container network namespace",
			payload: `{"network_mode":"container:peer","publish_all_ports":false,` +
				`"configured":{},"actual":{}}`,
			wantErr: "container network mode",
		},
		{
			name: "ipv4 loopback range",
			payload: `{"network_mode":"default","publish_all_ports":false,` +
				`"configured":{"6379/tcp":[{"HostIp":"127.12.34.56","HostPort":"6379"}]},` +
				`"actual":{}}`,
		},
		{
			name:    "publish all",
			payload: `{"network_mode":"default","publish_all_ports":true,"configured":{},"actual":{}}`,
			wantErr: "PublishAllPorts",
		},
		{
			name: "configured wildcard",
			payload: `{"network_mode":"default","publish_all_ports":false,` +
				`"configured":{"5432/tcp":[{"HostIp":"0.0.0.0","HostPort":"5432"}]},"actual":{}}`,
			wantErr: "configured published port",
		},
		{
			name: "effective wildcard",
			payload: `{"network_mode":"default","publish_all_ports":false,"configured":{},` +
				`"actual":{"19530/tcp":[{"HostIp":"","HostPort":"19530"}]}}`,
			wantErr: "actual published port",
		},
		{name: "malformed", payload: `{`, wantErr: "inspection is invalid"},
		{name: "missing fields", payload: `{}`, wantErr: "inspection is invalid"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateManagedStoragePortInspection([]byte(test.payload))
			if test.wantErr == "" && err != nil {
				t.Fatalf("validate inspection: %v", err)
			}
			if test.wantErr != "" && (err == nil || !strings.Contains(err.Error(), test.wantErr)) {
				t.Fatalf("error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}

func TestManagedRouterManagementPortInspectionFailsClosed(t *testing.T) {
	management := recipe.ActivationManagementListener{BindAddress: "0.0.0.0", Port: 8080, HostPort: 8080}
	tests := []struct {
		name    string
		payload string
		wantErr string
	}{
		{
			name: "exact loopback publication",
			payload: `{"network_mode":"vllm-sr-network","publish_all_ports":false,` +
				`"configured":{"8080/tcp":[{"HostIp":"127.0.0.1","HostPort":"8080"}]},` +
				`"actual":{"8080/tcp":[{"HostIp":"127.0.0.1","HostPort":"8080"}]}}`,
		},
		{
			name: "configured wildcard",
			payload: `{"network_mode":"vllm-sr-network","publish_all_ports":false,` +
				`"configured":{"8080/tcp":[{"HostIp":"0.0.0.0","HostPort":"8080"}]},` +
				`"actual":{"8080/tcp":[{"HostIp":"0.0.0.0","HostPort":"8080"}]}}`,
			wantErr: "not loopback",
		},
		{
			name: "actual mismatch",
			payload: `{"network_mode":"vllm-sr-network","publish_all_ports":false,` +
				`"configured":{"8080/tcp":[{"HostIp":"127.0.0.1","HostPort":"8080"}]},` +
				`"actual":{"8080/tcp":[{"HostIp":"::1","HostPort":"8080"}]}}`,
			wantErr: "do not match",
		},
		{
			name: "wrong host port",
			payload: `{"network_mode":"vllm-sr-network","publish_all_ports":false,` +
				`"configured":{"8080/tcp":[{"HostIp":"127.0.0.1","HostPort":"18080"}]},` +
				`"actual":{"8080/tcp":[{"HostIp":"127.0.0.1","HostPort":"18080"}]}}`,
			wantErr: "does not match",
		},
		{
			name:    "host network",
			payload: `{"network_mode":"host","publish_all_ports":false,"configured":{},"actual":{}}`,
			wantErr: "host network mode",
		},
		{
			name:    "container namespace",
			payload: `{"network_mode":"container:peer","publish_all_ports":false,"configured":{},"actual":{}}`,
			wantErr: "container network mode",
		},
		{
			name:    "publish all",
			payload: `{"network_mode":"vllm-sr-network","publish_all_ports":true,"configured":{},"actual":{}}`,
			wantErr: "PublishAllPorts",
		},
		{
			name: "missing actual",
			payload: `{"network_mode":"vllm-sr-network","publish_all_ports":false,` +
				`"configured":{"8080/tcp":[{"HostIp":"127.0.0.1","HostPort":"8080"}]},"actual":{}}`,
			wantErr: "actual Router management publication",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateManagedRouterPortInspection([]byte(test.payload), management)
			if test.wantErr == "" && err != nil {
				t.Fatalf("validate inspection: %v", err)
			}
			if test.wantErr != "" && (err == nil || !strings.Contains(err.Error(), test.wantErr)) {
				t.Fatalf("error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}

func TestManagedContainerNotFoundClassificationIsNarrow(t *testing.T) {
	if !managedContainerInspectReportsNotFound([]byte("Error: No such container: missing")) {
		t.Fatal("known no-such-container result was not classified as absent")
	}
	if managedContainerInspectReportsNotFound([]byte("permission denied")) {
		t.Fatal("runtime permission failure must not be classified as absent")
	}
}

func TestActivationPreviewRejectsUnsafeManagedStorageInventory(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	topology := testTopologyForManagedRuntime()
	topology.inventoryErr = errors.New("managed redis storage is unsafe: wildcard binding")
	activator := topologyTestActivator(store, configPath, topology, func(raw []byte) []byte { return raw })

	plan, err := activator.Preview(context.Background(), activationRequest(summary))
	if packageErrorCode(err) != recipe.ErrorManagedStorageIsolation || plan.PlanDigest != "" {
		t.Fatalf("unsafe inventory preview plan=%#v err=%v", plan, err)
	}
}

func (f *fakeRuntimeTopology) Apply(_ context.Context, state recipe.ActivationTopologyState, credential string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.applied = append(f.applied, state)
	f.credentials = append(f.credentials, credential)
	return f.applyErr
}

func (f *fakeRuntimeTopology) Rollback(_ context.Context, state recipe.ActivationTopologyState) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.rolledBack = append(f.rolledBack, state)
	return f.rollbackErr
}

func (f *fakeRuntimeTopology) Commit(_ context.Context, state recipe.ActivationTopologyState) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.committed = append(f.committed, state)
	return f.commitErr
}

func cloneRunningInventory(source map[string]bool) map[string]bool {
	result := make(map[string]bool, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}

func TestActivationPlanPreviewBindsExactConfirmationAndReconciles(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	topology := testTopologyForManagedRuntime()
	activator := topologyTestActivator(store, configPath, topology, func(raw []byte) []byte {
		return []byte(strings.Replace(string(raw), "port: 8899", "port: 9999", 1))
	})
	request := activationRequest(summary)
	plan, err := activator.Preview(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Mode != recipe.ActivationModeStackRecreation || !plan.RequiresConfirmation || plan.PlanDigest == "" {
		t.Fatalf("plan = %#v", plan)
	}
	if _, activationErr := activator.Activate(context.Background(), request); packageErrorCode(activationErr) != recipe.ErrorActivationConfirmation {
		t.Fatalf("unconfirmed activation error = %v", activationErr)
	}
	request.ExpectedPlanDigest = plan.PlanDigest
	request.ConfirmStackRecreation = true
	result, err := activator.Activate(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if result.PlanDigest != plan.PlanDigest || result.Mode != recipe.ActivationModeStackRecreation || len(topology.applied) != 1 || len(topology.committed) != 1 {
		t.Fatalf("result=%#v applied=%d committed=%d", result, len(topology.applied), len(topology.committed))
	}
	if _, _, err := store.Transaction(); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("transaction remains after commit: %v", err)
	}
}

func TestActivationPlanRejectsManagementPortChangeBeforeJournal(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	current := withManagedManagementListener(mustReadFile(t, configPath), 8080, "disabled")
	if err := writeActivationConfig(configPath, current); err != nil {
		t.Fatal(err)
	}
	topology := testTopologyForManagedRuntime()
	activator := topologyTestActivator(store, configPath, topology, func(raw []byte) []byte {
		return withManagedManagementListener(raw, 9090, "disabled")
	})

	plan, err := activator.Preview(context.Background(), activationRequest(summary))
	if packageErrorCode(err) != recipe.ErrorActivationIncompatible || plan.PlanDigest != "" {
		t.Fatalf("management port preview plan=%#v err=%v", plan, err)
	}
	if len(topology.applied) != 0 {
		t.Fatal("management port incompatibility mutated topology")
	}
	if _, _, err := store.Transaction(); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("activation journal exists: %v", err)
	}
}

func TestActivationPlanRejectsManagementBindChangeBeforeJournal(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	current := withManagedManagementListener(mustReadFile(t, configPath), 8080, "disabled")
	if err := writeActivationConfig(configPath, current); err != nil {
		t.Fatal(err)
	}
	topology := testTopologyForManagedRuntime()
	activator := topologyTestActivator(store, configPath, topology, func(raw []byte) []byte {
		target := withManagedManagementListener(raw, 8080, "disabled")
		return []byte(strings.Replace(string(target), "bind_address: 0.0.0.0", "bind_address: 127.0.0.1", 1))
	})

	plan, err := activator.Preview(context.Background(), activationRequest(summary))
	if packageErrorCode(err) != recipe.ErrorActivationIncompatible || plan.PlanDigest != "" {
		t.Fatalf("management bind preview plan=%#v err=%v", plan, err)
	}
	if len(topology.applied) != 0 {
		t.Fatal("management bind incompatibility mutated topology")
	}
	if _, _, err := store.Transaction(); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("activation journal exists: %v", err)
	}
}

func TestActivationPlanRequiresRecreationForManagementAuthChange(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	current := withManagedManagementListener(mustReadFile(t, configPath), 8080, "disabled")
	if err := writeActivationConfig(configPath, current); err != nil {
		t.Fatal(err)
	}
	topology := testTopologyForManagedRuntime()
	activator := topologyTestActivator(store, configPath, topology, func(raw []byte) []byte {
		return withManagedManagementListener(raw, 8080, "bearer")
	})

	plan, err := activator.Preview(context.Background(), activationRequest(summary))
	if err != nil {
		t.Fatal(err)
	}
	if plan.Mode != recipe.ActivationModeStackRecreation || !plan.RequiresConfirmation {
		t.Fatalf("auth transition plan=%#v", plan)
	}
	if plan.ManagementBefore.Port != 8080 || plan.ManagementAfter.BindAddress != "0.0.0.0" {
		t.Fatalf("management endpoints missing from plan: %#v", plan)
	}
}

func TestActivationPlanRejectsUnreachableManagedManagementBind(t *testing.T) {
	t.Setenv("TARGET_ROUTER_API_URL", "http://vllm-sr-router-container:8080")
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	current := withManagedManagementListener(mustReadFile(t, configPath), 8080, "disabled")
	if err := writeActivationConfig(configPath, current); err != nil {
		t.Fatal(err)
	}
	topology := testTopologyForManagedRuntime()
	activator := topologyTestActivator(store, configPath, topology, func(raw []byte) []byte {
		return []byte(strings.Replace(
			string(withManagedManagementListener(raw, 8080, "disabled")),
			"bind_address: 0.0.0.0", "bind_address: 127.0.0.1", 1,
		))
	})

	plan, err := activator.Preview(context.Background(), activationRequest(summary))
	if packageErrorCode(err) != recipe.ErrorActivationIncompatible || plan.PlanDigest != "" {
		t.Fatalf("unreachable management preview plan=%#v err=%v", plan, err)
	}
}

func TestManagedSplitIdentityRequiresReachableManagementBindDespiteStaleTargetURL(t *testing.T) {
	t.Setenv(routerContainerNameEnv, "managed-router")
	t.Setenv(dashboardContainerNameEnv, "managed-dashboard")
	t.Setenv("TARGET_ROUTER_API_URL", "http://localhost:8080")
	if !managedSplitManagementReachabilityRequired() {
		t.Fatal("split container identity must require a Dashboard-reachable management listener")
	}
}

func TestActivationPlanRepairsStoppedRequiredStorage(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	topology := &fakeRuntimeTopology{inventory: runtimeTopologyInventory{
		Storage: []string{"redis"},
		ContainerRunning: map[string]bool{
			managedContainerNameForService("router"): true,
			managedContainerNameForService("envoy"):  true,
			managedContainerNameForStorage("redis"):  false,
		},
	}}
	activator := topologyTestActivator(store, configPath, topology, func(raw []byte) []byte {
		return []byte(strings.Replace(string(raw), "global:\n", "global:\n  services:\n    response_api:\n      enabled: true\n      store_backend: redis\n      redis:\n        address: redis:6379\n", 1))
	})
	request := activationRequest(summary)
	plan, err := activator.Preview(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Mode != recipe.ActivationModeStackRecreation || !reflect.DeepEqual(plan.Storage.Repair, []string{"redis"}) || len(plan.Storage.Add) != 0 {
		t.Fatalf("repair plan = %#v", plan)
	}
	request.ExpectedPlanDigest, request.ConfirmStackRecreation = plan.PlanDigest, true
	if _, err := activator.Activate(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	var repair *recipe.ActivationContainerTransition
	for index := range topology.applied[0].Containers {
		transition := &topology.applied[0].Containers[index]
		if transition.Service == "redis" {
			repair = transition
		}
	}
	if repair == nil || repair.Action != "repair" || repair.WasRunning {
		t.Fatalf("repair transition = %#v", repair)
	}
}

func TestTopologyApplyFailureRestoresConfigPointerReachabilityAndJournal(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	previous := mustReadFile(t, configPath)
	topology := testTopologyForManagedRuntime()
	topology.applyErr = errors.New("replacement Envoy failed")
	activator := topologyTestActivator(store, configPath, topology, func(raw []byte) []byte {
		return []byte(strings.Replace(string(raw), "port: 8899", "port: 9999", 1))
	})
	request := activationRequest(summary)
	plan, err := activator.Preview(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	request.ExpectedPlanDigest, request.ConfirmStackRecreation = plan.PlanDigest, true
	if _, err := activator.Activate(context.Background(), request); packageErrorCode(err) != recipe.ErrorActivationFailed {
		t.Fatalf("activation error = %v", err)
	}
	if !bytes.Equal(previous, mustReadFile(t, configPath)) || len(topology.rolledBack) != 1 {
		t.Fatal("topology failure did not restore config and containers")
	}
	if _, state, err := store.ActivationStatus(); err != nil || state != recipe.ActivationNone {
		t.Fatalf("activation status=%q err=%v", state, err)
	}
}

func TestTopologyRecoveryRollsBackPendingJournalFromTopologyWriteCrashWindow(t *testing.T) {
	store, activator, topology, transaction, previous := managedTopologyRecoveryFixture(t)
	overwriteActivationTransaction(t, store, transaction)

	if err := activator.Recover(context.Background()); err != nil {
		t.Fatalf("Recover(): %v", err)
	}
	if len(topology.rolledBack) != 1 || topology.rolledBack[0].TransactionID != transaction.ID {
		t.Fatalf("rolled back topology = %#v", topology.rolledBack)
	}
	if !bytes.Equal(previous, mustReadFile(t, activator.configPath)) {
		t.Fatal("recovery did not restore the previous runtime config")
	}
	if _, _, err := store.Transaction(); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("transaction remains after recovery: %v", err)
	}
}

func TestTopologyRecoveryRetriesCrashWindowAfterRollbackFailure(t *testing.T) {
	store, activator, topology, transaction, _ := managedTopologyRecoveryFixture(t)
	overwriteActivationTransaction(t, store, transaction)
	topology.rollbackErr = errors.New("container runtime temporarily unavailable")

	if err := activator.Recover(context.Background()); packageErrorCode(err) != recipe.ErrorActivationRollback {
		t.Fatalf("first Recover() error = %v", err)
	}
	current, _, err := store.Transaction()
	if err != nil || current.State != recipe.ActivationInconsistent || current.TopologyMode != recipe.ActivationTopologyNone {
		t.Fatalf("retry transaction = %#v, %v", current, err)
	}
	topology.rollbackErr = nil
	if err := activator.Recover(context.Background()); err != nil {
		t.Fatalf("retry Recover(): %v", err)
	}
	if len(topology.rolledBack) != 2 {
		t.Fatalf("rollback attempts = %d, want 2", len(topology.rolledBack))
	}
	if _, _, err := store.Transaction(); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("transaction remains after retry: %v", err)
	}
}

func TestTopologyRecoveryKeepsManagedMissingJournalFailClosed(t *testing.T) {
	store, activator, topology, transaction, _ := managedTopologyRecoveryFixture(t)
	if err := os.Remove(activationTopologyPath(store, transaction)); err != nil {
		t.Fatal(err)
	}

	err := activator.Recover(context.Background())
	if packageErrorCode(err) != recipe.ErrorActivationRollback {
		t.Fatalf("Recover() error = %v", err)
	}
	if cause := errors.Unwrap(err); cause == nil || !strings.Contains(cause.Error(), "managed activation transaction is missing its topology journal") {
		t.Fatalf("Recover() cause = %v", cause)
	}
	if len(topology.rolledBack) != 0 {
		t.Fatal("missing managed topology must not attempt an inferred rollback")
	}
}

func TestTopologyRecoveryKeepsPendingCorruptJournalFailClosed(t *testing.T) {
	store, activator, topology, transaction, _ := managedTopologyRecoveryFixture(t)
	overwriteActivationTransaction(t, store, transaction)
	if err := os.WriteFile(activationTopologyPath(store, transaction), []byte("{\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	err := activator.Recover(context.Background())
	if packageErrorCode(err) != recipe.ErrorActivationRollback {
		t.Fatalf("Recover() error = %v", err)
	}
	if len(topology.rolledBack) != 0 {
		t.Fatal("corrupt topology must not attempt rollback")
	}
}

func TestTopologyCommitCleanupFailureRemainsDurableAndRecoveryFinalizes(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	topology := testTopologyForManagedRuntime()
	topology.commitErr = errors.New("backup cleanup interrupted")
	activator := topologyTestActivator(store, configPath, topology, func(raw []byte) []byte {
		return []byte(strings.Replace(string(raw), "port: 8899", "port: 9999", 1))
	})
	request := activationRequest(summary)
	plan, err := activator.Preview(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	request.ExpectedPlanDigest, request.ConfirmStackRecreation = plan.PlanDigest, true
	if _, activationErr := activator.Activate(context.Background(), request); packageErrorCode(activationErr) != recipe.ErrorActivationConflict {
		t.Fatalf("commit cleanup error = %v", activationErr)
	}
	transaction, _, err := store.Transaction()
	if err != nil || transaction.State != "committing" {
		t.Fatalf("transaction=%#v err=%v", transaction, err)
	}
	if _, err := store.ActivationTopology(transaction); err != nil {
		t.Fatalf("topology journal was deleted before cleanup: %v", err)
	}
	topology.commitErr = nil
	if err := activator.Recover(context.Background()); err != nil {
		t.Fatalf("Recover(): %v", err)
	}
	if _, state, err := store.ActivationStatus(); err != nil || state != recipe.ActivationActive {
		t.Fatalf("activation status=%q err=%v", state, err)
	}
}

func TestDeactivationPreviewsConfirmsAndReconcilesTopology(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	topology := testTopologyForManagedRuntime()
	activator := topologyTestActivator(store, configPath, topology, func(raw []byte) []byte {
		return []byte(strings.Replace(string(raw), "port: 8899", "port: 9999", 1))
	})
	activateRequest := activationRequest(summary)
	activationPlan, err := activator.Preview(context.Background(), activateRequest)
	if err != nil {
		t.Fatal(err)
	}
	activateRequest.ExpectedPlanDigest, activateRequest.ConfirmStackRecreation = activationPlan.PlanDigest, true
	if _, activationErr := activator.Activate(context.Background(), activateRequest); activationErr != nil {
		t.Fatal(activationErr)
	}
	deactivationPlan, err := activator.PreviewDeactivation(context.Background())
	if err != nil || deactivationPlan.Mode != recipe.ActivationModeStackRecreation {
		t.Fatalf("deactivation plan=%#v err=%v", deactivationPlan, err)
	}
	if _, deactivationErr := activator.Deactivate(context.Background()); packageErrorCode(deactivationErr) != recipe.ErrorActivationConfirmation {
		t.Fatalf("unconfirmed deactivation error=%v", deactivationErr)
	}
	result, err := activator.Deactivate(context.Background(), recipe.DeactivateRequest{
		ExpectedPlanDigest: deactivationPlan.PlanDigest, ConfirmStackRecreation: true,
	})
	if err != nil || result.Mode != recipe.ActivationModeStackRecreation {
		t.Fatalf("Deactivate()=%#v err=%v", result, err)
	}
	if len(topology.applied) != 2 || len(topology.committed) != 2 {
		t.Fatalf("topology calls applied=%d committed=%d", len(topology.applied), len(topology.committed))
	}
}

func TestBearerActivationUsesPrivateScopedCredentialWithoutResponseLeak(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	topology := testTopologyForManagedRuntime()
	activator := topologyTestActivator(store, configPath, topology, withManagementBearer)
	request := activationRequest(summary)
	plan, err := activator.Preview(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	assertBearerActivationPlan(t, plan)
	request.ExpectedPlanDigest, request.ConfirmStackRecreation = plan.PlanDigest, true
	result, err := activator.Activate(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	assertPrivateManagementCredential(t, store, topology, result, configPath)
}

func assertBearerActivationPlan(t *testing.T, plan recipe.ActivationPlan) {
	t.Helper()
	if plan.ManagementAuth.ServiceRole != recipe.ManagementCredentialRole || !reflect.DeepEqual(plan.ManagementAuth.ServicePermissions, recipe.ManagementCredentialPermissions()) {
		t.Fatalf("auth plan=%#v", plan.ManagementAuth)
	}
}

func assertPrivateManagementCredential(t *testing.T, store *recipe.Store, topology *fakeRuntimeTopology, result recipe.ActivateResult, configPath string) {
	t.Helper()
	token, err := store.ManagementCredential()
	if err != nil || len(token) != 64 || len(topology.credentials) != 1 || topology.credentials[0] != token {
		t.Fatalf("credential provisioning failed: err=%v", err)
	}
	encoded := []byte(result.RecipeDigest + result.PlanDigest)
	if bytes.Contains(encoded, []byte(token)) || bytes.Contains(mustReadFile(t, configPath), []byte(token)) {
		t.Fatal("management credential leaked into response or runtime config")
	}
	credentialPath := filepath.Join(store.Root(), "credentials", "router-management.token")
	info, err := os.Stat(credentialPath)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("credential mode=%v", info.Mode())
	}
}

func TestBearerActivationUsesAuthoritativeRuntimeCredentialInsteadOfStaleStoreToken(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	persistedToken, err := store.EnsureManagementCredential()
	if err != nil {
		t.Fatal(err)
	}
	runtimeToken := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	t.Setenv(recipe.ManagementCredentialEnv, runtimeToken)
	topology := testTopologyForManagedRuntime()
	activator := topologyTestActivator(store, configPath, topology, withManagementBearer)
	request := activationRequest(summary)
	plan, err := activator.Preview(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	request.ExpectedPlanDigest, request.ConfirmStackRecreation = plan.PlanDigest, true
	if _, err := activator.Activate(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	if len(topology.credentials) != 1 || topology.credentials[0] != runtimeToken {
		t.Fatalf("activation credential = %q, want runtime credential", topology.credentials)
	}
	credentialPath := filepath.Join(store.Root(), "credentials", "router-management.token")
	persistedAfter := strings.TrimSpace(string(mustReadFile(t, credentialPath)))
	if persistedAfter != persistedToken {
		t.Fatalf("persisted credential changed from %q to %q", persistedToken, persistedAfter)
	}
}

func topologyTestActivator(store *recipe.Store, configPath string, topology *fakeRuntimeTopology, realize func([]byte) []byte) *RecipeActivator {
	return NewRecipeActivator(RecipeActivatorOptions{
		Store: store, ConfigPath: configPath, ConfigDir: filepath.Dir(configPath), Topology: topology,
		RealizeConfig: func(raw []byte, _ string) ([]byte, error) { return realize(raw), nil },
		ApplyRuntime:  func(path, _ string) (string, error) { return path, nil },
		VerifyRuntime: func(context.Context, string) error { return nil },
		VerifyEnvoy:   func(context.Context) error { return nil },
	})
}

func managedTopologyRecoveryFixture(t *testing.T) (*recipe.Store, *RecipeActivator, *fakeRuntimeTopology, recipe.ActivationTransaction, []byte) {
	t.Helper()
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	previous := mustReadFile(t, configPath)
	topology := testTopologyForManagedRuntime()
	activator := topologyTestActivator(store, configPath, topology, func(raw []byte) []byte {
		return []byte(strings.Replace(string(raw), "port: 8899", "port: 9999", 1))
	})
	plan, err := activator.Preview(context.Background(), activationRequest(summary))
	if err != nil {
		t.Fatal(err)
	}
	transaction, err := store.BeginActivation(summary.RecipeDigest, previous)
	if err != nil {
		t.Fatal(err)
	}
	state := topologyStateForPlan(plan, transaction, topology.inventory)
	if err := store.WriteActivationTopology(transaction, state); err != nil {
		t.Fatal(err)
	}
	return store, activator, topology, transaction, previous
}

func overwriteActivationTransaction(t *testing.T, store *recipe.Store, transaction recipe.ActivationTransaction) {
	t.Helper()
	if transaction.State != "pending" || transaction.TopologyMode != recipe.ActivationTopologyNone {
		t.Fatalf("pre-topology transaction = %#v", transaction)
	}
	journalPath := filepath.Join(store.Root(), "activation-pending.json")
	journal := mustReadFile(t, journalPath)
	updated := bytes.Replace(journal, []byte(`"topology_mode": "managed"`), []byte(`"topology_mode": "none"`), 1)
	if bytes.Equal(updated, journal) {
		t.Fatal("managed topology mode not found in activation journal")
	}
	if err := os.WriteFile(journalPath, updated, 0o600); err != nil {
		t.Fatal(err)
	}
}

func activationTopologyPath(store *recipe.Store, transaction recipe.ActivationTransaction) string {
	return filepath.Join(store.Root(), "transactions", transaction.ID, "topology.json")
}

func withManagedManagementListener(config []byte, port int, mode string) []byte {
	auth := "      auth:\n        mode: disabled\n"
	if mode == "bearer" {
		auth = "      auth:\n        mode: bearer\n        tokens:\n          - env: VLLM_SR_DASHBOARD_RECIPE_TOKEN\n            role: dashboard_control_plane\n        roles:\n          dashboard_control_plane:\n            - config.read\n"
	}
	block := fmt.Sprintf(
		"global:\n  services:\n    management_api:\n      bind_address: 0.0.0.0\n      port: %d\n      remote_exposure: false\n%s",
		port,
		auth,
	)
	return []byte(strings.Replace(string(config), "global:\n", block, 1))
}

func testTopologyForManagedRuntime() *fakeRuntimeTopology {
	running := map[string]bool{}
	for _, service := range []string{"router", "envoy"} {
		running[managedContainerNameForService(service)] = true
	}
	for _, backend := range []string{"redis", "postgres", "milvus"} {
		running[managedContainerNameForStorage(backend)] = true
	}
	return &fakeRuntimeTopology{inventory: runtimeTopologyInventory{
		Storage: []string{"milvus", "postgres", "redis"}, ContainerRunning: running,
	}}
}

func packageErrorCode(err error) string {
	packageErr, _ := recipe.AsPackageError(err)
	if packageErr == nil {
		return ""
	}
	return packageErr.Code
}
