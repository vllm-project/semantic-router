package handlers

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
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
}

func (f *fakeRuntimeTopology) Inventory(context.Context) (runtimeTopologyInventory, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.inventoryErr != nil {
		return runtimeTopologyInventory{}, f.inventoryErr
	}
	return runtimeTopologyInventory{Storage: append([]string(nil), f.inventory.Storage...)}, nil
}

// runtimeCalls counts what an activation asked of the running containers.
type runtimeCalls struct {
	mu       sync.Mutex
	applied  int
	verified int
}

func (c *runtimeCalls) counts() (int, int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.applied, c.verified
}

func TestEnvironmentTopologyReadsTheStorageServeStarted(t *testing.T) {
	t.Setenv(managedStorageBackends, "redis, milvus")
	inventory, err := environmentRuntimeTopology{}.Inventory(context.Background())
	if err != nil || !reflect.DeepEqual(inventory.Storage, []string{"milvus", "redis"}) {
		t.Fatalf("Inventory() = %#v, %v", inventory, err)
	}
	t.Setenv(managedStorageBackends, "redis,unknown")
	if _, err := (environmentRuntimeTopology{}).Inventory(context.Background()); err == nil {
		t.Fatal("an unknown storage backend must not pass as inventory")
	}
}

func TestStackRecreationIsConfirmedThenLeftToServe(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	calls := &runtimeCalls{}
	activator := topologyTestActivator(store, configPath, testTopologyForManagedRuntime(), movedListener, calls)
	request := activationRequest(summary)
	plan, err := activator.Preview(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Mode != recipe.ActivationModeStackRecreation || !plan.RequiresConfirmation || plan.PlanDigest == "" {
		t.Fatalf("plan = %#v", plan)
	}
	effects := strings.Join(plan.Effects, "\n")
	if !strings.Contains(effects, "the next `vllm-sr serve` recreates the managed Router and Envoy containers") {
		t.Fatalf("effects = %q", effects)
	}
	if _, activationErr := activator.Activate(context.Background(), request); packageErrorCode(activationErr) != recipe.ErrorActivationConfirmation {
		t.Fatalf("unconfirmed activation error = %v", activationErr)
	}

	result, err := activator.Activate(context.Background(), confirmed(request, plan))
	if err != nil {
		t.Fatal(err)
	}
	if result.Status != recipe.ActivationResultRestartRequired || result.Message != "Restart required: run `vllm-sr serve` to apply." ||
		result.PlanDigest != plan.PlanDigest || result.Mode != recipe.ActivationModeStackRecreation {
		t.Fatalf("result = %#v", result)
	}
	if applied, verified := calls.counts(); applied != 0 || verified != 0 {
		t.Fatalf("a deferred activation touched the running containers: applied=%d verified=%d", applied, verified)
	}
	published := mustReadFile(t, configPath)
	if !strings.Contains(string(published), "port: 9999") {
		t.Fatal("the Recipe's config was not published")
	}
	assertPendingRestart(t, configPath, published, recipeRecreationDetail)
	active, state, err := store.ActivationStatus()
	if err != nil || state != recipe.ActivationActive || active.RecipeDigest != summary.RecipeDigest ||
		active.RealizedConfigDigest != activationDigest(published) {
		t.Fatalf("active=%#v state=%q err=%v", active, state, err)
	}
	assertNoActivationJournal(t, store)
}

func TestStandaloneStackRecreationNamesOnlyTheRouter(t *testing.T) {
	t.Setenv("VLLM_SR_GATEWAY", "standalone")
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	activator := topologyTestActivator(store, configPath, testTopologyForManagedRuntime(), movedListener, nil)
	plan, err := activator.Preview(context.Background(), activationRequest(summary))
	if err != nil {
		t.Fatal(err)
	}
	if plan.Mode != recipe.ActivationModeStackRecreation ||
		!strings.Contains(strings.Join(plan.Effects, "\n"), "recreates the managed Router container, which serves the listeners") {
		t.Fatalf("plan = %#v", plan)
	}
	if listeners := plan.ListenersAfter; len(listeners) != 1 || listeners[0].Port != 9999 {
		t.Fatalf("listeners = %#v, want the target listener", listeners)
	}
}

func TestAddedStorageIsStartedByServe(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	activator := topologyTestActivator(store, configPath, &fakeRuntimeTopology{}, func(raw []byte) []byte {
		return []byte(strings.Replace(string(raw), "global:\n", "global:\n  services:\n    response_api:\n      enabled: true\n      store_backend: redis\n      redis:\n        address: redis:6379\n", 1))
	}, nil)
	request := activationRequest(summary)
	plan, err := activator.Preview(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Mode != recipe.ActivationModeStackRecreation || !reflect.DeepEqual(plan.Storage.Add, []string{"redis"}) ||
		len(plan.Storage.Repair) != 0 {
		t.Fatalf("storage plan = %#v", plan.Storage)
	}
	if !strings.Contains(strings.Join(plan.Effects, "\n"), "the next `vllm-sr serve` starts the storage the Recipe adds") {
		t.Fatalf("effects = %q", plan.Effects)
	}
	result, err := activator.Activate(context.Background(), confirmed(request, plan))
	if err != nil || result.Status != recipe.ActivationResultRestartRequired {
		t.Fatalf("Activate() = %#v, %v", result, err)
	}
}

func TestActivationPlanRejectsManagementPortChangeBeforeJournal(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	current := withManagedManagementListener(mustReadFile(t, configPath), 8080, "disabled")
	if err := writeActivationConfig(configPath, current); err != nil {
		t.Fatal(err)
	}
	activator := topologyTestActivator(store, configPath, testTopologyForManagedRuntime(), func(raw []byte) []byte {
		return withManagedManagementListener(raw, 9090, "disabled")
	}, nil)

	plan, err := activator.Preview(context.Background(), activationRequest(summary))
	if packageErrorCode(err) != recipe.ErrorActivationIncompatible || plan.PlanDigest != "" {
		t.Fatalf("management port preview plan=%#v err=%v", plan, err)
	}
	assertNoActivationJournal(t, store)
}

func TestActivationPlanRejectsManagementBindChangeBeforeJournal(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	current := withManagedManagementListener(mustReadFile(t, configPath), 8080, "disabled")
	if err := writeActivationConfig(configPath, current); err != nil {
		t.Fatal(err)
	}
	activator := topologyTestActivator(store, configPath, testTopologyForManagedRuntime(), func(raw []byte) []byte {
		target := withManagedManagementListener(raw, 8080, "disabled")
		return []byte(strings.Replace(string(target), "bind_address: 0.0.0.0", "bind_address: 127.0.0.1", 1))
	}, nil)

	plan, err := activator.Preview(context.Background(), activationRequest(summary))
	if packageErrorCode(err) != recipe.ErrorActivationIncompatible || plan.PlanDigest != "" {
		t.Fatalf("management bind preview plan=%#v err=%v", plan, err)
	}
	assertNoActivationJournal(t, store)
}

// The management credential `vllm-sr serve` passes the Dashboard.
const testManagementCredential = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" //nolint:gosec // Test fixture, not a credential.

func TestActivationPlanRequiresRecreationForManagementAuthChange(t *testing.T) {
	t.Setenv(recipe.ManagementCredentialEnv, testManagementCredential)
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	current := withManagedManagementListener(mustReadFile(t, configPath), 8080, "disabled")
	if err := writeActivationConfig(configPath, current); err != nil {
		t.Fatal(err)
	}
	activator := topologyTestActivator(store, configPath, testTopologyForManagedRuntime(), func(raw []byte) []byte {
		return withManagedManagementListener(raw, 8080, "bearer")
	}, nil)

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
	if !strings.Contains(strings.Join(plan.Effects, "\n"), "the recreated Router takes the target management listener and authentication boundary") {
		t.Fatalf("effects = %q", plan.Effects)
	}
}

func TestActivationPlanRejectsUnreachableManagedManagementBind(t *testing.T) {
	t.Setenv("TARGET_ROUTER_API_URL", "http://vllm-sr-router-container:8080")
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	current := withManagedManagementListener(mustReadFile(t, configPath), 8080, "disabled")
	if err := writeActivationConfig(configPath, current); err != nil {
		t.Fatal(err)
	}
	activator := topologyTestActivator(store, configPath, testTopologyForManagedRuntime(), func(raw []byte) []byte {
		return []byte(strings.Replace(
			string(withManagedManagementListener(raw, 8080, "disabled")),
			"bind_address: 0.0.0.0", "bind_address: 127.0.0.1", 1,
		))
	}, nil)

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

func TestHotSwitchTheRouterRefusesOnlyForARestartIsLeftToServe(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	activator := topologyTestActivator(store, configPath, testTopologyForHotSwitch(), realizedMarker, nil)
	activator.verifyRuntime = func(context.Context, string) error {
		return &restartNeededError{detail: "listener http timeout changed"}
	}
	plan, err := activator.Preview(context.Background(), activationRequest(summary))
	if err != nil || plan.Mode != recipe.ActivationModeHotSwitch {
		t.Fatalf("plan = %#v, %v", plan, err)
	}
	result, err := activator.Activate(context.Background(), activationRequest(summary))
	if err != nil || result.Status != recipe.ActivationResultRestartRequired || result.Mode != recipe.ActivationModeHotSwitch {
		t.Fatalf("Activate() = %#v, %v", result, err)
	}
	published := mustReadFile(t, configPath)
	assertPendingRestart(t, configPath, published, "listener http timeout changed")
	if _, state, _ := store.ActivationStatus(); state != recipe.ActivationActive {
		t.Fatalf("activation state = %q, want active", state)
	}

	again, err := activator.Activate(context.Background(), activationRequest(summary))
	if err != nil || again.Status != recipe.ActivationResultRestartRequired {
		t.Fatalf("repeated Activate() = %#v, %v", again, err)
	}
	assertNoActivationJournal(t, store)
}

func TestHotSwitchWithAChangedEnvoyConfigIsLeftToServe(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	calls := &runtimeCalls{}
	activator := topologyTestActivator(store, configPath, testTopologyForHotSwitch(), realizedMarker, calls)
	activator.applyRuntime = func(path, _ string) (string, error) {
		return path, &restartNeededError{detail: envoyRestartDetail}
	}
	result, err := activator.Activate(context.Background(), activationRequest(summary))
	if err != nil || result.Status != recipe.ActivationResultRestartRequired {
		t.Fatalf("Activate() = %#v, %v", result, err)
	}
	if _, verified := calls.counts(); verified != 1 {
		t.Fatalf("the Router's part was verified %d times, want 1", verified)
	}
	assertPendingRestart(t, configPath, mustReadFile(t, configPath), envoyRestartDetail)
}

func TestRollbackKeepsTheRestartTheRestoredConfigNeeded(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	previous := mustReadFile(t, configPath)
	activator := topologyTestActivator(store, configPath, testTopologyForHotSwitch(), realizedMarker, nil)
	activator.verifyEnvoy = func(context.Context) error {
		if bytes.Equal(mustReadFile(t, configPath), previous) {
			return nil
		}
		return errors.New("the Recipe's Envoy is not ready")
	}
	activator.applyRuntime = func(path, _ string) (string, error) {
		if bytes.Equal(mustReadFile(t, path), previous) {
			return path, &restartNeededError{detail: envoyRestartDetail}
		}
		return path, nil
	}
	if _, err := activator.Activate(context.Background(), activationRequest(summary)); packageErrorCode(err) != recipe.ErrorActivationFailed {
		t.Fatalf("Activate() error = %v", err)
	}
	if !bytes.Equal(previous, mustReadFile(t, configPath)) {
		t.Fatal("the failed activation did not restore the previous config")
	}
	if _, err := os.Stat(pendingActivationPath(configPath, pendingActivationSuffix)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("a rolled-back activation recorded a pending activation: %v", err)
	}
	if _, state, _ := store.ActivationStatus(); state != recipe.ActivationNone {
		t.Fatalf("activation state = %q, want none", state)
	}
	assertNoActivationJournal(t, store)
}

func TestRecoveryLeavesContainerRecreatingJournalsToServe(t *testing.T) {
	for _, name := range []string{"topology journal", "managed topology mode"} {
		t.Run(name, func(t *testing.T) {
			store, summary, configPath := importedActivationFixture(t, "accuracy")
			previous := mustReadFile(t, configPath)
			transaction, err := store.BeginActivation(summary.RecipeDigest, previous)
			if err != nil {
				t.Fatal(err)
			}
			if name == "topology journal" {
				topologyPath := filepath.Join(store.Root(), "transactions", transaction.ID, "topology.json")
				if writeErr := os.WriteFile(topologyPath, []byte("{}\n"), 0o600); writeErr != nil {
					t.Fatal(writeErr)
				}
			} else {
				markJournalTopologyManaged(t, store)
			}
			if writeErr := writeActivationConfig(configPath, append(append([]byte(nil), previous...), "# half applied\n"...)); writeErr != nil {
				t.Fatal(writeErr)
			}
			activator := topologyTestActivator(store, configPath, testTopologyForManagedRuntime(), realizedMarker, nil)

			err = activator.Recover(context.Background())
			packageErr, ok := recipe.AsPackageError(err)
			if !ok || packageErr.Status != http.StatusConflict || packageErr.Safe != serveRecoversTopologyMsg {
				t.Fatalf("Recover() error = %#v", err)
			}
			if _, _, err := store.Transaction(); err != nil {
				t.Fatalf("the journal must stay for `vllm-sr serve`: %v", err)
			}
			if bytes.Equal(previous, mustReadFile(t, configPath)) {
				t.Fatal("the Dashboard rolled back a transaction only `vllm-sr serve` may recover")
			}
		})
	}
}

func TestDeactivationThatRecreatesTheStackIsLeftToServe(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	source := mustReadFile(t, configPath)
	activator := topologyTestActivator(store, configPath, testTopologyForManagedRuntime(), movedListener, nil)
	request := activationRequest(summary)
	activationPlan, err := activator.Preview(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if _, activationErr := activator.Activate(context.Background(), confirmed(request, activationPlan)); activationErr != nil {
		t.Fatal(activationErr)
	}
	deactivationPlan, err := activator.PreviewDeactivation(context.Background())
	if err != nil || deactivationPlan.Mode != recipe.ActivationModeStackRecreation ||
		deactivationPlan.Effects[0] != "restore the durable source runtime config" {
		t.Fatalf("deactivation plan=%#v err=%v", deactivationPlan, err)
	}
	if _, deactivationErr := activator.Deactivate(context.Background()); packageErrorCode(deactivationErr) != recipe.ErrorActivationConfirmation {
		t.Fatalf("unconfirmed deactivation error=%v", deactivationErr)
	}
	result, err := activator.Deactivate(context.Background(), recipe.DeactivateRequest{
		ExpectedPlanDigest: deactivationPlan.PlanDigest, ConfirmStackRecreation: true,
	})
	if err != nil || result.Status != recipe.ActivationResultRestartRequired || result.Mode != recipe.ActivationModeStackRecreation ||
		result.PreviousRecipeDigest != summary.RecipeDigest {
		t.Fatalf("Deactivate()=%#v err=%v", result, err)
	}
	restored := mustReadFile(t, configPath)
	if !bytes.Equal(source, restored) {
		t.Fatal("deactivation did not restore the source config")
	}
	assertPendingRestart(t, configPath, restored, recipeRestorationDetail)
	if _, state, _ := store.ActivationStatus(); state != recipe.ActivationNone {
		t.Fatalf("activation state = %q, want none", state)
	}
}

func TestBearerActivationBindsTheCredentialServePassedWithoutWritingIt(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	token := testManagementCredential
	t.Setenv(recipe.ManagementCredentialEnv, token)
	activator := topologyTestActivator(store, configPath, testTopologyForManagedRuntime(), withManagementBearer, nil)
	request := activationRequest(summary)
	plan, err := activator.Preview(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if plan.ManagementAuth.ServiceRole != recipe.ManagementCredentialRole ||
		!reflect.DeepEqual(plan.ManagementAuth.ServicePermissions, recipe.ManagementCredentialPermissions()) {
		t.Fatalf("auth plan=%#v", plan.ManagementAuth)
	}
	result, err := activator.Activate(context.Background(), confirmed(request, plan))
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	runtimeConfig := mustReadFile(t, configPath)
	if bytes.Contains(encoded, []byte(token)) || bytes.Contains(runtimeConfig, []byte(token)) {
		t.Fatal("management credential leaked into response or runtime config")
	}
	if !bytes.Contains(runtimeConfig, []byte("env: "+recipe.ManagementCredentialEnv)) {
		t.Fatalf("runtime config does not bind the credential by name:\n%s", runtimeConfig)
	}
	err = filepath.WalkDir(store.Root(), func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil || entry.IsDir() {
			return walkErr
		}
		if bytes.Contains(mustReadFile(t, path), []byte(token)) {
			t.Errorf("the Recipe store wrote the management credential to %s", path)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestBearerActivationWithoutACredentialSaysWhoProvidesIt(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	t.Setenv(recipe.ManagementCredentialEnv, "")
	previous := mustReadFile(t, configPath)
	activator := topologyTestActivator(store, configPath, testTopologyForManagedRuntime(), withManagementBearer, nil)

	_, err := activator.Preview(context.Background(), activationRequest(summary))
	var packageErr *recipe.PackageError
	if !errors.As(err, &packageErr) || packageErr.Code != recipe.ErrorActivationIncompatible ||
		!strings.Contains(packageErr.Safe, "vllm-sr serve") {
		t.Fatalf("Preview() error = %v, want an explanation that names vllm-sr serve", err)
	}
	if _, state, _ := store.ActivationStatus(); state != recipe.ActivationNone {
		t.Fatalf("activation state = %q, want none", state)
	}
	if !bytes.Equal(mustReadFile(t, configPath), previous) {
		t.Fatal("a refused activation changed the runtime config")
	}
}

func TestRestartRequiredActivationResultIsAccepted(t *testing.T) {
	if status := activationResultHTTPStatus(recipe.ActivationResultRestartRequired); status != http.StatusAccepted {
		t.Fatalf("restart_required status = %d, want 202", status)
	}
	for _, status := range []string{"active", "inactive"} {
		if code := activationResultHTTPStatus(status); code != http.StatusOK {
			t.Fatalf("%s status = %d, want 200", status, code)
		}
	}
}

func assertPendingRestart(t *testing.T, configPath string, config []byte, detail string) {
	t.Helper()
	var record pendingActivationRecord
	if err := json.Unmarshal(mustReadFile(t, pendingActivationPath(configPath, pendingActivationSuffix)), &record); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(config)
	if record.Reason != activationReasonRestart || record.Detail != detail || record.ConfigSHA256 != hex.EncodeToString(digest[:]) {
		t.Fatalf("pending activation = %#v, want reason restart, detail %q, the published config", record, detail)
	}
}

func confirmed(request recipe.ActivateRequest, plan recipe.ActivationPlan) recipe.ActivateRequest {
	request.ExpectedPlanDigest, request.ConfirmStackRecreation = plan.PlanDigest, true
	return request
}

func movedListener(raw []byte) []byte {
	return []byte(strings.Replace(string(raw), "port: 8899", "port: 9999", 1))
}

func realizedMarker(raw []byte) []byte {
	return append(append([]byte(nil), raw...), "\n# realized recipe\n"...)
}

func topologyTestActivator(store *recipe.Store, configPath string, topology *fakeRuntimeTopology, realize func([]byte) []byte, calls *runtimeCalls) *RecipeActivator {
	if calls == nil {
		calls = &runtimeCalls{}
	}
	return NewRecipeActivator(RecipeActivatorOptions{
		Store: store, ConfigPath: configPath, ConfigDir: filepath.Dir(configPath), Topology: topology,
		RealizeConfig: func(raw []byte, _ string) ([]byte, error) { return realize(raw), nil },
		ApplyRuntime: func(path, _ string) (string, error) {
			calls.mu.Lock()
			defer calls.mu.Unlock()
			calls.applied++
			return path, nil
		},
		VerifyRuntime: func(context.Context, string) error {
			calls.mu.Lock()
			defer calls.mu.Unlock()
			calls.verified++
			return nil
		},
		VerifyEnvoy: func(context.Context) error { return nil },
	})
}

func markJournalTopologyManaged(t *testing.T, store *recipe.Store) {
	t.Helper()
	journalPath := filepath.Join(store.Root(), "activation-pending.json")
	journal := mustReadFile(t, journalPath)
	updated := bytes.Replace(journal, []byte(`"topology_mode": "none"`), []byte(`"topology_mode": "managed"`), 1)
	if bytes.Equal(updated, journal) {
		t.Fatal("topology mode not found in activation journal")
	}
	if err := os.WriteFile(journalPath, updated, 0o600); err != nil {
		t.Fatal(err)
	}
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
	return &fakeRuntimeTopology{inventory: runtimeTopologyInventory{Storage: []string{"milvus", "postgres", "redis"}}}
}

func packageErrorCode(err error) string {
	packageErr, _ := recipe.AsPackageError(err)
	if packageErr == nil {
		return ""
	}
	return packageErr.Code
}
