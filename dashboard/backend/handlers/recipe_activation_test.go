package handlers

import (
	"archive/zip"
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
)

func TestRecipeActivatorCommitsRawAndRealizedDigests(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	previous := mustReadFile(t, configPath)
	activator := NewRecipeActivator(RecipeActivatorOptions{
		Store:      store,
		ConfigPath: configPath,
		ConfigDir:  filepath.Dir(configPath),
		RealizeConfig: func(raw []byte, _ string) ([]byte, error) {
			return append(append([]byte(nil), raw...), []byte("\n# realized\n")...), nil
		},
		ApplyRuntime: func(path, _ string) (string, error) { return path, nil },
		VerifyRuntime: func(_ context.Context, digest string) error {
			if digest == activationDigest(previous) {
				t.Fatal("activation verified the previous config")
			}
			return nil
		},
		VerifyEnvoy: func(context.Context) error { return nil },
	})
	result, err := activator.Activate(context.Background(), activationRequest(summary))
	if err != nil {
		t.Fatalf("Activate(): %v", err)
	}
	if result.Status != "active" || result.RecipeDigest != summary.RecipeDigest {
		t.Fatalf("result = %#v", result)
	}
	pointer, state, err := store.ActivationStatus()
	if err != nil || state != recipe.ActivationActive {
		t.Fatalf("ActivationStatus() = %#v, %q, %v", pointer, state, err)
	}
	if pointer.ConfigDigest != summary.ConfigDigest || pointer.RealizedConfigDigest == pointer.ConfigDigest {
		t.Fatalf("pointer = %#v", pointer)
	}
	active := mustReadFile(t, configPath)
	if !strings.HasSuffix(string(active), "# realized\n") || activationDigest(active) != pointer.RealizedConfigDigest {
		t.Fatalf("active runtime config does not match pointer")
	}
}

func TestRecipeActivatorFailureRollsBackConfigRuntimeAndJournal(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	previous := mustReadFile(t, configPath)
	var verifyCalls atomic.Int32
	activator := NewRecipeActivator(RecipeActivatorOptions{
		Store:         store,
		ConfigPath:    configPath,
		ConfigDir:     filepath.Dir(configPath),
		RealizeConfig: func(raw []byte, _ string) ([]byte, error) { return raw, nil },
		ApplyRuntime:  func(path, _ string) (string, error) { return path, nil },
		VerifyRuntime: func(_ context.Context, _ string) error {
			if verifyCalls.Add(1) == 1 {
				return errors.New("target runtime did not activate")
			}
			return nil
		},
		VerifyEnvoy: func(context.Context) error { return nil },
	})
	_, err := activator.Activate(context.Background(), activationRequest(summary))
	packageErr, ok := recipe.AsPackageError(err)
	if !ok || packageErr.Code != recipe.ErrorActivationFailed {
		t.Fatalf("Activate() error = %#v", err)
	}
	if got := mustReadFile(t, configPath); string(got) != string(previous) {
		t.Fatalf("config not rolled back: %q", got)
	}
	if _, state, statusErr := store.ActivationStatus(); statusErr != nil || state != recipe.ActivationNone {
		t.Fatalf("ActivationStatus() state=%q err=%v", state, statusErr)
	}
	if verifyCalls.Load() != 2 {
		t.Fatalf("verify calls = %d, want target and rollback", verifyCalls.Load())
	}
}

func TestRecipeActivatorRepairsPointerRuntimeDrift(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	activator := NewRecipeActivator(RecipeActivatorOptions{
		Store:         store,
		ConfigPath:    configPath,
		ConfigDir:     filepath.Dir(configPath),
		RealizeConfig: func(raw []byte, _ string) ([]byte, error) { return raw, nil },
		ApplyRuntime:  func(path, _ string) (string, error) { return path, nil },
		VerifyRuntime: func(context.Context, string) error { return nil },
		VerifyEnvoy:   func(context.Context) error { return nil },
	})
	if _, err := activator.Activate(context.Background(), activationRequest(summary)); err != nil {
		t.Fatal(err)
	}
	drifted := append(mustReadFile(t, configPath), []byte("\n# manual drift\n")...)
	if err := writeActivationConfig(configPath, drifted); err != nil {
		t.Fatal(err)
	}
	list, err := store.List(recipe.PackageListOptions{})
	if err != nil || list.ActivationState != recipe.ActivationInconsistent || list.ActiveDigest != "" || list.Items[0].Active {
		t.Fatalf("drift list = %#v, %v", list, err)
	}
	if _, err := activator.Activate(context.Background(), activationRequest(summary)); err != nil {
		t.Fatalf("repair Activate(): %v", err)
	}
	if _, state, err := store.ActivationStatus(); err != nil || state != recipe.ActivationActive {
		t.Fatalf("repaired state = %q, %v", state, err)
	}
}

func TestRecipeActivatorIdempotentActivationDoesNotReapplyRuntime(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	var realizeCalls atomic.Int32
	var applyCalls atomic.Int32
	activator := NewRecipeActivator(RecipeActivatorOptions{
		Store:      store,
		ConfigPath: configPath,
		ConfigDir:  filepath.Dir(configPath),
		RealizeConfig: func(raw []byte, _ string) ([]byte, error) {
			realizeCalls.Add(1)
			return raw, nil
		},
		ApplyRuntime: func(path, _ string) (string, error) {
			applyCalls.Add(1)
			return path, nil
		},
		VerifyRuntime: func(context.Context, string) error { return nil },
		VerifyEnvoy:   func(context.Context) error { return nil },
	})
	request := activationRequest(summary)
	if _, err := activator.Activate(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	result, err := activator.Activate(context.Background(), request)
	if err != nil || result.Status != "active" || result.RecipeDigest != summary.RecipeDigest {
		t.Fatalf("second Activate() = %#v, %v", result, err)
	}
	if realizeCalls.Load() != 1 || applyCalls.Load() != 1 {
		t.Fatalf("idempotent activation reapplied runtime: realize=%d apply=%d", realizeCalls.Load(), applyCalls.Load())
	}
}

func TestRecipeActivatorIdempotentPointerWithStaleRouterReappliesRuntime(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	var verifyCalls atomic.Int32
	var applyCalls atomic.Int32
	activator := NewRecipeActivator(RecipeActivatorOptions{
		Store:         store,
		ConfigPath:    configPath,
		ConfigDir:     filepath.Dir(configPath),
		RealizeConfig: func(raw []byte, _ string) ([]byte, error) { return raw, nil },
		ApplyRuntime: func(path, _ string) (string, error) {
			applyCalls.Add(1)
			return path, nil
		},
		VerifyRuntime: func(context.Context, string) error {
			call := verifyCalls.Add(1)
			if call == 2 {
				return errors.New("router hash is stale")
			}
			return nil
		},
		VerifyEnvoy: func(context.Context) error { return nil },
	})
	request := activationRequest(summary)
	if _, err := activator.Activate(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	if _, err := activator.Activate(context.Background(), request); err != nil {
		t.Fatalf("repair Activate(): %v", err)
	}
	if applyCalls.Load() != 2 || verifyCalls.Load() != 3 {
		t.Fatalf("stale router repair calls: apply=%d verify=%d", applyCalls.Load(), verifyCalls.Load())
	}
}

func TestRecipeActivatorRequiresConfirmedPlanForListenerAndStorageRecreation(t *testing.T) {
	tests := []struct {
		name       string
		realize    func([]byte) []byte
		storageEnv string
	}{
		{
			name: "listener",
			realize: func(raw []byte) []byte {
				return []byte(strings.Replace(string(raw), "port: 8899", "port: 9999", 1))
			},
			storageEnv: "milvus,postgres,redis",
		},
		{
			name:       "storage",
			realize:    withManagedPostgresReplay,
			storageEnv: "redis",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			store, summary, configPath := importedActivationFixture(t, "accuracy")
			t.Setenv(managedStorageBackends, test.storageEnv)
			previous := mustReadFile(t, configPath)
			var applyCalls atomic.Int32
			activator := NewRecipeActivator(RecipeActivatorOptions{
				Store:      store,
				ConfigPath: configPath,
				ConfigDir:  filepath.Dir(configPath),
				RealizeConfig: func(raw []byte, _ string) ([]byte, error) {
					return test.realize(raw), nil
				},
				ApplyRuntime: func(path, _ string) (string, error) {
					applyCalls.Add(1)
					return path, nil
				},
				VerifyRuntime: func(context.Context, string) error { return nil },
				VerifyEnvoy:   func(context.Context) error { return nil },
			})
			_, err := activator.Activate(context.Background(), activationRequest(summary))
			packageErr, ok := recipe.AsPackageError(err)
			if !ok || packageErr.Code != recipe.ErrorActivationConfirmation || packageErr.Status != http.StatusConflict {
				t.Fatalf("Activate() error = %#v", err)
			}
			if applyCalls.Load() != 0 || string(mustReadFile(t, configPath)) != string(previous) {
				t.Fatal("unconfirmed activation mutated the runtime")
			}
			if _, _, err := store.Transaction(); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("activation journal exists: %v", err)
			}
		})
	}
}

func TestRecipeActivatorRejectsManagementAPIAuthBeforeJournal(t *testing.T) {
	for _, bearerSide := range []string{"current", "target"} {
		t.Run(bearerSide, func(t *testing.T) {
			assertManagementAuthRejectedBeforeJournal(t, bearerSide)
		})
	}
}

func assertManagementAuthRejectedBeforeJournal(t *testing.T, bearerSide string) {
	t.Helper()
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	if bearerSide == "current" {
		mustWriteActivationConfig(t, configPath, withManagementBearer(mustReadFile(t, configPath)))
	}
	before := mustReadFile(t, configPath)
	var applyCalls atomic.Int32
	activator := NewRecipeActivator(RecipeActivatorOptions{
		Store:      store,
		ConfigPath: configPath,
		ConfigDir:  filepath.Dir(configPath),
		RealizeConfig: func(raw []byte, _ string) ([]byte, error) {
			if bearerSide == "target" {
				return withManagementBearer(raw), nil
			}
			return raw, nil
		},
		ApplyRuntime: func(path, _ string) (string, error) {
			applyCalls.Add(1)
			return path, nil
		},
		VerifyRuntime: func(context.Context, string) error { return nil },
		VerifyEnvoy:   func(context.Context) error { return nil },
	})
	_, err := activator.Activate(context.Background(), activationRequest(summary))
	assertManagementAuthActivationError(t, bearerSide, err)
	if applyCalls.Load() != 0 || !bytes.Equal(before, mustReadFile(t, configPath)) {
		t.Fatal("management auth incompatibility mutated the runtime")
	}
	if _, _, err := store.Transaction(); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("activation journal exists: %v", err)
	}
}

func assertManagementAuthActivationError(t *testing.T, bearerSide string, err error) {
	t.Helper()
	packageErr, ok := recipe.AsPackageError(err)
	expectedCode := recipe.ErrorActivationConfirmation
	if bearerSide == "current" {
		expectedCode = recipe.ErrorActivationIncompatible
	}
	if !ok || packageErr.Code != expectedCode || packageErr.Status != http.StatusConflict {
		t.Fatalf("Activate() error = %#v", err)
	}
}

func TestRecipeActivatorEnvoyReadinessFailureRollsBack(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	previous := mustReadFile(t, configPath)
	var envoyCalls atomic.Int32
	activator := NewRecipeActivator(RecipeActivatorOptions{
		Store:         store,
		ConfigPath:    configPath,
		ConfigDir:     filepath.Dir(configPath),
		RealizeConfig: func(raw []byte, _ string) ([]byte, error) { return raw, nil },
		ApplyRuntime:  func(path, _ string) (string, error) { return path, nil },
		VerifyRuntime: func(context.Context, string) error { return nil },
		VerifyEnvoy: func(context.Context) error {
			if envoyCalls.Add(1) == 1 {
				return errors.New("Envoy is not ready")
			}
			return nil
		},
	})
	_, err := activator.Activate(context.Background(), activationRequest(summary))
	packageErr, ok := recipe.AsPackageError(err)
	if !ok || packageErr.Code != recipe.ErrorActivationFailed {
		t.Fatalf("Activate() error = %#v", err)
	}
	if !bytes.Equal(previous, mustReadFile(t, configPath)) {
		t.Fatal("Envoy readiness failure did not restore the previous runtime")
	}
	if _, state, statusErr := store.ActivationStatus(); statusErr != nil || state != recipe.ActivationNone {
		t.Fatalf("ActivationStatus() state=%q err=%v", state, statusErr)
	}
}

func TestRecipeActivatorTargetTimeoutLeavesIndependentRollbackBudget(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	previous := mustReadFile(t, configPath)
	var envoyCalls atomic.Int32
	var rollbackSawLiveContext atomic.Bool
	activator := NewRecipeActivator(RecipeActivatorOptions{
		Store:          store,
		ConfigPath:     configPath,
		ConfigDir:      filepath.Dir(configPath),
		AttemptTimeout: 500 * time.Millisecond,
		RealizeConfig:  func(raw []byte, _ string) ([]byte, error) { return raw, nil },
		ApplyRuntime:   func(path, _ string) (string, error) { return path, nil },
		VerifyRuntime: func(ctx context.Context, _ string) error {
			if envoyCalls.Load() > 0 && ctx.Err() == nil {
				rollbackSawLiveContext.Store(true)
			}
			return ctx.Err()
		},
		VerifyEnvoy: func(ctx context.Context) error {
			if envoyCalls.Add(1) == 1 {
				<-ctx.Done()
				return ctx.Err()
			}
			return nil
		},
	})
	_, err := activator.Activate(context.Background(), activationRequest(summary))
	packageErr, ok := recipe.AsPackageError(err)
	if !ok || packageErr.Code != recipe.ErrorActivationFailed {
		t.Fatalf("Activate() error = %#v", err)
	}
	if !rollbackSawLiveContext.Load() || !bytes.Equal(previous, mustReadFile(t, configPath)) {
		t.Fatal("target timeout consumed rollback context budget")
	}
	if _, state, statusErr := store.ActivationStatus(); statusErr != nil || state != recipe.ActivationNone {
		t.Fatalf("ActivationStatus() state=%q err=%v", state, statusErr)
	}
}

func TestRecipeActivatorFailureRollbackDetachesFromCanceledParent(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	previous := mustReadFile(t, configPath)
	transaction, err := store.BeginActivation(summary.RecipeDigest, previous)
	if err != nil {
		t.Fatal(err)
	}
	const attemptTimeout = 5 * time.Second
	rollbackVerified := false
	activator := NewRecipeActivator(RecipeActivatorOptions{
		Store:          store,
		ConfigPath:     configPath,
		ConfigDir:      filepath.Dir(configPath),
		AttemptTimeout: attemptTimeout,
		ApplyRuntime:   func(path, _ string) (string, error) { return path, nil },
		VerifyRuntime: func(ctx context.Context, _ string) error {
			assertLiveBoundedRollbackContext(t, ctx, attemptTimeout)
			rollbackVerified = true
			return nil
		},
		VerifyEnvoy: func(context.Context) error { return nil },
	})
	parent, cancel := context.WithCancel(context.Background())
	cancel()

	err = activator.failAndRollback(parent, transaction, previous, errors.New("activation failed"))
	packageErr, ok := recipe.AsPackageError(err)
	if !ok || packageErr.Code != recipe.ErrorActivationFailed {
		t.Fatalf("failAndRollback() error = %#v", err)
	}
	if !rollbackVerified {
		t.Fatal("rollback did not verify the restored runtime")
	}
	if _, state, statusErr := store.ActivationStatus(); statusErr != nil || state != recipe.ActivationNone {
		t.Fatalf("ActivationStatus() state=%q err=%v", state, statusErr)
	}
}

func TestRecipeActivatorRecoveryRollbackDetachesFromExpiredParent(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	previous := mustReadFile(t, configPath)
	transaction, err := store.BeginActivation(summary.RecipeDigest, previous)
	if err != nil {
		t.Fatal(err)
	}
	const attemptTimeout = 5 * time.Second
	rollbackVerified := false
	activator := NewRecipeActivator(RecipeActivatorOptions{
		Store:          store,
		ConfigPath:     configPath,
		ConfigDir:      filepath.Dir(configPath),
		AttemptTimeout: attemptTimeout,
		ApplyRuntime:   func(path, _ string) (string, error) { return path, nil },
		VerifyRuntime: func(ctx context.Context, _ string) error {
			assertLiveBoundedRollbackContext(t, ctx, attemptTimeout)
			rollbackVerified = true
			return nil
		},
		VerifyEnvoy: func(context.Context) error { return nil },
	})
	parent, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()

	if err := activator.recoverRollback(parent, transaction, previous); err != nil {
		t.Fatalf("recoverRollback(): %v", err)
	}
	if !rollbackVerified {
		t.Fatal("recovery rollback did not verify the restored runtime")
	}
	if _, state, statusErr := store.ActivationStatus(); statusErr != nil || state != recipe.ActivationNone {
		t.Fatalf("ActivationStatus() state=%q err=%v", state, statusErr)
	}
}

func assertLiveBoundedRollbackContext(t *testing.T, ctx context.Context, timeout time.Duration) {
	t.Helper()
	if err := ctx.Err(); err != nil {
		t.Fatalf("rollback context inherited parent cancellation: %v", err)
	}
	deadline, ok := ctx.Deadline()
	if !ok {
		t.Fatal("rollback context has no deadline")
	}
	if remaining := time.Until(deadline); remaining <= 0 || remaining > timeout {
		t.Fatalf("rollback context deadline remaining = %v, want (0, %v]", remaining, timeout)
	}
}

func TestRecipeActivatorDeactivateRestoresOriginalSourceAcrossPackageSwitches(t *testing.T) {
	store, first, configPath := importedActivationFixture(t, "accuracy")
	original := mustReadFile(t, configPath)
	activator := NewRecipeActivator(RecipeActivatorOptions{
		Store:         store,
		ConfigPath:    configPath,
		ConfigDir:     filepath.Dir(configPath),
		RealizeConfig: func(raw []byte, _ string) ([]byte, error) { return raw, nil },
		ApplyRuntime:  func(path, _ string) (string, error) { return path, nil },
		VerifyRuntime: func(context.Context, string) error { return nil },
		VerifyEnvoy:   func(context.Context) error { return nil },
	})
	mustActivateRecipe(t, activator, first)
	second := importRecipeIntoStore(t, store, "privacy")
	mustActivateRecipe(t, activator, second)
	mustDeactivateRecipe(t, activator, second.RecipeDigest)
	if !bytes.Equal(original, mustReadFile(t, configPath)) {
		t.Fatal("deactivation restored the previous package instead of the original source baseline")
	}
	assertActivationNone(t, store)
	mustDeactivateRecipe(t, activator, "")
	editedSource := append(append([]byte(nil), original...), []byte("\n# ordinary source edit\n")...)
	mustWriteActivationConfig(t, configPath, editedSource)
	mustActivateRecipe(t, activator, first)
	mustDeactivateRecipe(t, activator, first.RecipeDigest)
	if !bytes.Equal(editedSource, mustReadFile(t, configPath)) {
		t.Fatal("second deactivation restored a stale source baseline")
	}
}

func mustActivateRecipe(t *testing.T, activator *RecipeActivator, summary recipe.PackageSummary) {
	t.Helper()
	if _, err := activator.Activate(context.Background(), activationRequest(summary)); err != nil {
		t.Fatalf("Activate(): %v", err)
	}
}

func mustDeactivateRecipe(t *testing.T, activator *RecipeActivator, previousDigest string) {
	t.Helper()
	result, err := activator.Deactivate(context.Background())
	if err != nil || result.Status != "inactive" || result.PreviousRecipeDigest != previousDigest {
		t.Fatalf("Deactivate() = %#v, %v", result, err)
	}
}

func assertActivationNone(t *testing.T, store *recipe.Store) {
	t.Helper()
	if _, state, err := store.ActivationStatus(); err != nil || state != recipe.ActivationNone {
		t.Fatalf("ActivationStatus() state=%q err=%v", state, err)
	}
}

func mustWriteActivationConfig(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := writeActivationConfig(path, data); err != nil {
		t.Fatal(err)
	}
}

func TestRecipeActivatorRecoversInconsistentJournalThenRepairs(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	previous := mustReadFile(t, configPath)
	transaction, err := store.BeginActivation(summary.RecipeDigest, previous)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.MarkTransactionInconsistent(transaction); err != nil {
		t.Fatal(err)
	}
	var applyCalls atomic.Int32
	activator := NewRecipeActivator(RecipeActivatorOptions{
		Store:         store,
		ConfigPath:    configPath,
		ConfigDir:     filepath.Dir(configPath),
		RealizeConfig: func(raw []byte, _ string) ([]byte, error) { return raw, nil },
		ApplyRuntime: func(path, _ string) (string, error) {
			applyCalls.Add(1)
			return path, nil
		},
		VerifyRuntime: func(context.Context, string) error { return nil },
		VerifyEnvoy:   func(context.Context) error { return nil },
	})
	if _, err := activator.Activate(context.Background(), activationRequest(summary)); err != nil {
		t.Fatalf("repair Activate(): %v", err)
	}
	if applyCalls.Load() != 2 {
		t.Fatalf("apply calls = %d, want recovery and target", applyCalls.Load())
	}
	if _, state, err := store.ActivationStatus(); err != nil || state != recipe.ActivationActive {
		t.Fatalf("repaired state = %q, %v", state, err)
	}
}

func TestHotSwitchCommittingRecoveryDoesNotRequireTopologyJournal(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	previous := mustReadFile(t, configPath)
	_, targetConfig, err := store.ActivationTarget(summary.RecipeDigest, true)
	if err != nil {
		t.Fatal(err)
	}
	transaction, err := store.BeginActivation(summary.RecipeDigest, previous)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeActivationConfig(configPath, targetConfig); err != nil {
		t.Fatal(err)
	}
	if err := store.PrepareActivationCommit(transaction, recipe.ActivePointer{
		RecipeDigest:         summary.RecipeDigest,
		ConfigDigest:         summary.ConfigDigest,
		RealizedConfigDigest: activationDigest(targetConfig),
	}); err != nil {
		t.Fatal(err)
	}
	activator := NewRecipeActivator(RecipeActivatorOptions{
		Store:         store,
		ConfigPath:    configPath,
		ConfigDir:     filepath.Dir(configPath),
		VerifyRuntime: func(context.Context, string) error { return nil },
		VerifyEnvoy:   func(context.Context) error { return nil },
	})
	if err := activator.Recover(context.Background()); err != nil {
		t.Fatalf("Recover(): %v", err)
	}
	if _, state, err := store.ActivationStatus(); err != nil || state != recipe.ActivationActive {
		t.Fatalf("activation status=%q err=%v", state, err)
	}
}

func TestRecipeActivatorRollbackFailureLeavesInconsistentJournal(t *testing.T) {
	store, summary, configPath := importedActivationFixture(t, "accuracy")
	activator := NewRecipeActivator(RecipeActivatorOptions{
		Store:         store,
		ConfigPath:    configPath,
		ConfigDir:     filepath.Dir(configPath),
		RealizeConfig: func(raw []byte, _ string) ([]byte, error) { return raw, nil },
		ApplyRuntime:  func(path, _ string) (string, error) { return path, nil },
		VerifyRuntime: func(context.Context, string) error { return errors.New("router remains pending") },
		VerifyEnvoy:   func(context.Context) error { return nil },
	})
	_, err := activator.Activate(context.Background(), activationRequest(summary))
	packageErr, ok := recipe.AsPackageError(err)
	if !ok || packageErr.Code != recipe.ErrorActivationRollback {
		t.Fatalf("Activate() error = %#v", err)
	}
	if _, state, _ := store.ActivationStatus(); state != recipe.ActivationInconsistent {
		t.Fatalf("activation state = %q, want inconsistent", state)
	}
}

func TestRouterActivationVerifierPollsPendingUntilExactActiveHash(t *testing.T) {
	expected := strings.Repeat("a", 64)
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			_, _ = w.Write([]byte(`{"runtime_hash":"` + expected + `","active_hash":"old","status":"pending"}`))
			return
		}
		_, _ = w.Write([]byte(`{"runtime_hash":"` + expected + `","active_hash":"` + expected + `","status":"active"}`))
	}))
	defer server.Close()
	verify := newRouterActivationVerifier(server.URL, server.Client())
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := verify(ctx, "sha256:"+expected); err != nil {
		t.Fatalf("verify(): %v", err)
	}
	if calls.Load() < 2 {
		t.Fatalf("calls = %d", calls.Load())
	}
}

func TestRouterActivationVerifierStopsOnContextDeadline(t *testing.T) {
	expected := strings.Repeat("c", 64)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"runtime_hash":"old","active_hash":"old","status":"pending"}`))
	}))
	defer server.Close()
	verify := newRouterActivationVerifier(server.URL, server.Client())
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if err := verify(ctx, "sha256:"+expected); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("verify() error = %v", err)
	}
}

func TestEnvoyActivationVerifierPollsUntilReady(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()
	verify := newEnvoyActivationVerifier(server.URL+"/ready", server.Client())
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := verify(ctx); err != nil {
		t.Fatalf("verify(): %v", err)
	}
	if calls.Load() < 2 {
		t.Fatalf("calls = %d", calls.Load())
	}
}

func TestRecipeRuntimeMaterializeArgsForceSideEffectFreePackageMode(t *testing.T) {
	args := recipeRuntimeMaterializeArgs("/state/raw.yaml", "/state/realized.yaml")
	if !slices.Contains(args, "--package-activation") {
		t.Fatalf("runtime materialize args = %#v", args)
	}
}

func importedActivationFixture(t *testing.T, recipeName string) (*recipe.Store, recipe.PackageSummary, string) {
	t.Helper()
	directory := filepath.Join("..", "..", "..", "config", "recipes", recipeName)
	archive := zipRecipeDirectory(t, directory)
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(archive) }))
	t.Cleanup(server.Close)
	root := t.TempDir()
	configPath := filepath.Join(root, "runtime-config.yaml")
	activeConfig, err := os.ReadFile(filepath.Join(directory, "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if writeErr := os.WriteFile(configPath, activeConfig, 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	t.Setenv("VLLM_SR_RECIPE_ENV_ALLOWLIST", "OPENROUTER_API_KEY")
	t.Setenv("OPENROUTER_API_KEY", "test-only")
	t.Setenv(managedStorageBackends, "")
	store := recipe.NewStore(recipe.StoreOptions{Root: filepath.Join(root, "recipe-store"), ConfigPath: configPath, HTTPClient: server.Client()})
	summary, _, err := store.Import(context.Background(), recipe.ImportRequest{URL: server.URL + "/recipe.zip"})
	if err != nil {
		t.Fatalf("Import(): %v", err)
	}
	return store, summary, configPath
}

func activationRequest(summary recipe.PackageSummary) recipe.ActivateRequest {
	return recipe.ActivateRequest{RecipeDigest: summary.RecipeDigest, AcknowledgeWarnings: true}
}

func withManagementBearer(config []byte) []byte {
	return []byte(strings.Replace(string(config), "global:\n", "global:\n  services:\n    management_api:\n      auth:\n        mode: bearer\n        tokens:\n          - env: TEST_MANAGEMENT_TOKEN\n            role: admin\n", 1))
}

func withManagedPostgresReplay(config []byte) []byte {
	return []byte(strings.Replace(string(config), "global:\n", "global:\n  services:\n    router_replay:\n      enabled: true\n      store_backend: postgres\n      postgres:\n        host: postgres\n        port: 5432\n        database: vsr\n        user: router\n        password: test-only\n        ssl_mode: disable\n        table_name: router_replay\n", 1))
}

func importRecipeIntoStore(t *testing.T, store *recipe.Store, name string) recipe.PackageSummary {
	t.Helper()
	directory := filepath.Join("..", "..", "..", "config", "recipes", name)
	archive := zipRecipeDirectory(t, directory)
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(archive) }))
	t.Cleanup(server.Close)
	summary, _, err := store.Import(context.Background(), recipe.ImportRequest{URL: server.URL + "/recipe.zip"})
	if err != nil {
		t.Fatalf("Import(%s): %v", name, err)
	}
	return summary
}

func mustReadFile(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func zipRecipeDirectory(t *testing.T, directory string) []byte {
	t.Helper()
	var buffer bytes.Buffer
	writer := zip.NewWriter(&buffer)
	for _, name := range []string{"metadata.yaml", "config.yaml", "probes.yaml", "recipe.dsl", "README.md"} {
		input, err := os.Open(filepath.Join(directory, name))
		if err != nil {
			t.Fatal(err)
		}
		output, err := writer.Create(name)
		if err == nil {
			_, err = io.Copy(output, input)
		}
		_ = input.Close()
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return buffer.Bytes()
}
