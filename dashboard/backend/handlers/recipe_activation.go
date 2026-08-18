package handlers

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
)

const (
	maxActivationConfigBytes = 16 << 20
	routerActivationTimeout  = 300 * time.Second
	routerActiveCheckTimeout = 5 * time.Second
)

type RecipeActivatorOptions struct {
	Store          *recipe.Store
	ConfigPath     string
	ConfigDir      string
	RouterAPIURL   string
	HTTPClient     *http.Client
	RealizeConfig  func([]byte, string) ([]byte, error)
	ApplyRuntime   func(string, string) (string, error)
	VerifyRuntime  func(context.Context, string) error
	VerifyEnvoy    func(context.Context) error
	Topology       runtimeTopologyReconciler
	AttemptTimeout time.Duration
}

// RecipeActivator coordinates a rollback-capable activation transaction under
// the same deploy lock used by ordinary Dashboard config deployment.
type RecipeActivator struct {
	store          *recipe.Store
	configPath     string
	configDir      string
	applyRuntime   func(string, string) (string, error)
	realizeConfig  func([]byte, string) ([]byte, error)
	verifyRuntime  func(context.Context, string) error
	verifyEnvoy    func(context.Context) error
	topology       runtimeTopologyReconciler
	attemptTimeout time.Duration
}

type activationTopologyExecution struct {
	state      *recipe.ActivationTopologyState
	credential string
}

func NewRecipeActivator(options RecipeActivatorOptions) *RecipeActivator {
	activator := &RecipeActivator{
		store:         options.Store,
		configPath:    filepath.Clean(options.ConfigPath),
		configDir:     filepath.Clean(options.ConfigDir),
		applyRuntime:  options.ApplyRuntime,
		realizeConfig: options.RealizeConfig,
	}
	activator.attemptTimeout = options.AttemptTimeout
	if activator.attemptTimeout <= 0 {
		activator.attemptTimeout = routerActivationTimeout
	}
	if activator.applyRuntime == nil {
		activator.applyRuntime = applyRecipeRuntime
	}
	if activator.realizeConfig == nil {
		activator.realizeConfig = realizeRecipeRuntimeConfig
	}
	activator.verifyRuntime = options.VerifyRuntime
	if activator.verifyRuntime == nil {
		activator.verifyRuntime = newRouterActivationVerifier(options.RouterAPIURL, options.HTTPClient, activator.store)
	}
	activator.verifyEnvoy = options.VerifyEnvoy
	if activator.verifyEnvoy == nil {
		activator.verifyEnvoy = newEnvoyActivationVerifier(managedEnvoyReadyURL(), options.HTTPClient)
	}
	activator.topology = options.Topology
	if activator.topology == nil {
		activator.topology = newManagedRuntimeTopologyReconciler(options.Store, options.ConfigPath)
	}
	return activator
}

func (a *RecipeActivator) Activate(ctx context.Context, request recipe.ActivateRequest) (recipe.ActivateResult, error) {
	release, err := beginRecipeActivationMutation(a.store.Root())
	if err != nil {
		return recipe.ActivateResult{}, packageRuntimeConfigMutationError(err)
	}
	defer release()
	operationContext, cancel := context.WithTimeout(context.WithoutCancel(ctx), 2*a.attemptTimeout+2*time.Minute)
	defer cancel()
	if recoveryErr := a.recoverLocked(operationContext); recoveryErr != nil {
		return recipe.ActivateResult{}, recoveryErr
	}
	return a.activateLocked(operationContext, request)
}

func (a *RecipeActivator) activateLocked(ctx context.Context, request recipe.ActivateRequest) (recipe.ActivateResult, error) {
	target, _, err := a.store.ActivationTarget(request.RecipeDigest, request.AcknowledgeWarnings)
	if err != nil {
		return recipe.ActivateResult{}, err
	}
	active, state, _ := a.store.ActivationStatus()
	if state == recipe.ActivationActive && active.RecipeDigest == target.RecipeDigest {
		checkContext, checkCancel := context.WithTimeout(ctx, routerActiveCheckTimeout)
		activeCheckErr := a.verifyActivatedRuntime(checkContext, active.RealizedConfigDigest)
		checkCancel()
		if activeCheckErr == nil {
			return recipe.ActivateResult{Status: "active", RecipeDigest: target.RecipeDigest}, nil
		}
	}
	target, plan, previousConfig, realizedConfig, err := a.prepareActivation(ctx, request)
	if err != nil {
		return recipe.ActivateResult{}, err
	}
	if state == recipe.ActivationNone {
		if baselineErr := a.store.RefreshSourceBaseline(previousConfig); baselineErr != nil {
			return recipe.ActivateResult{}, activationFailed("Source Recipe baseline could not be preserved.", baselineErr)
		}
	}
	if confirmationErr := requireActivationConfirmation(request, plan); confirmationErr != nil {
		return recipe.ActivateResult{}, confirmationErr
	}
	transaction, err := a.store.BeginActivation(target.RecipeDigest, previousConfig)
	if err != nil {
		return recipe.ActivateResult{}, activationFailed("Recipe activation transaction could not be started.", err)
	}
	return a.executeActivation(ctx, target, plan, transaction, previousConfig, realizedConfig)
}

func (a *RecipeActivator) executeActivation(ctx context.Context, target recipe.PackageSummary, plan recipe.ActivationPlan, transaction recipe.ActivationTransaction, previousConfig, realizedConfig []byte) (recipe.ActivateResult, error) {
	topology, err := a.prepareActivationTopology(ctx, plan, transaction)
	if err != nil {
		return recipe.ActivateResult{}, a.failAndRollback(ctx, transaction, previousConfig, err)
	}
	if publishConfigErr := a.publishActivationConfig(ctx, realizedConfig, topology); publishConfigErr != nil {
		return recipe.ActivateResult{}, a.failAndRollback(ctx, transaction, previousConfig, publishConfigErr)
	}
	realizedConfig, err = readActivationConfig(a.configPath)
	if err != nil {
		return recipe.ActivateResult{}, a.failAndRollback(ctx, transaction, previousConfig, err)
	}
	realizedDigest := activationDigest(realizedConfig)
	if verificationErr := a.verifyRuntimeAttempt(ctx, realizedDigest); verificationErr != nil {
		return recipe.ActivateResult{}, a.failAndRollback(ctx, transaction, previousConfig, verificationErr)
	}
	pointer := recipe.ActivePointer{
		RecipeDigest:         target.RecipeDigest,
		ConfigDigest:         target.ConfigDigest,
		RealizedConfigDigest: realizedDigest,
	}
	if commitErr := a.store.PrepareActivationCommit(transaction, pointer); commitErr != nil {
		return recipe.ActivateResult{}, a.failAndRollback(ctx, transaction, previousConfig, commitErr)
	}
	if commitErr := a.commitActivation(ctx, transaction, topology.state); commitErr != nil {
		return recipe.ActivateResult{}, activationCommitIncomplete(commitErr)
	}
	return recipe.ActivateResult{
		Status:               "active",
		RecipeDigest:         target.RecipeDigest,
		PreviousRecipeDigest: transaction.PreviousRecipeDigest,
		PlanDigest:           plan.PlanDigest,
		Mode:                 plan.Mode,
	}, nil
}

func (a *RecipeActivator) Deactivate(ctx context.Context, requests ...recipe.DeactivateRequest) (recipe.DeactivateResult, error) {
	release, err := beginRecipeActivationMutation(a.store.Root())
	if err != nil {
		return recipe.DeactivateResult{}, packageRuntimeConfigMutationError(err)
	}
	defer release()
	operationContext, cancel := context.WithTimeout(context.WithoutCancel(ctx), 2*a.attemptTimeout+2*time.Minute)
	defer cancel()
	if recoveryErr := a.recoverLocked(operationContext); recoveryErr != nil {
		return recipe.DeactivateResult{}, recoveryErr
	}
	return a.deactivateLocked(operationContext, requests...)
}

func (a *RecipeActivator) deactivateLocked(ctx context.Context, requests ...recipe.DeactivateRequest) (recipe.DeactivateResult, error) {
	active, plan, previousConfig, baseline, err := a.prepareDeactivation(ctx)
	if err != nil {
		return recipe.DeactivateResult{}, err
	}
	if active.RecipeDigest == "" {
		return recipe.DeactivateResult{Status: "inactive"}, nil
	}
	request := recipe.DeactivateRequest{}
	if len(requests) > 0 {
		request = requests[0]
	}
	if confirmationErr := requireDeactivationConfirmation(request, plan); confirmationErr != nil {
		return recipe.DeactivateResult{}, confirmationErr
	}
	transaction, err := a.store.BeginDeactivation(previousConfig)
	if err != nil {
		return recipe.DeactivateResult{}, activationFailed("Recipe deactivation transaction could not be started.", err)
	}
	return a.executeDeactivation(ctx, active, plan, transaction, previousConfig, baseline)
}

func (a *RecipeActivator) executeDeactivation(ctx context.Context, active recipe.ActivePointer, plan recipe.ActivationPlan, transaction recipe.ActivationTransaction, previousConfig, baseline []byte) (recipe.DeactivateResult, error) {
	topology, err := a.prepareActivationTopology(ctx, plan, transaction)
	if err != nil {
		return recipe.DeactivateResult{}, a.failDeactivationAndRollback(ctx, transaction, previousConfig, err)
	}
	if publishConfigErr := a.publishActivationConfig(ctx, baseline, topology); publishConfigErr != nil {
		return recipe.DeactivateResult{}, a.failDeactivationAndRollback(ctx, transaction, previousConfig, publishConfigErr)
	}
	restored, err := readActivationConfig(a.configPath)
	if err == nil {
		err = a.verifyRuntimeAttempt(ctx, activationDigest(restored))
	}
	if err != nil {
		return recipe.DeactivateResult{}, a.failDeactivationAndRollback(ctx, transaction, previousConfig, err)
	}
	if commitErr := a.store.PrepareDeactivationCommit(transaction); commitErr != nil {
		return recipe.DeactivateResult{}, a.failDeactivationAndRollback(ctx, transaction, previousConfig, commitErr)
	}
	if commitErr := a.commitDeactivation(ctx, transaction, topology.state); commitErr != nil {
		return recipe.DeactivateResult{}, activationCommitIncomplete(commitErr)
	}
	return recipe.DeactivateResult{Status: "inactive", PreviousRecipeDigest: active.RecipeDigest, PlanDigest: plan.PlanDigest, Mode: plan.Mode}, nil
}

func (a *RecipeActivator) prepareActivationTopology(ctx context.Context, plan recipe.ActivationPlan, transaction recipe.ActivationTransaction) (activationTopologyExecution, error) {
	if plan.Mode != recipe.ActivationModeStackRecreation {
		return activationTopologyExecution{}, nil
	}
	inventory, err := a.topology.Inventory(ctx)
	if err == nil && !activationInventoryMatchesPlan(inventory, plan) {
		err = errors.New("managed storage topology changed after preview")
	}
	if err != nil {
		return activationTopologyExecution{}, err
	}
	state := topologyStateForPlan(plan, transaction, inventory)
	if topologyStoreErr := a.store.WriteActivationTopology(transaction, state); topologyStoreErr != nil {
		return activationTopologyExecution{}, topologyStoreErr
	}
	execution := activationTopologyExecution{state: &state}
	if state.CredentialEnv != "" {
		execution.credential, err = a.store.EnsureManagementCredential()
	}
	return execution, err
}

func (a *RecipeActivator) publishActivationConfig(ctx context.Context, config []byte, topology activationTopologyExecution) error {
	if err := writeActivationConfig(a.configPath, config); err != nil {
		return err
	}
	if topology.state != nil {
		if err := refreshManagedSplitEnvoyConfig(a.configPath); err != nil {
			return err
		}
		return a.topology.Apply(ctx, *topology.state, topology.credential)
	}
	realizedPath, err := a.applyRuntime(a.configPath, a.configDir)
	if err != nil {
		return err
	}
	if filepath.Clean(realizedPath) != a.configPath {
		return errors.New("runtime config synchronization selected a different active path")
	}
	return nil
}

func (a *RecipeActivator) commitActivation(ctx context.Context, transaction recipe.ActivationTransaction, topology *recipe.ActivationTopologyState) error {
	if topology != nil {
		if err := a.topology.Commit(ctx, *topology); err != nil {
			return err
		}
	}
	return a.store.FinalizeActivationCommit(transaction)
}

func (a *RecipeActivator) commitDeactivation(ctx context.Context, transaction recipe.ActivationTransaction, topology *recipe.ActivationTopologyState) error {
	if topology != nil {
		if err := a.topology.Commit(ctx, *topology); err != nil {
			return err
		}
	}
	return a.store.FinalizeDeactivationCommit(transaction)
}

func (a *RecipeActivator) verifyActivatedRuntime(ctx context.Context, digest string) error {
	if err := a.verifyRuntime(ctx, digest); err != nil {
		return err
	}
	return a.verifyEnvoy(ctx)
}

func (a *RecipeActivator) verifyRuntimeAttempt(parent context.Context, digest string) error {
	attempt, cancel := context.WithTimeout(parent, a.attemptTimeout)
	defer cancel()
	return a.verifyActivatedRuntime(attempt, digest)
}

func (a *RecipeActivator) Recover(ctx context.Context) error {
	release, err := beginRecipeActivationMutation(a.store.Root())
	if err != nil {
		return packageRuntimeConfigMutationError(err)
	}
	defer release()
	operationContext, cancel := context.WithTimeout(context.WithoutCancel(ctx), a.attemptTimeout+2*time.Minute)
	defer cancel()
	return a.recoverLocked(operationContext)
}

func (a *RecipeActivator) recoverLocked(ctx context.Context) error {
	transaction, previousConfig, err := a.store.Transaction()
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return activationRollbackFailed(err)
	}
	switch transaction.State {
	case recipe.ActivationRollbackFinalizing:
		return a.recoverFinalizingRollback(transaction)
	case "committing", recipe.ActivationFinalizing:
		return a.recoverCommit(ctx, transaction)
	default:
		return a.recoverRollback(ctx, transaction, previousConfig)
	}
}

func (a *RecipeActivator) recoverFinalizingRollback(transaction recipe.ActivationTransaction) error {
	currentConfig, err := readActivationConfig(a.configPath)
	if err == nil && activationDigest(currentConfig) != transaction.PreviousConfigDigest {
		err = errors.New("rollback-finalizing runtime config does not match the journal")
	}
	if err != nil {
		return activationRollbackFailed(err)
	}
	if err := a.store.CompleteRollback(transaction); err != nil {
		return activationRollbackFailed(err)
	}
	return nil
}

func (a *RecipeActivator) recoverCommit(ctx context.Context, transaction recipe.ActivationTransaction) error {
	topologyState, err := a.recoveryCommitTopology(transaction)
	if err != nil {
		return activationCommitIncomplete(err)
	}
	expectedDigest, isDeactivation, err := a.recoveryCommitDigest(transaction)
	if err != nil {
		return activationCommitIncomplete(err)
	}
	if expectedDigest != transaction.CommitConfigDigest {
		return activationCommitIncomplete(errors.New("committing runtime config digest does not match the journal"))
	}
	if verificationErr := a.verifyActivatedRuntime(ctx, expectedDigest); verificationErr != nil {
		return activationCommitIncomplete(verificationErr)
	}
	if topologyState != nil {
		if commitErr := a.topology.Commit(ctx, *topologyState); commitErr != nil {
			return activationCommitIncomplete(commitErr)
		}
	}
	if isDeactivation {
		err = a.store.FinalizeDeactivationCommit(transaction)
	} else {
		err = a.store.FinalizeActivationCommit(transaction)
	}
	if err != nil {
		return activationCommitIncomplete(err)
	}
	return nil
}

func (a *RecipeActivator) recoveryCommitTopology(transaction recipe.ActivationTransaction) (*recipe.ActivationTopologyState, error) {
	if transaction.State != "committing" {
		return nil, nil
	}
	topology, err := a.store.ActivationTopology(transaction)
	if err != nil {
		if transaction.TopologyMode == recipe.ActivationTopologyManaged || !errors.Is(err, os.ErrNotExist) {
			return nil, err
		}
		return nil, nil
	}
	if transaction.TopologyMode == recipe.ActivationTopologyNone {
		return nil, errors.New("hot-switch transaction unexpectedly contains a topology journal")
	}
	return &topology, nil
}

func (a *RecipeActivator) recoveryCommitDigest(transaction recipe.ActivationTransaction) (string, bool, error) {
	isDeactivation := transaction.Operation == recipe.ActivationOperationDeactivate
	if isDeactivation {
		if _, err := a.store.ReadActivePointer(); !errors.Is(err, os.ErrNotExist) {
			return "", true, errors.New("committing deactivation retained an active pointer")
		}
		currentConfig, err := readActivationConfig(a.configPath)
		if err != nil {
			return "", true, err
		}
		return activationDigest(currentConfig), true, nil
	}
	active, err := a.store.ReadActivePointer()
	if err == nil && active.RecipeDigest != transaction.TargetRecipeDigest {
		err = errors.New("committing activation pointer does not match the target")
	}
	if err != nil {
		return "", false, err
	}
	return active.RealizedConfigDigest, false, nil
}

func (a *RecipeActivator) recoverRollback(ctx context.Context, transaction recipe.ActivationTransaction, previousConfig []byte) error {
	rollbackContext, cancel := a.independentRollbackContext(ctx)
	defer cancel()
	if err := a.rollback(rollbackContext, transaction, previousConfig); err != nil {
		_ = a.store.MarkTransactionInconsistent(transaction)
		return activationRollbackFailed(err)
	}
	return nil
}

func (a *RecipeActivator) failAndRollback(ctx context.Context, transaction recipe.ActivationTransaction, previousConfig []byte, activationErr error) error {
	rollbackContext, cancel := a.independentRollbackContext(ctx)
	defer cancel()
	if rollbackErr := a.rollback(rollbackContext, transaction, previousConfig); rollbackErr != nil {
		_ = a.store.MarkTransactionInconsistent(transaction)
		return activationRollbackFailed(errors.Join(activationErr, rollbackErr))
	}
	return activationFailed("Recipe activation failed; the previous runtime config was restored.", activationErr)
}

func (a *RecipeActivator) failDeactivationAndRollback(ctx context.Context, transaction recipe.ActivationTransaction, previousConfig []byte, deactivationErr error) error {
	rollbackContext, cancel := a.independentRollbackContext(ctx)
	defer cancel()
	if rollbackErr := a.rollback(rollbackContext, transaction, previousConfig); rollbackErr != nil {
		_ = a.store.MarkTransactionInconsistent(transaction)
		return activationRollbackFailed(errors.Join(deactivationErr, rollbackErr))
	}
	return activationFailed("Recipe deactivation failed; the active package runtime was restored.", deactivationErr)
}

func (a *RecipeActivator) independentRollbackContext(parent context.Context) (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.WithoutCancel(parent), a.attemptTimeout)
}

func (a *RecipeActivator) rollback(ctx context.Context, transaction recipe.ActivationTransaction, previousConfig []byte) error {
	// WriteActivationTopology durably upgrades the transaction from a hot switch
	// to managed topology, but the BeginActivation value held by this process is
	// intentionally immutable. Always make rollback decisions from the durable
	// journal so an apply or verification failure after topology persistence can
	// still restore the previous runtime.
	current, _, err := a.store.Transaction()
	if err != nil {
		return err
	}
	if current.ID != transaction.ID {
		return errors.New("activation transaction changed unexpectedly")
	}
	topologyState, hasTopology, err := a.rollbackTopology(current)
	if err != nil {
		return err
	}
	if err := writeActivationConfig(a.configPath, previousConfig); err != nil {
		return err
	}
	if hasTopology {
		return a.rollbackManagedRuntime(ctx, current, topologyState)
	}
	return a.rollbackHotSwitchRuntime(ctx, current)
}

func (a *RecipeActivator) rollbackTopology(transaction recipe.ActivationTransaction) (recipe.ActivationTopologyState, bool, error) {
	topologyState, topologyErr := a.store.ActivationTopology(transaction)
	// WriteActivationTopology persists the validated topology before upgrading
	// the transaction journal. A process or filesystem failure between those
	// writes therefore leaves a pending/none transaction whose durable topology
	// is the authoritative evidence that managed rollback is required. Preserve
	// that classification if an initial rollback attempt marked the transaction
	// inconsistent so recovery remains retryable.
	topologyWriteCrashWindow := topologyErr == nil &&
		transaction.TopologyMode == recipe.ActivationTopologyNone &&
		(transaction.State == "pending" || transaction.State == recipe.ActivationInconsistent)
	if topologyErr == nil && transaction.TopologyMode != recipe.ActivationTopologyManaged && !topologyWriteCrashWindow {
		return recipe.ActivationTopologyState{}, false, errors.New("hot-switch transaction unexpectedly contains a topology journal")
	}
	if errors.Is(topologyErr, os.ErrNotExist) && transaction.TopologyMode == recipe.ActivationTopologyManaged {
		return recipe.ActivationTopologyState{}, false, errors.New("managed activation transaction is missing its topology journal")
	}
	if topologyErr != nil && !errors.Is(topologyErr, os.ErrNotExist) {
		return recipe.ActivationTopologyState{}, false, topologyErr
	}
	return topologyState, topologyErr == nil, nil
}

func (a *RecipeActivator) rollbackManagedRuntime(ctx context.Context, transaction recipe.ActivationTransaction, topology recipe.ActivationTopologyState) error {
	if err := refreshManagedSplitEnvoyConfig(a.configPath); err != nil {
		return err
	}
	if err := a.topology.Rollback(ctx, topology); err != nil {
		return err
	}
	return a.verifyAndCompleteRollback(ctx, transaction)
}

func (a *RecipeActivator) rollbackHotSwitchRuntime(ctx context.Context, transaction recipe.ActivationTransaction) error {
	realizedPath, err := a.applyRuntime(a.configPath, a.configDir)
	if err != nil {
		return err
	}
	if filepath.Clean(realizedPath) != a.configPath {
		return errors.New("rollback selected a different active runtime config path")
	}
	return a.verifyAndCompleteRollback(ctx, transaction)
}

func (a *RecipeActivator) verifyAndCompleteRollback(ctx context.Context, transaction recipe.ActivationTransaction) error {
	restoredConfig, err := readActivationConfig(a.configPath)
	if err != nil {
		return err
	}
	if err := a.verifyActivatedRuntime(ctx, activationDigest(restoredConfig)); err != nil {
		return err
	}
	return a.store.CompleteRollback(transaction)
}

func applyRecipeRuntime(configPath string, configDir string) (string, error) {
	effectiveConfigPath, err := syncRuntimeConfigForCurrentRuntime(configPath)
	if err != nil {
		return "", err
	}
	if isRunningInContainer() && isManagedContainerConfigPath(configPath) && managedRuntimeUsesSplitContainers() {
		if err := restartSetupManagedServices(effectiveConfigPath); err != nil {
			return "", err
		}
		return effectiveConfigPath, nil
	}
	if err := propagateConfigToRuntime(configPath, configDir); err != nil {
		return "", err
	}
	return effectiveConfigPath, nil
}

func realizeRecipeRuntimeConfig(raw []byte, targetPath string) ([]byte, error) {
	if len(raw) > maxActivationConfigBytes {
		return nil, errors.New("recipe package config exceeds its size limit")
	}
	// Runtime realization also injects stack-local service/store defaults, so it
	// must run even when platform and algorithm overrides are absent.
	return realizeRecipeRuntimeConfigWithCLI(raw, targetPath)
}

func readActivationConfig(path string) ([]byte, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() || info.Size() > maxActivationConfigBytes {
		return nil, errors.New("active runtime config must be a bounded regular file")
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()
	openedInfo, err := file.Stat()
	if err != nil || !os.SameFile(info, openedInfo) {
		return nil, errors.New("active runtime config changed while opening")
	}
	data, err := io.ReadAll(io.LimitReader(file, maxActivationConfigBytes+1))
	if err != nil || len(data) > maxActivationConfigBytes {
		return nil, errors.New("active runtime config exceeds its size limit")
	}
	return data, nil
}

func writeActivationConfig(path string, data []byte) error {
	if len(data) > maxActivationConfigBytes {
		return errors.New("active runtime config exceeds its size limit")
	}
	if err := validateActivationConfigDestination(path); err != nil {
		return err
	}
	parent := filepath.Dir(path)
	file, err := os.CreateTemp(parent, ".recipe-active-*.tmp")
	if err != nil {
		return err
	}
	temp := file.Name()
	defer func() { _ = os.Remove(temp) }()
	if writeErr := writeActivationTempFile(file, data); writeErr != nil {
		return writeErr
	}
	if renameErr := os.Rename(temp, path); renameErr != nil {
		return renameErr
	}
	directory, err := os.Open(parent)
	if err != nil {
		return err
	}
	defer func() { _ = directory.Close() }()
	return directory.Sync()
}

func validateActivationConfigDestination(path string) error {
	info, err := os.Lstat(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
		return errors.New("active runtime config must be a regular file")
	}
	return nil
}

func writeActivationTempFile(file *os.File, data []byte) error {
	err := file.Chmod(0o600)
	if err == nil {
		_, err = file.Write(data)
	}
	if err == nil {
		err = file.Sync()
	}
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	return err
}

func activationDigest(data []byte) string {
	digest := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(digest[:])
}

type routerConfigHash struct {
	RuntimeHash string `json:"runtime_hash"`
	ActiveHash  string `json:"active_hash"`
	Status      string `json:"status"`
}

func newRouterActivationVerifier(routerAPIURL string, client *http.Client, credentialProvider ...routerauth.CredentialProvider) func(context.Context, string) error {
	endpoint, endpointErr := routerConfigHashURL(routerAPIURL)
	if client == nil {
		client = &http.Client{
			Transport: &http.Transport{Proxy: nil},
			Timeout:   5 * time.Second,
			CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
				return http.ErrUseLastResponse
			},
		}
	}
	return func(ctx context.Context, expectedDigest string) error {
		if endpointErr != nil {
			return endpointErr
		}
		expected := strings.TrimPrefix(expectedDigest, "sha256:")
		deadline := time.Now().Add(routerActivationTimeout)
		var last routerConfigHash
		for time.Now().Before(deadline) {
			var provider routerauth.CredentialProvider
			if len(credentialProvider) > 0 {
				provider = credentialProvider[0]
			}
			response, err := fetchRouterConfigHash(ctx, client, endpoint, provider)
			if err == nil {
				last = response
				if response.Status == "active" && response.RuntimeHash == expected && response.ActiveHash == expected {
					return nil
				}
			}
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(500 * time.Millisecond):
			}
		}
		return fmt.Errorf("router config did not become active (last status %q)", last.Status)
	}
}

func routerConfigHashURL(base string) (string, error) {
	parsed, err := url.Parse(strings.TrimSpace(base))
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return "", errors.New("router API URL is not configured")
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/") + "/config/hash"
	parsed.RawQuery = ""
	parsed.Fragment = ""
	parsed.User = nil
	return parsed.String(), nil
}

func fetchRouterConfigHash(ctx context.Context, client *http.Client, endpoint string, provider routerauth.CredentialProvider) (routerConfigHash, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return routerConfigHash{}, err
	}
	if authErr := routerauth.RewriteAuthorization(request, provider); authErr != nil {
		return routerConfigHash{}, authErr
	}
	response, err := client.Do(request)
	if err != nil {
		return routerConfigHash{}, err
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		return routerConfigHash{}, fmt.Errorf("router config hash returned HTTP %d", response.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, 64<<10))
	if err != nil {
		return routerConfigHash{}, err
	}
	var result routerConfigHash
	decoder := json.NewDecoder(bytes.NewReader(body))
	if err := decoder.Decode(&result); err != nil {
		return routerConfigHash{}, err
	}
	return result, nil
}

func activationFailed(message string, cause error) error {
	return recipe.NewPackageError(recipe.ErrorActivationFailed, http.StatusInternalServerError, message, cause)
}

func activationRollbackFailed(cause error) error {
	return recipe.NewPackageError(recipe.ErrorActivationRollback, http.StatusInternalServerError, "Recipe activation failed and automatic recovery could not restore a consistent runtime.", cause)
}

func activationCommitIncomplete(cause error) error {
	return recipe.NewPackageError(recipe.ErrorActivationConflict, http.StatusConflict, "Recipe activation commit cleanup is incomplete and will be resumed safely.", cause)
}

func activationIncompatible(cause error) error {
	return recipe.NewPackageError(recipe.ErrorActivationIncompatible, http.StatusConflict, "Recipe package is incompatible with the running stack.", cause)
}
