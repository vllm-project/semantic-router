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
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
	Topology       runtimeTopologySource
	AttemptTimeout time.Duration
}

// RecipeActivator coordinates a rollback-capable activation transaction under
// the same deploy lock used by ordinary Dashboard config deployment. An
// activation the running containers can't take is committed as a pending
// activation that the next `vllm-sr serve` applies.
type RecipeActivator struct {
	store          *recipe.Store
	configPath     string
	configDir      string
	applyRuntime   func(string, string) (string, error)
	realizeConfig  func([]byte, string) ([]byte, error)
	verifyRuntime  func(context.Context, string) error
	verifyEnvoy    func(context.Context) error
	topology       runtimeTopologySource
	attemptTimeout time.Duration
}

const (
	recipeRecreationDetail   = "the activated Recipe needs the containers recreated"
	recipeRestorationDetail  = "the restored source config needs the containers recreated"
	serveRecoversTopologyMsg = "A Recipe activation that recreated containers is waiting for recovery. Run `vllm-sr stop`, then `vllm-sr serve`."
)

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
		activator.topology = environmentRuntimeTopology{}
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
	if authorizationErr := revalidateRecipeMutation(operationContext); authorizationErr != nil {
		return recipe.ActivateResult{}, authorizationErr
	}
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
		if _, pending := asRestartNeeded(activeCheckErr); pending {
			return recipe.ActivateResult{
				Status:       recipe.ActivationResultRestartRequired,
				Message:      restartRequiredMessage(a.configPath),
				RecipeDigest: target.RecipeDigest,
			}, nil
		}
	}
	target, plan, previousConfig, realizedConfig, err := a.prepareActivation(ctx, request)
	if err != nil {
		return recipe.ActivateResult{}, err
	}
	if confirmationErr := requireActivationConfirmation(request, plan); confirmationErr != nil {
		return recipe.ActivateResult{}, confirmationErr
	}
	// Recovery above repairs an older transaction. Check this request's live
	// permission after planning, just before its first baseline or journal write.
	if revalidationErr := revalidateRecipeMutation(ctx); revalidationErr != nil {
		return recipe.ActivateResult{}, revalidationErr
	}
	if state == recipe.ActivationNone {
		if baselineErr := a.store.RefreshSourceBaseline(previousConfig); baselineErr != nil {
			return recipe.ActivateResult{}, activationFailed("Source Recipe baseline could not be preserved.", baselineErr)
		}
	}
	transaction, err := a.store.BeginActivation(target.RecipeDigest, previousConfig)
	if err != nil {
		return recipe.ActivateResult{}, activationFailed("Recipe activation transaction could not be started.", err)
	}
	return a.executeActivation(ctx, target, plan, transaction, previousConfig, realizedConfig)
}

func (a *RecipeActivator) executeActivation(ctx context.Context, target recipe.PackageSummary, plan recipe.ActivationPlan, transaction recipe.ActivationTransaction, previousConfig, realizedConfig []byte) (recipe.ActivateResult, error) {
	published, restart, err := a.publishActivation(ctx, plan, realizedConfig, recipeRecreationDetail)
	if err != nil {
		return recipe.ActivateResult{}, a.failAndRollback(ctx, transaction, previousConfig, err)
	}
	pointer := recipe.ActivePointer{
		RecipeDigest:         target.RecipeDigest,
		ConfigDigest:         target.ConfigDigest,
		RealizedConfigDigest: activationDigest(published),
	}
	if commitErr := a.store.PrepareActivationCommit(transaction, pointer); commitErr != nil {
		return recipe.ActivateResult{}, a.failAndRollback(ctx, transaction, previousConfig, commitErr)
	}
	result := recipe.ActivateResult{
		Status:               "active",
		RecipeDigest:         target.RecipeDigest,
		PreviousRecipeDigest: transaction.PreviousRecipeDigest,
		PlanDigest:           plan.PlanDigest,
		Mode:                 plan.Mode,
	}
	if restart != nil {
		if recordErr := recordPendingActivation(a.configPath, published, activationReasonRestart, restart.detail); recordErr != nil {
			return recipe.ActivateResult{}, activationCommitIncomplete(recordErr)
		}
		result.Status, result.Message = recipe.ActivationResultRestartRequired, restartRequiredMessage(a.configPath)
	}
	if commitErr := a.store.FinalizeActivationCommit(transaction); commitErr != nil {
		return recipe.ActivateResult{}, activationCommitIncomplete(commitErr)
	}
	return result, nil
}

func (a *RecipeActivator) Deactivate(ctx context.Context, requests ...recipe.DeactivateRequest) (recipe.DeactivateResult, error) {
	release, err := beginRecipeActivationMutation(a.store.Root())
	if err != nil {
		return recipe.DeactivateResult{}, packageRuntimeConfigMutationError(err)
	}
	defer release()
	operationContext, cancel := context.WithTimeout(context.WithoutCancel(ctx), 2*a.attemptTimeout+2*time.Minute)
	defer cancel()
	if authorizationErr := revalidateRecipeMutation(operationContext); authorizationErr != nil {
		return recipe.DeactivateResult{}, authorizationErr
	}
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
	// Recovery above repairs an older transaction. Check this request's live
	// permission after planning, just before its deactivation journal write.
	if revalidationErr := revalidateRecipeMutation(ctx); revalidationErr != nil {
		return recipe.DeactivateResult{}, revalidationErr
	}
	transaction, err := a.store.BeginDeactivation(previousConfig)
	if err != nil {
		return recipe.DeactivateResult{}, activationFailed("Recipe deactivation transaction could not be started.", err)
	}
	return a.executeDeactivation(ctx, active, plan, transaction, previousConfig, baseline)
}

func (a *RecipeActivator) executeDeactivation(ctx context.Context, active recipe.ActivePointer, plan recipe.ActivationPlan, transaction recipe.ActivationTransaction, previousConfig, baseline []byte) (recipe.DeactivateResult, error) {
	published, restart, err := a.publishActivation(ctx, plan, baseline, recipeRestorationDetail)
	if err != nil {
		return recipe.DeactivateResult{}, a.failDeactivationAndRollback(ctx, transaction, previousConfig, err)
	}
	if commitErr := a.store.PrepareDeactivationCommit(transaction); commitErr != nil {
		return recipe.DeactivateResult{}, a.failDeactivationAndRollback(ctx, transaction, previousConfig, commitErr)
	}
	result := recipe.DeactivateResult{Status: "inactive", PreviousRecipeDigest: active.RecipeDigest, PlanDigest: plan.PlanDigest, Mode: plan.Mode}
	if restart != nil {
		if recordErr := recordPendingActivation(a.configPath, published, activationReasonRestart, restart.detail); recordErr != nil {
			return recipe.DeactivateResult{}, activationCommitIncomplete(recordErr)
		}
		result.Status, result.Message = recipe.ActivationResultRestartRequired, restartRequiredMessage(a.configPath)
	}
	if commitErr := a.store.FinalizeDeactivationCommit(transaction); commitErr != nil {
		return recipe.DeactivateResult{}, activationCommitIncomplete(commitErr)
	}
	return result, nil
}

// publishActivation writes config as the runtime config and returns it as
// written, with the restart it needs, if any. The running containers take it,
// or it waits for the next `vllm-sr serve` to create them anew: when the plan
// recreates them (recreationDetail says why), when Envoy's generated config
// changes, or when the Router refuses it only as restart_required. Anything
// else must become active or the activation fails.
func (a *RecipeActivator) publishActivation(ctx context.Context, plan recipe.ActivationPlan, config []byte, recreationDetail string) ([]byte, *restartNeededError, error) {
	if err := revalidateRecipeMutation(ctx); err != nil {
		return nil, nil, err
	}
	if err := writeActivationConfig(a.configPath, config); err != nil {
		return nil, nil, err
	}
	restart, err := a.applyPublishedConfig(ctx, plan, recreationDetail)
	if err != nil {
		return nil, nil, err
	}
	published, err := readActivationConfig(a.configPath)
	if err != nil {
		return nil, nil, err
	}
	if plan.Mode != recipe.ActivationModeStackRecreation {
		verificationErr := a.verifyRuntimeAttempt(ctx, activationDigest(published))
		if routerRestart, ok := asRestartNeeded(verificationErr); ok {
			restart, verificationErr = routerRestart, nil
		}
		if verificationErr != nil {
			return nil, nil, verificationErr
		}
	}
	if err := revalidateRecipeMutation(ctx); err != nil {
		return nil, nil, err
	}
	return published, restart, nil
}

func (a *RecipeActivator) applyPublishedConfig(ctx context.Context, plan recipe.ActivationPlan, recreationDetail string) (*restartNeededError, error) {
	if plan.Mode == recipe.ActivationModeStackRecreation {
		if _, err := refreshManagedSplitEnvoyConfig(a.configPath); err != nil {
			return nil, err
		}
		return &restartNeededError{detail: recreationDetail}, nil
	}
	if err := revalidateRecipeMutation(ctx); err != nil {
		return nil, err
	}
	realizedPath, err := a.applyRuntime(a.configPath, a.configDir)
	restart, _ := asRestartNeeded(err)
	if restart != nil {
		err = nil
	}
	if err != nil {
		return nil, err
	}
	if filepath.Clean(realizedPath) != a.configPath {
		return nil, errors.New("runtime config synchronization selected a different active path")
	}
	return restart, nil
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
	if a.recreatedContainers(transaction) {
		return recipe.NewPackageError(recipe.ErrorActivationConflict, http.StatusConflict, serveRecoversTopologyMsg, nil)
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

// recreatedContainers reports a transaction an earlier Dashboard began when it
// still recreated containers itself. Only `vllm-sr serve`, which owns the
// containers, can roll it back or finish it.
func (a *RecipeActivator) recreatedContainers(transaction recipe.ActivationTransaction) bool {
	if transaction.TopologyMode == recipe.ActivationTopologyManaged {
		return true
	}
	_, err := a.store.ActivationTopology(transaction)
	return !errors.Is(err, os.ErrNotExist)
}

func (a *RecipeActivator) recoverCommit(ctx context.Context, transaction recipe.ActivationTransaction) error {
	expectedDigest, isDeactivation, err := a.recoveryCommitDigest(transaction)
	if err != nil {
		return activationCommitIncomplete(err)
	}
	if expectedDigest != transaction.CommitConfigDigest {
		return activationCommitIncomplete(errors.New("committing runtime config digest does not match the journal"))
	}
	// A commit that waits for `vllm-sr serve` is complete once its pointer is
	// written; the running containers keep the previous config until then.
	verificationErr := a.verifyActivatedRuntime(ctx, expectedDigest)
	if _, pending := asRestartNeeded(verificationErr); verificationErr != nil && !pending {
		return activationCommitIncomplete(verificationErr)
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
	if isRecipeAuthorizationError(activationErr) {
		return activationErr
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
	if isRecipeAuthorizationError(deactivationErr) {
		return deactivationErr
	}
	return activationFailed("Recipe deactivation failed; the active package runtime was restored.", deactivationErr)
}

func (a *RecipeActivator) independentRollbackContext(parent context.Context) (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.WithoutCancel(parent), a.attemptTimeout)
}

// rollback restores the previous runtime config. A restart the restored config
// needs is the one it needed before the transaction began, so it stays as it
// was rather than failing the rollback.
func (a *RecipeActivator) rollback(ctx context.Context, transaction recipe.ActivationTransaction, previousConfig []byte) error {
	current, _, err := a.store.Transaction()
	if err != nil {
		return err
	}
	if current.ID != transaction.ID {
		return errors.New("activation transaction changed unexpectedly")
	}
	if a.recreatedContainers(current) {
		return errors.New("only `vllm-sr serve` can recover a Recipe activation that recreated containers")
	}
	if writeErr := writeActivationConfig(a.configPath, previousConfig); writeErr != nil {
		return writeErr
	}
	realizedPath, err := a.applyRuntime(a.configPath, a.configDir)
	if _, restart := asRestartNeeded(err); restart {
		err = nil
	}
	if err != nil {
		return err
	}
	if filepath.Clean(realizedPath) != a.configPath {
		return errors.New("rollback selected a different active runtime config path")
	}
	restoredConfig, err := readActivationConfig(a.configPath)
	if err != nil {
		return err
	}
	verificationErr := a.verifyActivatedRuntime(ctx, activationDigest(restoredConfig))
	if _, restart := asRestartNeeded(verificationErr); restart {
		verificationErr = nil
	}
	if verificationErr != nil {
		return verificationErr
	}
	return a.store.CompleteRollback(current)
}

func applyRecipeRuntime(configPath string, configDir string) (string, error) {
	effectiveConfigPath, err := syncRuntimeConfigForCurrentRuntime(configPath)
	if err != nil {
		return "", err
	}
	if isRunningInContainer() && isManagedContainerConfigPath(configPath) && managedRuntimeUsesSplitContainers() {
		// The Router hot-reloads a Recipe hot switch. Envoy reads its config
		// when it starts, so a changed one waits for `vllm-sr serve`.
		envoyChanged, err := refreshManagedSplitEnvoyConfig(effectiveConfigPath)
		if err != nil {
			return "", err
		}
		if envoyChanged {
			return effectiveConfigPath, &restartNeededError{detail: envoyRestartDetail}
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

// writeActivationTempFile stages the active config with the mode of every
// config the Dashboard saves: `vllm-sr serve` reads it as its own user, and the
// `.vllm-sr` directory, not the file, keeps other users out.
func writeActivationTempFile(file *os.File, data []byte) error {
	err := file.Chmod(0o644)
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
	GeneratedRuntimeHash string               `json:"generated_runtime_hash"`
	ActiveRuntimeHash    string               `json:"active_runtime_hash"`
	ActivationStatus     string               `json:"activation_status"`
	Activation           *routerConfigAttempt `json:"activation"`
}

// routerConfigAttempt is the Router's latest attempt to activate a document.
type routerConfigAttempt struct {
	DocumentHash string               `json:"document_hash"`
	Status       string               `json:"status"`
	Reasons      []routerConfigReason `json:"reasons"`
}

type routerConfigReason struct {
	Code    string `json:"code"`
	Path    string `json:"path"`
	Message string `json:"message"`
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
				if response.ActivationStatus == "active" && response.GeneratedRuntimeHash == expected && response.ActiveRuntimeHash == expected {
					return nil
				}
				if detail, done, restart := judgeRouterAttempt(response, expected); done && restart {
					return &restartNeededError{detail: detail}
				}
			}
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(500 * time.Millisecond):
			}
		}
		return fmt.Errorf("router config did not become active (last status %q)", last.ActivationStatus)
	}
}

func routerConfigHashURL(base string) (string, error) {
	parsed, err := url.Parse(strings.TrimSpace(base))
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return "", errors.New("router API URL is not configured")
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/") + "/api/v1/config/hash"
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

// requireManagementCredential refuses a plan with bearer authentication when
// the Dashboard has no management credential: the Router `vllm-sr serve`
// creates for it would accept nothing from the Dashboard.
func (a *RecipeActivator) requireManagementCredential(plan recipe.ActivationPlan) error {
	if plan.ManagementAuth.Mode != routerconfig.ManagementAuthModeBearer || a.store.HasManagementCredential() {
		return nil
	}
	return recipe.NewPackageError(recipe.ErrorActivationIncompatible, http.StatusConflict,
		"This Recipe turns on bearer authentication for the Router management API, and the Dashboard has no management credential. Start the stack with `vllm-sr serve`, which provides it, or set "+recipe.ManagementCredentialEnv+" for both the Dashboard and the Router.", nil)
}
