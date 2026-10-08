package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type runtimeTopologyInventory struct {
	Storage []string
}

type activationPlanInputs struct {
	currentListeners         []recipe.ActivationListener
	targetListeners          []recipe.ActivationListener
	currentStorage           []string
	targetStorage            []string
	currentManagement        recipe.ActivationManagementListener
	targetManagement         recipe.ActivationManagementListener
	currentManagementConfig  routerconfig.ManagementAPIConfig
	targetManagementConfig   routerconfig.ManagementAPIConfig
	currentManagedCredential bool
	targetManagedCredential  bool
}

type runtimeTopologySource interface {
	Inventory(context.Context) (runtimeTopologyInventory, error)
}

func (a *RecipeActivator) Preview(ctx context.Context, request recipe.ActivateRequest) (recipe.ActivationPlan, error) {
	release, err := beginRecipeActivationMutation(a.store.Root())
	if err != nil {
		return recipe.ActivationPlan{}, packageRuntimeConfigMutationError(err)
	}
	defer release()
	if recoveryErr := a.recoverLocked(ctx); recoveryErr != nil {
		return recipe.ActivationPlan{}, recoveryErr
	}
	_, plan, _, _, err := a.prepareActivation(ctx, request)
	return plan, err
}

func (a *RecipeActivator) PreviewDeactivation(ctx context.Context) (recipe.ActivationPlan, error) {
	release, err := beginRecipeActivationMutation(a.store.Root())
	if err != nil {
		return recipe.ActivationPlan{}, packageRuntimeConfigMutationError(err)
	}
	defer release()
	if recoveryErr := a.recoverLocked(ctx); recoveryErr != nil {
		return recipe.ActivationPlan{}, recoveryErr
	}
	_, plan, _, _, err := a.prepareDeactivation(ctx)
	return plan, err
}

func (a *RecipeActivator) prepareDeactivation(ctx context.Context) (recipe.ActivePointer, recipe.ActivationPlan, []byte, []byte, error) {
	active, state, statusErr := a.store.ActivationStatus()
	if state == recipe.ActivationNone && statusErr == nil {
		return active, recipe.ActivationPlan{Mode: recipe.ActivationModeHotSwitch}, nil, nil, nil
	}
	if active.RecipeDigest == "" {
		return active, recipe.ActivationPlan{}, nil, nil, activationFailed("Active Recipe state could not be read.", statusErr)
	}
	baseline, err := a.store.SourceBaseline()
	if err != nil {
		return active, recipe.ActivationPlan{}, nil, nil, activationFailed("Source Recipe baseline could not be read.", err)
	}
	baseline, _, err = bindActivationManagementCredential(baseline)
	if err != nil {
		return active, recipe.ActivationPlan{}, nil, nil, activationFailed("Source management credential could not be bound.", err)
	}
	previousConfig, err := readActivationConfig(a.configPath)
	if err != nil {
		return active, recipe.ActivationPlan{}, nil, nil, activationFailed("Active runtime config could not be read.", err)
	}
	inventory, err := a.topology.Inventory(ctx)
	if err != nil {
		return active, recipe.ActivationPlan{}, nil, nil, activationFailed("Managed runtime topology could not be inspected.", err)
	}
	plan, err := buildActivationPlan(active.RecipeDigest, previousConfig, baseline, inventory, a.store.HasManagementCredential())
	if err != nil {
		return active, recipe.ActivationPlan{}, nil, nil, activationIncompatible(err)
	}
	if credentialErr := a.requireManagementCredential(plan); credentialErr != nil {
		return active, recipe.ActivationPlan{}, nil, nil, credentialErr
	}
	plan.Effects[0] = "restore the durable source runtime config"
	plan.PlanDigest = ""
	encoded, err := json.Marshal(plan)
	if err != nil {
		return active, recipe.ActivationPlan{}, nil, nil, err
	}
	plan.PlanDigest = activationDigest(encoded)
	return active, plan, previousConfig, baseline, nil
}

func (a *RecipeActivator) prepareActivation(ctx context.Context, request recipe.ActivateRequest) (recipe.PackageSummary, recipe.ActivationPlan, []byte, []byte, error) {
	target, rawConfig, err := a.store.ActivationTarget(request.RecipeDigest, request.AcknowledgeWarnings)
	if err != nil {
		return recipe.PackageSummary{}, recipe.ActivationPlan{}, nil, nil, err
	}
	if bindingErr := recipe.ValidateEnvironmentBindings(rawConfig); bindingErr != nil {
		return recipe.PackageSummary{}, recipe.ActivationPlan{}, nil, nil, bindingErr
	}
	previousConfig, err := readActivationConfig(a.configPath)
	if err != nil {
		return recipe.PackageSummary{}, recipe.ActivationPlan{}, nil, nil, activationFailed("Active runtime config could not be read.", err)
	}
	realizedConfig, err := a.realizeConfig(rawConfig, a.configPath)
	if err != nil {
		return recipe.PackageSummary{}, recipe.ActivationPlan{}, nil, nil, activationFailed("Recipe runtime configuration could not be realized.", err)
	}
	realizedConfig, _, err = bindActivationManagementCredential(realizedConfig)
	if err != nil {
		return recipe.PackageSummary{}, recipe.ActivationPlan{}, nil, nil, activationFailed("Recipe management credential could not be bound.", err)
	}
	inventory, err := a.topology.Inventory(ctx)
	if err != nil {
		return recipe.PackageSummary{}, recipe.ActivationPlan{}, nil, nil, activationFailed("Managed runtime topology could not be inspected.", err)
	}
	plan, err := buildActivationPlan(target.RecipeDigest, previousConfig, realizedConfig, inventory, a.store.HasManagementCredential())
	if err != nil {
		return recipe.PackageSummary{}, recipe.ActivationPlan{}, nil, nil, activationIncompatible(err)
	}
	if credentialErr := a.requireManagementCredential(plan); credentialErr != nil {
		return recipe.PackageSummary{}, recipe.ActivationPlan{}, nil, nil, credentialErr
	}
	return target, plan, previousConfig, realizedConfig, nil
}

func buildActivationPlan(recipeDigest string, currentConfig, targetConfig []byte, inventory runtimeTopologyInventory, credentialAvailable bool) (recipe.ActivationPlan, error) {
	inputs, err := inspectActivationPlanInputs(currentConfig, targetConfig, inventory)
	if err != nil {
		return recipe.ActivationPlan{}, err
	}
	if validationErr := validateActivationPlanManagement(inputs, credentialAvailable); validationErr != nil {
		return recipe.ActivationPlan{}, validationErr
	}
	plan := newActivationPlan(recipeDigest, inputs, credentialAvailable)
	encoded, err := json.Marshal(plan)
	if err != nil {
		return recipe.ActivationPlan{}, err
	}
	plan.PlanDigest = activationDigest(encoded)
	return plan, nil
}

func newActivationPlan(recipeDigest string, inputs activationPlanInputs, credentialAvailable bool) recipe.ActivationPlan {
	storage := activationStorageDiff(inputs)
	recreate, managementBoundaryChanged := activationPlanNeedsRecreation(inputs, storage, credentialAvailable)
	plan := recipe.ActivationPlan{
		RecipeDigest:     recipeDigest,
		Mode:             recipe.ActivationModeHotSwitch,
		ListenersBefore:  inputs.currentListeners,
		ListenersAfter:   inputs.targetListeners,
		Storage:          storage,
		ManagementBefore: inputs.currentManagement,
		ManagementAfter:  inputs.targetManagement,
		ManagementAuth:   activationPlanManagementAuth(inputs.targetManagementConfig.Auth.Mode),
		Effects:          []string{"publish the selected Recipe runtime config", "verify Router config identity and Envoy readiness"},
	}
	configureActivationRecreation(&plan, recreate, managementBoundaryChanged)
	return plan
}

// activationStorageDiff compares the storage the stack runs with the target's.
// Repair stays empty: the Dashboard can't see container states, and the next
// `vllm-sr serve` starts every storage container the config needs.
func activationStorageDiff(inputs activationPlanInputs) recipe.ActivationStorageDiff {
	add, remove := storageDelta(inputs.currentStorage, inputs.targetStorage)
	return recipe.ActivationStorageDiff{
		Before: inputs.currentStorage,
		After:  inputs.targetStorage,
		Add:    add,
		Remove: remove,
		Repair: []string{},
	}
}

func activationPlanNeedsRecreation(inputs activationPlanInputs, storage recipe.ActivationStorageDiff, credentialAvailable bool) (bool, bool) {
	managementBoundaryChanged := !reflect.DeepEqual(inputs.currentManagementConfig, inputs.targetManagementConfig)
	topologyChanged := !reflect.DeepEqual(inputs.currentListeners, inputs.targetListeners) ||
		managementBoundaryChanged || len(storage.Add) > 0 || len(storage.Remove) > 0
	authRequiresRecreation := inputs.targetManagementConfig.Auth.Mode == routerconfig.ManagementAuthModeBearer &&
		(!inputs.targetManagedCredential || !inputs.currentManagedCredential || !credentialAvailable)
	return topologyChanged || authRequiresRecreation, managementBoundaryChanged
}

func activationPlanManagementAuth(mode string) recipe.ActivationManagementAuth {
	auth := recipe.ActivationManagementAuth{Mode: mode, ServicePermissions: []string{}}
	if mode == routerconfig.ManagementAuthModeBearer {
		auth.ServiceRole = recipe.ManagementCredentialRole
		auth.ServicePermissions = recipe.ManagementCredentialPermissions()
	}
	return auth
}

// configureActivationRecreation turns a plan the running containers can't take
// into a pending activation: the Dashboard publishes the config, and the next
// `vllm-sr serve` creates the containers anew from it.
func configureActivationRecreation(plan *recipe.ActivationPlan, recreate, managementBoundaryChanged bool) {
	if !recreate {
		return
	}
	plan.Mode = recipe.ActivationModeStackRecreation
	plan.RequiresConfirmation = true
	containers := "the managed Router and Envoy containers"
	if !managedStackRunsEnvoy() {
		containers = "the managed Router container, which serves the listeners"
	}
	plan.Effects = []string{
		plan.Effects[0],
		"record a pending restart: the next `vllm-sr serve` recreates " + containers + "; until then they keep serving",
	}
	if managementBoundaryChanged {
		plan.Effects = append(plan.Effects, "the recreated Router takes the target management listener and authentication boundary")
	}
	if len(plan.Storage.Add) > 0 {
		plan.Effects = append(plan.Effects, "the next `vllm-sr serve` starts the storage the Recipe adds")
	}
	if len(plan.Storage.Remove) > 0 {
		plan.Effects = append(plan.Effects, "storage the Recipe stops using keeps its data and runs until `vllm-sr stop`")
	}
}

func inspectActivationPlanInputs(currentConfig, targetConfig []byte, inventory runtimeTopologyInventory) (activationPlanInputs, error) {
	var inputs activationPlanInputs
	var err error
	inputs.currentListeners, err = activationListeners(currentConfig)
	if err != nil {
		return inputs, fmt.Errorf("current listener topology is invalid: %w", err)
	}
	inputs.targetListeners, err = activationListeners(targetConfig)
	if err != nil {
		return inputs, fmt.Errorf("target listener topology is invalid: %w", err)
	}
	inputs.targetStorage, err = requiredManagedStorageList(targetConfig)
	if err != nil {
		return inputs, err
	}
	inputs.currentStorage = normalizedStorageSet(inventory.Storage)
	requireManagedReachability := managedSplitManagementReachabilityRequired()
	inputs.currentManagement, inputs.currentManagementConfig, inputs.currentManagedCredential, err = activationManagementAPI(currentConfig, requireManagedReachability)
	if err != nil {
		return inputs, fmt.Errorf("current management API configuration is invalid: %w", err)
	}
	inputs.targetManagement, inputs.targetManagementConfig, inputs.targetManagedCredential, err = activationManagementAPI(targetConfig, requireManagedReachability)
	if err != nil {
		return inputs, fmt.Errorf("target management API configuration is invalid: %w", err)
	}
	return inputs, nil
}

func validateActivationPlanManagement(inputs activationPlanInputs, credentialAvailable bool) error {
	if inputs.currentManagement.Port != inputs.targetManagement.Port {
		return errors.New("recipe activation cannot change the Router management API port while the Dashboard is running")
	}
	if inputs.currentManagement.BindAddress != inputs.targetManagement.BindAddress {
		return errors.New("recipe activation cannot change the Router management API bind address while the Dashboard is running")
	}
	targetMode := inputs.targetManagementConfig.Auth.Mode
	if targetMode == routerconfig.ManagementAuthModeBearer && !inputs.targetManagedCredential {
		return errors.New("target management API does not contain the Dashboard service credential binding")
	}
	currentMode := inputs.currentManagementConfig.Auth.Mode
	if currentMode == routerconfig.ManagementAuthModeBearer && (!inputs.currentManagedCredential || !credentialAvailable) {
		return errors.New("the running authenticated management API does not have a recoverable Dashboard service credential")
	}
	for _, listener := range inputs.targetListeners {
		if listener.HostPort == inputs.targetManagement.HostPort {
			return errors.New("target management API host port conflicts with an Envoy listener host port")
		}
	}
	return nil
}

func normalizedStorageSet(values []string) []string {
	seen := map[string]struct{}{}
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "redis" || value == "postgres" || value == "milvus" {
			seen[value] = struct{}{}
		}
	}
	result := make([]string, 0, len(seen))
	for value := range seen {
		result = append(result, value)
	}
	sort.Strings(result)
	return result
}

func storageDelta(before, after []string) ([]string, []string) {
	beforeSet, afterSet := map[string]struct{}{}, map[string]struct{}{}
	for _, value := range before {
		beforeSet[value] = struct{}{}
	}
	for _, value := range after {
		afterSet[value] = struct{}{}
	}
	add, remove := []string{}, []string{}
	for _, value := range after {
		if _, ok := beforeSet[value]; !ok {
			add = append(add, value)
		}
	}
	for _, value := range before {
		if _, ok := afterSet[value]; !ok {
			remove = append(remove, value)
		}
	}
	return add, remove
}

func requireActivationConfirmation(request recipe.ActivateRequest, plan recipe.ActivationPlan) error {
	if request.ExpectedPlanDigest != "" && request.ExpectedPlanDigest != plan.PlanDigest {
		return recipe.NewPackageError(recipe.ErrorActivationConfirmation, 409, "The activation plan changed after preview; review the current plan before activating.", nil)
	}
	if plan.Mode != recipe.ActivationModeStackRecreation {
		return nil
	}
	if !request.ConfirmStackRecreation || request.ExpectedPlanDigest != plan.PlanDigest {
		return recipe.NewPackageError(recipe.ErrorActivationConfirmation, 409, "Stack recreation requires explicit confirmation of the current activation plan.", nil)
	}
	return nil
}

func requireDeactivationConfirmation(request recipe.DeactivateRequest, plan recipe.ActivationPlan) error {
	if request.ExpectedPlanDigest != "" && request.ExpectedPlanDigest != plan.PlanDigest {
		return recipe.NewPackageError(recipe.ErrorActivationConfirmation, 409, "The deactivation plan changed after preview; review the current plan before restoring the source runtime.", nil)
	}
	if plan.Mode != recipe.ActivationModeStackRecreation {
		return nil
	}
	if !request.ConfirmStackRecreation || request.ExpectedPlanDigest != plan.PlanDigest {
		return recipe.NewPackageError(recipe.ErrorActivationConfirmation, 409, "Stack recreation requires explicit confirmation of the current deactivation plan.", nil)
	}
	return nil
}

func environmentStorageInventory() ([]string, error) {
	return managedStorageInventory(os.Getenv(managedStorageBackends))
}
