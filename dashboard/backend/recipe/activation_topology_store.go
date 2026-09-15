package recipe

import (
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

const (
	ActivationTopologySchema   = "vllm-sr/recipe-activation-topology/v1"
	ManagementCredentialEnv    = "VLLM_SR_DASHBOARD_RECIPE_TOKEN" //nolint:gosec // This is an environment variable name, not a credential.
	ManagementCredentialRole   = "dashboard_control_plane"
	managementCredentialFile   = "router-management.token"
	maxActivationTopologyBytes = 256 << 10
)

var managementCredentialPermissions = [...]string{
	"cache.invalidate",
	"cache.manage",
	"cache.read",
	"classify.invoke",
	"compression.manage",
	"compression.preview",
	"compression.read",
	"config.read",
	"config.write",
	"learning.ingest",
	"ready.read",
	"replay.detail",
	"replay.read",
}

// ManagementCredentialPermissions returns the exact Router permissions used
// by Dashboard validation, topology, plugin operations, config-backed KB
// management, feedback, and runtime readiness. The returned slice is a copy so
// callers cannot mutate the canonical role contract.
func ManagementCredentialPermissions() []string {
	return append([]string(nil), managementCredentialPermissions[:]...)
}

var managedContainerNamePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$`)

func (s *Store) WriteActivationTopology(transaction ActivationTransaction, state ActivationTopologyState) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	current, err := s.readTransaction()
	if err != nil || current.ID != transaction.ID || current.State != "pending" {
		return wrapPackageError(ErrorActivationConflict, 409, "Recipe activation transaction changed unexpectedly.", err)
	}
	state.SchemaVersion = ActivationTopologySchema
	state.TransactionID = transaction.ID
	normalizeActivationTopologyState(&state)
	if err := validateActivationTopologyState(state); err != nil {
		return err
	}
	if err := writeJSONAtomically(s.topologyStatePath(transaction.ID), state, 0o600); err != nil {
		return err
	}
	current.TopologyMode = ActivationTopologyManaged
	return writeJSONAtomically(s.transactionPath(), current, 0o600)
}

func (s *Store) ActivationTopology(transaction ActivationTransaction) (ActivationTopologyState, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	current, err := s.readTransaction()
	if err != nil || current.ID != transaction.ID {
		return ActivationTopologyState{}, wrapPackageError(ErrorActivationConflict, 409, "Recipe activation transaction changed unexpectedly.", err)
	}
	return s.readActivationTopology(transaction.ID)
}

func (s *Store) readActivationTopology(transactionID string) (ActivationTopologyState, error) {
	var state ActivationTopologyState
	if !validTransactionID(transactionID) {
		return state, errors.New("invalid activation transaction identity")
	}
	if err := readStrictJSONFile(s.topologyStatePath(transactionID), maxActivationTopologyBytes, &state); err != nil {
		return state, err
	}
	if err := validateActivationTopologyState(state); err != nil {
		return ActivationTopologyState{}, err
	}
	if state.TransactionID != transactionID {
		return ActivationTopologyState{}, errors.New("activation topology transaction mismatch")
	}
	return state, nil
}

func (s *Store) topologyStatePath(transactionID string) string {
	return filepath.Join(s.root, "transactions", transactionID, "topology.json")
}

func normalizeActivationTopologyState(state *ActivationTopologyState) {
	sort.Strings(state.StorageBefore)
	sort.Strings(state.StorageAfter)
	sort.Slice(state.Listeners, func(i, j int) bool {
		if state.Listeners[i].Name != state.Listeners[j].Name {
			return state.Listeners[i].Name < state.Listeners[j].Name
		}
		if state.Listeners[i].Address != state.Listeners[j].Address {
			return state.Listeners[i].Address < state.Listeners[j].Address
		}
		return state.Listeners[i].Port < state.Listeners[j].Port
	})
	sort.Slice(state.Containers, func(i, j int) bool {
		if state.Containers[i].Service != state.Containers[j].Service {
			return state.Containers[i].Service < state.Containers[j].Service
		}
		return state.Containers[i].Name < state.Containers[j].Name
	})
}

func validateActivationTopologyState(state ActivationTopologyState) error {
	if state.SchemaVersion != ActivationTopologySchema || !validTransactionID(state.TransactionID) || !validDigest(state.PlanDigest) {
		return errors.New("invalid activation topology identity")
	}
	if state.CredentialEnv != "" && state.CredentialEnv != ManagementCredentialEnv {
		return errors.New("invalid activation topology credential binding")
	}
	if err := validateStorageSet(state.StorageBefore); err != nil {
		return err
	}
	if err := validateStorageSet(state.StorageAfter); err != nil {
		return err
	}
	storageBefore, storageAfter := topologyStringSet(state.StorageBefore), topologyStringSet(state.StorageAfter)
	transitions, err := validateActivationContainerTransitions(state.Containers, storageBefore, storageAfter)
	if err != nil {
		return err
	}
	if err := validateRequiredRuntimeTransitions(transitions); err != nil {
		return err
	}
	if err := validateManagedStorageTransitions(transitions, storageBefore, storageAfter); err != nil {
		return err
	}
	return validateActivationListeners(state.Listeners)
}

func validateActivationContainerTransitions(
	containers []ActivationContainerTransition,
	storageBefore, storageAfter map[string]struct{},
) (map[string]string, error) {
	seen := map[string]struct{}{}
	transitions := map[string]string{}
	for _, transition := range containers {
		if err := validateActivationContainerTransition(transition, seen, storageBefore, storageAfter); err != nil {
			return nil, err
		}
		transitions[transition.Service] = transition.Action
	}
	return transitions, nil
}

func validateActivationContainerTransition(
	transition ActivationContainerTransition,
	seen, storageBefore, storageAfter map[string]struct{},
) error {
	if err := validateActivationContainerTransitionShape(transition); err != nil {
		return err
	}
	if err := registerActivationContainerIdentities(seen, transition.Name, transition.BackupName); err != nil {
		return err
	}
	return validateActivationContainerTransitionAction(transition, storageBefore, storageAfter)
}

func validateActivationContainerTransitionShape(transition ActivationContainerTransition) error {
	if !validActivationTopologyService(transition.Service) {
		return errors.New("invalid activation topology service")
	}
	if !managedContainerNamePattern.MatchString(transition.Name) ||
		(transition.BackupName != "" && !managedContainerNamePattern.MatchString(transition.BackupName)) {
		return errors.New("invalid activation topology container name")
	}
	if !validActivationTopologyAction(transition.Action) {
		return errors.New("invalid activation topology action")
	}
	if transition.Action == "add" && transition.BackupName != "" ||
		transition.Action != "add" && transition.BackupName == "" {
		return errors.New("invalid activation topology backup")
	}
	return nil
}

func validateActivationContainerTransitionAction(
	transition ActivationContainerTransition,
	storageBefore, storageAfter map[string]struct{},
) error {
	if transition.Service == "router" || transition.Service == "envoy" {
		if transition.Action != "replace" {
			return errors.New("invalid runtime topology action")
		}
		return nil
	}
	_, before := storageBefore[transition.Service]
	_, after := storageAfter[transition.Service]
	if invalidStorageTopologyAction(transition.Action, before, after) {
		return errors.New("invalid storage topology action")
	}
	return nil
}

func invalidStorageTopologyAction(action string, before, after bool) bool {
	return action == "replace" ||
		action == "add" && (before || !after) ||
		action == "remove" && (!before || after) ||
		action == "repair" && (!before || !after)
}

func validActivationTopologyService(service string) bool {
	return service == "router" || service == "envoy" || service == "redis" || service == "postgres" || service == "milvus"
}

func validActivationTopologyAction(action string) bool {
	return action == "replace" || action == "add" || action == "remove" || action == "repair"
}

func registerActivationContainerIdentities(seen map[string]struct{}, identities ...string) error {
	for _, identity := range identities {
		if identity == "" {
			continue
		}
		if _, ok := seen[identity]; ok {
			return errors.New("duplicate activation topology container identity")
		}
		seen[identity] = struct{}{}
	}
	return nil
}

func validateRequiredRuntimeTransitions(transitions map[string]string) error {
	if transitions["router"] != "replace" || transitions["envoy"] != "replace" {
		return errors.New("incomplete runtime topology transitions")
	}
	return nil
}

func validateManagedStorageTransitions(
	transitions map[string]string,
	storageBefore, storageAfter map[string]struct{},
) error {
	for _, service := range []string{"redis", "postgres", "milvus"} {
		_, before := storageBefore[service]
		_, after := storageAfter[service]
		action, present := transitions[service]
		expected := expectedManagedStorageTransition(before, after, present)
		if present != (expected != "") || present && action != expected {
			return errors.New("storage transition does not match managed set")
		}
	}
	return nil
}

func expectedManagedStorageTransition(before, after, present bool) string {
	switch {
	case !before && after:
		return "add"
	case before && !after:
		return "remove"
	case before && after && present:
		return "repair"
	default:
		return ""
	}
}

func validateActivationListeners(listeners []ActivationListener) error {
	if len(listeners) == 0 {
		return errors.New("activation topology listeners are empty")
	}
	listenerNames := map[string]struct{}{}
	hostBindings := make([]ActivationListener, 0, len(listeners))
	for _, listener := range listeners {
		if strings.TrimSpace(listener.Name) == "" || !publishableListenerAddress(listener.Address) || listener.Port < 1 || listener.Port > 65535 || listener.HostPort < 1 || listener.HostPort > 65535 {
			return errors.New("invalid activation topology listener")
		}
		if _, ok := listenerNames[listener.Name]; ok {
			return errors.New("duplicate activation topology listener")
		}
		for _, existing := range hostBindings {
			if ActivationListenerBindingsConflict(listener, existing) {
				return errors.New("conflicting activation topology listener")
			}
		}
		listenerNames[listener.Name] = struct{}{}
		hostBindings = append(hostBindings, listener)
	}
	return nil
}

// ActivationListenerBindingsConflict reports whether two host publications
// overlap within the same address family. It is shared by preview generation
// and the persisted topology validator so confirmation cannot bless a plan
// that Docker cannot bind.
func ActivationListenerBindingsConflict(left, right ActivationListener) bool {
	if left.HostPort != right.HostPort {
		return false
	}
	leftIPv6 := strings.Contains(left.Address, ":")
	rightIPv6 := strings.Contains(right.Address, ":")
	if leftIPv6 != rightIPv6 {
		return false
	}
	wildcard := "0.0.0.0"
	if leftIPv6 {
		wildcard = "::"
	}
	return left.Address == right.Address || left.Address == wildcard || right.Address == wildcard
}

func publishableListenerAddress(value string) bool {
	switch strings.TrimSpace(value) {
	case "0.0.0.0", "::", "127.0.0.1", "::1":
		return true
	default:
		return false
	}
}

func topologyStringSet(values []string) map[string]struct{} {
	result := make(map[string]struct{}, len(values))
	for _, value := range values {
		result[value] = struct{}{}
	}
	return result
}

func validateStorageSet(values []string) error {
	seen := map[string]struct{}{}
	for _, value := range values {
		if value != "redis" && value != "postgres" && value != "milvus" {
			return errors.New("invalid managed storage backend")
		}
		if _, ok := seen[value]; ok {
			return errors.New("duplicate managed storage backend")
		}
		seen[value] = struct{}{}
	}
	return nil
}

func (s *Store) EnsureManagementCredential() (string, error) {
	if token, err := validateManagementCredential(os.Getenv(ManagementCredentialEnv)); err == nil {
		return token, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensureLayout(); err != nil {
		return "", err
	}
	if token, err := s.readManagementCredentialLocked(); err == nil {
		return token, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", err
	}
	left, err := randomID()
	if err != nil {
		return "", err
	}
	right, err := randomID()
	if err != nil {
		return "", err
	}
	token := left + right
	if err := writeFileAtomically(s.managementCredentialPath(), []byte(token+"\n"), 0o600); err != nil {
		return "", err
	}
	return token, nil
}

func (s *Store) ManagementCredential() (string, error) {
	if s == nil {
		return "", os.ErrNotExist
	}
	if token, err := validateManagementCredential(os.Getenv(ManagementCredentialEnv)); err == nil {
		return token, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.readManagementCredentialLocked()
}

func (s *Store) HasManagementCredential() bool {
	_, err := s.ManagementCredential()
	return err == nil
}

func (s *Store) readManagementCredentialLocked() (string, error) {
	data, err := readBoundedFile(s.managementCredentialPath(), 256)
	if err != nil {
		return "", err
	}
	return validateManagementCredential(string(data))
}

func validateManagementCredential(value string) (string, error) {
	token := strings.TrimSpace(value)
	if token == "" {
		return "", os.ErrNotExist
	}
	if len(token) != 64 || !regexp.MustCompile(`^[0-9a-f]{64}$`).MatchString(token) {
		return "", errors.New("invalid managed Router credential")
	}
	return token, nil
}

func (s *Store) managementCredentialPath() string {
	return filepath.Join(s.root, "credentials", managementCredentialFile)
}
