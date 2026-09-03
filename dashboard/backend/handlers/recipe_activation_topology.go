package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
)

const (
	topologyHelperTimeout          = 3 * time.Minute
	managedContainerInspectTimeout = 5 * time.Second
	managedStorageInspectFormat    = `{"network_mode":{{json .HostConfig.NetworkMode}},"publish_all_ports":{{json .HostConfig.PublishAllPorts}},"configured":{{json .HostConfig.PortBindings}},"actual":{{json .NetworkSettings.Ports}}}`
)

type managedStoragePortBinding struct {
	HostIP   *string `json:"HostIp"`
	HostPort *string `json:"HostPort"`
}

type managedStoragePortInspection struct {
	NetworkMode     *string         `json:"network_mode"`
	PublishAllPorts *bool           `json:"publish_all_ports"`
	Configured      json.RawMessage `json:"configured"`
	Actual          json.RawMessage `json:"actual"`
}

type managedRuntimeTopologyReconciler struct {
	storeRoot  string
	configPath string
}

func newManagedRuntimeTopologyReconciler(store *recipe.Store, configPath string) runtimeTopologyReconciler {
	return &managedRuntimeTopologyReconciler{storeRoot: store.Root(), configPath: filepath.Clean(configPath)}
}

func (r *managedRuntimeTopologyReconciler) Inventory(ctx context.Context) (runtimeTopologyInventory, error) {
	inventory := runtimeTopologyInventory{ContainerRunning: map[string]bool{}}
	if !isRunningInContainer() || !managedRuntimeUsesSplitContainers() {
		storage, err := environmentStorageInventory()
		if err != nil {
			return inventory, err
		}
		inventory.Storage = storage
		return inventory, nil
	}
	if err := r.inventoryManagedServices(ctx, &inventory); err != nil {
		return inventory, err
	}
	if err := inventoryManagedStorage(ctx, &inventory); err != nil {
		return inventory, err
	}
	return inventory, nil
}

func (r *managedRuntimeTopologyReconciler) inventoryManagedServices(ctx context.Context, inventory *runtimeTopologyInventory) error {
	for _, service := range []string{"router", "envoy"} {
		name := managedContainerNameForService(service)
		if err := validateTopologyContainerName(name); err != nil {
			return err
		}
		status, err := strictManagedContainerStatus(ctx, name)
		if err != nil {
			return err
		}
		inventory.ContainerRunning[name] = status == "running"
		if service == "router" && status != "not found" {
			if err := r.validateManagedRouterIsolation(ctx, name); err != nil {
				return err
			}
		}
	}
	return nil
}

func (r *managedRuntimeTopologyReconciler) validateManagedRouterIsolation(ctx context.Context, name string) error {
	data, err := readActivationConfig(r.configPath)
	if err != nil {
		return errors.New("active runtime config could not be inspected for Router management isolation")
	}
	management, _, _, err := activationManagementAPI(data, false)
	if err != nil {
		return errors.New("active Router management listener configuration is invalid")
	}
	if err := validateManagedRouterPortIsolation(ctx, name, management); err != nil {
		return fmt.Errorf("managed Router management publication is unsafe: %w", err)
	}
	return nil
}

func inventoryManagedStorage(ctx context.Context, inventory *runtimeTopologyInventory) error {
	for _, backend := range []string{"redis", "postgres", "milvus"} {
		name := managedContainerNameForStorage(backend)
		if err := validateTopologyContainerName(name); err != nil {
			return err
		}
		status, err := strictManagedContainerStatus(ctx, name)
		if err != nil {
			return err
		}
		if status != "not found" {
			if err := validateManagedStoragePortIsolation(ctx, name); err != nil {
				return fmt.Errorf("managed %s storage is unsafe: %w", backend, err)
			}
			inventory.Storage = append(inventory.Storage, backend)
		}
		inventory.ContainerRunning[name] = status == "running"
	}
	return nil
}

func validateManagedRouterPortIsolation(parent context.Context, containerName string, management recipe.ActivationManagementListener) error {
	ctx, cancel := context.WithTimeout(parent, managedContainerInspectTimeout)
	defer cancel()
	command := exec.CommandContext(ctx, "docker", "container", "inspect", "--format", managedStorageInspectFormat, containerName)
	output, err := command.CombinedOutput()
	if err != nil {
		if ctx.Err() != nil {
			return errors.New("router management port isolation inspection timed out")
		}
		return errors.New("router management port isolation could not be inspected")
	}
	return validateManagedRouterPortInspection(output, management)
}

func validateManagedRouterPortInspection(data []byte, management recipe.ActivationManagementListener) error {
	inspection, configured, actual, err := decodeManagedPortInspection(data)
	if err != nil {
		return errors.New("router management port isolation inspection is invalid")
	}
	networkMode := strings.ToLower(strings.TrimSpace(*inspection.NetworkMode))
	if networkMode == "host" {
		return errors.New("host network mode bypasses Router management loopback isolation")
	}
	if strings.HasPrefix(networkMode, "container:") {
		return errors.New("container network mode bypasses Router management loopback isolation")
	}
	if *inspection.PublishAllPorts {
		return errors.New("PublishAllPorts bypasses Router management loopback isolation")
	}
	containerPort := strconv.Itoa(management.Port) + "/tcp"
	configuredBinding, err := exactManagedRouterBinding(configured, containerPort, management.HostPort)
	if err != nil {
		return fmt.Errorf("configured Router management publication is invalid: %w", err)
	}
	actualBinding, err := exactManagedRouterBinding(actual, containerPort, management.HostPort)
	if err != nil {
		return fmt.Errorf("actual Router management publication is invalid: %w", err)
	}
	if configuredBinding != actualBinding {
		return errors.New("configured and actual Router management publications do not match")
	}
	return nil
}

func exactManagedRouterBinding(bindings map[string][]managedStoragePortBinding, containerPort string, expectedHostPort int) (string, error) {
	entries, ok := bindings[containerPort]
	if !ok || len(entries) != 1 {
		return "", errors.New("expected exactly one published binding")
	}
	binding := entries[0]
	ip, hostPort, err := managedRouterHostBinding(binding, expectedHostPort)
	if err != nil {
		return "", err
	}
	if managedRouterHostPortShared(bindings, containerPort, hostPort) {
		return "", errors.New("published host port is shared with another container port")
	}
	return net.JoinHostPort(ip.String(), hostPort), nil
}

func managedRouterHostBinding(binding managedStoragePortBinding, expectedHostPort int) (net.IP, string, error) {
	if binding.HostIP == nil || binding.HostPort == nil {
		return nil, "", errors.New("published binding is incomplete")
	}
	hostPort := strings.TrimSpace(*binding.HostPort)
	parsedPort, err := strconv.Atoi(hostPort)
	if err != nil || parsedPort != expectedHostPort {
		return nil, "", errors.New("published host port does not match the managed stack")
	}
	ip := net.ParseIP(strings.TrimSpace(*binding.HostIP))
	if ip == nil || !ip.IsLoopback() {
		return nil, "", errors.New("published host address is not loopback")
	}
	return ip, hostPort, nil
}

func managedRouterHostPortShared(bindings map[string][]managedStoragePortBinding, containerPort, hostPort string) bool {
	for otherPort, otherEntries := range bindings {
		if otherPort == containerPort {
			continue
		}
		for _, other := range otherEntries {
			if other.HostPort != nil && strings.TrimSpace(*other.HostPort) == hostPort {
				return true
			}
		}
	}
	return false
}

func decodeManagedPortInspection(data []byte) (managedStoragePortInspection, map[string][]managedStoragePortBinding, map[string][]managedStoragePortBinding, error) {
	inspection, err := decodeManagedPortInspectionEnvelope(data)
	if err != nil {
		return inspection, nil, nil, err
	}
	configured, err := decodeManagedStoragePortBindings(inspection.Configured)
	if err != nil {
		return inspection, nil, nil, err
	}
	actual, err := decodeManagedStoragePortBindings(inspection.Actual)
	if err != nil {
		return inspection, nil, nil, err
	}
	return inspection, configured, actual, nil
}

func decodeManagedPortInspectionEnvelope(data []byte) (managedStoragePortInspection, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var inspection managedStoragePortInspection
	if err := decoder.Decode(&inspection); err != nil {
		return inspection, err
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return inspection, errors.New("inspection has trailing data")
	}
	if inspection.NetworkMode == nil || inspection.PublishAllPorts == nil || len(inspection.Configured) == 0 || len(inspection.Actual) == 0 {
		return inspection, errors.New("inspection fields are incomplete")
	}
	return inspection, nil
}

func strictManagedContainerStatus(parent context.Context, containerName string) (string, error) {
	ctx, cancel := context.WithTimeout(parent, managedContainerInspectTimeout)
	defer cancel()
	command := exec.CommandContext(ctx, "docker", "container", "inspect", "--format", "{{.State.Status}}", containerName)
	output, err := command.CombinedOutput()
	if err != nil {
		if ctx.Err() != nil {
			return "", errors.New("managed container inspection timed out")
		}
		if managedContainerInspectReportsNotFound(output) {
			return "not found", nil
		}
		return "", errors.New("managed container status could not be inspected")
	}
	status := strings.ToLower(strings.TrimSpace(string(output)))
	switch status {
	case "created", "restarting", "running", "removing", "paused", "exited", "dead":
		return status, nil
	default:
		return "", errors.New("managed container status inspection is invalid")
	}
}

func managedContainerInspectReportsNotFound(output []byte) bool {
	message := strings.ToLower(string(output))
	return strings.Contains(message, "no such container") ||
		strings.Contains(message, "no such object") ||
		strings.Contains(message, "no container with name or id")
}

func validateManagedStoragePortIsolation(parent context.Context, containerName string) error {
	ctx, cancel := context.WithTimeout(parent, managedContainerInspectTimeout)
	defer cancel()
	command := exec.CommandContext(ctx, "docker", "container", "inspect", "--format", managedStorageInspectFormat, containerName)
	output, err := command.CombinedOutput()
	if err != nil {
		if ctx.Err() != nil {
			return errors.New("container port isolation inspection timed out")
		}
		return errors.New("container port isolation could not be inspected")
	}
	return validateManagedStoragePortInspection(output)
}

func validateManagedStoragePortInspection(data []byte) error {
	inspection, err := decodeManagedPortInspectionEnvelope(data)
	if err != nil {
		return errors.New("container port isolation inspection is invalid")
	}
	networkMode := strings.ToLower(strings.TrimSpace(*inspection.NetworkMode))
	if networkMode == "host" {
		return errors.New("host network mode bypasses loopback port isolation")
	}
	if strings.HasPrefix(networkMode, "container:") {
		return errors.New("container network mode bypasses loopback port isolation")
	}
	if *inspection.PublishAllPorts {
		return errors.New("PublishAllPorts bypasses explicit loopback bindings")
	}
	for label, raw := range map[string]json.RawMessage{
		"configured": inspection.Configured,
		"actual":     inspection.Actual,
	} {
		ports, err := decodeManagedStoragePortBindings(raw)
		if err != nil {
			return errors.New("container port isolation inspection is invalid")
		}
		if err := validateManagedStorageBindings(label, ports); err != nil {
			return err
		}
	}
	return nil
}

func validateManagedStorageBindings(label string, ports map[string][]managedStoragePortBinding) error {
	for containerPort, bindings := range ports {
		if strings.TrimSpace(containerPort) == "" {
			return errors.New("container port isolation inspection is invalid")
		}
		for _, binding := range bindings {
			if err := validateManagedStorageBinding(label, containerPort, binding); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateManagedStorageBinding(label, containerPort string, binding managedStoragePortBinding) error {
	if binding.HostIP == nil || binding.HostPort == nil {
		return errors.New("container port isolation inspection is invalid")
	}
	hostIP := strings.TrimSpace(*binding.HostIP)
	port, err := strconv.Atoi(strings.TrimSpace(*binding.HostPort))
	if err != nil || port < 1 || port > 65535 {
		return errors.New("container port isolation inspection is invalid")
	}
	ip := net.ParseIP(hostIP)
	if ip == nil || !ip.IsLoopback() {
		return fmt.Errorf(
			"%s published port %s uses non-loopback host address %q; preserve its data mounts and recreate it with loopback bindings before activation",
			label,
			containerPort,
			hostIP,
		)
	}
	return nil
}

func decodeManagedStoragePortBindings(raw json.RawMessage) (map[string][]managedStoragePortBinding, error) {
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return map[string][]managedStoragePortBinding{}, nil
	}
	var documents map[string]json.RawMessage
	if err := json.Unmarshal(raw, &documents); err != nil || documents == nil {
		return nil, errors.New("invalid port bindings")
	}
	bindings := make(map[string][]managedStoragePortBinding, len(documents))
	for port, document := range documents {
		if bytes.Equal(bytes.TrimSpace(document), []byte("null")) {
			bindings[port] = nil
			continue
		}
		var entries []json.RawMessage
		if err := json.Unmarshal(document, &entries); err != nil {
			return nil, errors.New("invalid port bindings")
		}
		for _, entry := range entries {
			decoder := json.NewDecoder(bytes.NewReader(entry))
			decoder.DisallowUnknownFields()
			var binding managedStoragePortBinding
			if err := decoder.Decode(&binding); err != nil {
				return nil, errors.New("invalid port bindings")
			}
			var trailing any
			if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
				return nil, errors.New("invalid port bindings")
			}
			bindings[port] = append(bindings[port], binding)
		}
	}
	return bindings, nil
}

func (r *managedRuntimeTopologyReconciler) Apply(ctx context.Context, state recipe.ActivationTopologyState, credential string) error {
	if state.CredentialEnv != "" && strings.TrimSpace(credential) == "" {
		return errors.New("managed Router credential is unavailable")
	}
	return r.run(ctx, "apply", state, credential)
}

func (r *managedRuntimeTopologyReconciler) Rollback(ctx context.Context, state recipe.ActivationTopologyState) error {
	return r.run(ctx, "rollback", state, "")
}

func (r *managedRuntimeTopologyReconciler) Commit(ctx context.Context, state recipe.ActivationTopologyState) error {
	return r.run(ctx, "commit", state, "")
}

func (r *managedRuntimeTopologyReconciler) run(parent context.Context, action string, state recipe.ActivationTopologyState, credential string) error {
	if !isRunningInContainer() || !managedRuntimeUsesSplitContainers() {
		return errors.New("managed stack recreation requires the split container runtime")
	}
	if action != "apply" && action != "rollback" && action != "commit" {
		return errors.New("invalid topology reconciliation action")
	}
	cliRoot, python, statePath, err := r.topologyHelperRuntime(state)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(parent, topologyHelperTimeout)
	defer cancel()
	command := exec.CommandContext(ctx, python, "-m", "cli.recipe_topology_reconcile", action, "--state", statePath, "--config", r.configPath)
	command.Dir = cliRoot
	command.Env = append(os.Environ(), "PYTHONPATH="+cliRoot)
	if credential != "" {
		command.Stdin = strings.NewReader(credential + "\n")
	}
	var output bytes.Buffer
	command.Stdout = &output
	command.Stderr = &output
	if err := command.Run(); err != nil {
		return fmt.Errorf("topology %s failed: %w (%s)", action, err, strings.TrimSpace(output.String()))
	}
	return validateTopologyHelperReceipt(output.String())
}

func (r *managedRuntimeTopologyReconciler) topologyHelperRuntime(state recipe.ActivationTopologyState) (string, string, string, error) {
	cliRoot := detectPythonCLIRoot()
	if cliRoot == "" {
		return "", "", "", errors.New("python CLI is unavailable for topology reconciliation")
	}
	python, err := runtimeSyncPythonBinary()
	if err != nil {
		return "", "", "", err
	}
	statePath := filepath.Join(r.storeRoot, "transactions", state.TransactionID, "topology.json")
	info, err := os.Lstat(statePath)
	if err != nil || info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
		return "", "", "", errors.New("activation topology journal is unavailable")
	}
	return cliRoot, python, statePath, nil
}

func validateTopologyHelperReceipt(output string) error {
	lines := strings.Split(strings.TrimSpace(output), "\n")
	if len(lines) == 0 {
		return errors.New("topology helper returned an empty receipt")
	}
	var response struct {
		Status string `json:"status"`
	}
	decoder := json.NewDecoder(strings.NewReader(lines[len(lines)-1]))
	if err := decoder.Decode(&response); err != nil || response.Status != "ok" {
		return errors.New("topology helper returned an invalid receipt")
	}
	return nil
}

func validateTopologyContainerName(value string) error {
	if !regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$`).MatchString(value) {
		return errors.New("managed container name is invalid")
	}
	return nil
}
