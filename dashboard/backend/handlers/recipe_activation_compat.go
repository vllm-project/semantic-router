package handlers

import (
	"errors"
	"fmt"
	"net"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const managedStorageBackends = "VLLM_SR_MANAGED_STORAGE_BACKENDS"

func managementAPIState(data []byte) (string, bool, error) {
	_, config, managed, err := activationManagementAPI(data, false)
	if err != nil {
		return "", false, err
	}
	return config.Auth.Mode, managed, nil
}

func activationManagementAPI(data []byte, requireManagedReachability bool) (recipe.ActivationManagementListener, routerconfig.ManagementAPIConfig, bool, error) {
	parsed, err := routerconfig.ParseYAMLBytesWithoutEnvExpansion(data)
	if err != nil {
		return recipe.ActivationManagementListener{}, routerconfig.ManagementAPIConfig{}, false, err
	}
	management := normalizedActivationManagementAPI(parsed.ManagementAPI)
	if validationErr := validateActivationManagementAPI(management, requireManagedReachability); validationErr != nil {
		return recipe.ActivationManagementListener{}, routerconfig.ManagementAPIConfig{}, false, validationErr
	}
	offset, err := runtimePortOffset()
	if err != nil || management.Port+offset > 65535 {
		return recipe.ActivationManagementListener{}, routerconfig.ManagementAPIConfig{}, false, errors.New("management API host port is outside the managed host range")
	}
	return recipe.ActivationManagementListener{
		BindAddress: management.BindAddress,
		Port:        management.Port,
		HostPort:    management.Port + offset,
	}, management, hasManagedManagementCredential(management), nil
}

func normalizedActivationManagementAPI(management routerconfig.ManagementAPIConfig) routerconfig.ManagementAPIConfig {
	management.BindAddress = strings.TrimSpace(management.BindAddress)
	if management.BindAddress == "" {
		management.BindAddress = routerconfig.DefaultManagementAPIConfig().BindAddress
	}
	if management.Port == 0 {
		management.Port = routerconfig.DefaultManagementAPIConfig().Port
	}
	management.Auth.Mode = strings.TrimSpace(management.Auth.Mode)
	if management.Auth.Mode == "" {
		management.Auth.Mode = routerconfig.ManagementAuthModeDisabled
	}
	return management
}

func validateActivationManagementAPI(management routerconfig.ManagementAPIConfig, requireManagedReachability bool) error {
	if management.Port < 1 || management.Port > 65535 {
		return errors.New("management API port is outside the valid range")
	}
	if management.Port == 50051 || management.Port == 9190 {
		return errors.New("management API port conflicts with a Router service port")
	}
	if ip := net.ParseIP(management.BindAddress); ip == nil && management.BindAddress != "localhost" {
		return errors.New("management API bind address must be an IP address or localhost")
	}
	if requireManagedReachability && management.BindAddress != "0.0.0.0" {
		return errors.New("managed split activation requires management_api.bind_address 0.0.0.0")
	}
	return validateActivationManagementAuth(management)
}

func validateActivationManagementAuth(management routerconfig.ManagementAPIConfig) error {
	if management.Auth.Mode != routerconfig.ManagementAuthModeDisabled && management.Auth.Mode != routerconfig.ManagementAuthModeBearer {
		return errors.New("management API auth mode is unsupported")
	}
	if management.RemoteExposure && (management.Auth.Mode != routerconfig.ManagementAuthModeBearer || len(management.Auth.Tokens) == 0) {
		return errors.New("management API remote exposure requires bearer auth tokens")
	}
	return nil
}

func hasManagedManagementCredential(management routerconfig.ManagementAPIConfig) bool {
	for _, token := range management.Auth.Tokens {
		if token.Env == recipe.ManagementCredentialEnv && token.Role == recipe.ManagementCredentialRole {
			return true
		}
	}
	return false
}

func managedSplitManagementReachabilityRequired() bool {
	if strings.TrimSpace(os.Getenv(routerContainerNameEnv)) != "" &&
		strings.TrimSpace(os.Getenv(dashboardContainerNameEnv)) != "" &&
		managedRuntimeUsesSplitContainers() {
		return true
	}
	value := strings.TrimSpace(os.Getenv("TARGET_ROUTER_API_URL"))
	if value == "" {
		return false
	}
	parsed, err := url.Parse(value)
	if err != nil || parsed.Hostname() == "" {
		return false
	}
	return strings.EqualFold(parsed.Hostname(), managedContainerNameForService("router"))
}

type listenerIdentity struct {
	name    string
	address string
	port    int
}

func listenerTopology(data []byte) ([]listenerIdentity, error) {
	config, err := routerconfig.ParseYAMLBytesWithoutEnvExpansion(data)
	if err != nil {
		return nil, err
	}
	result := make([]listenerIdentity, 0, len(config.Listeners))
	for _, listener := range config.Listeners {
		result = append(result, listenerIdentity{
			name:    listener.Name,
			address: listener.Address,
			port:    listener.Port,
		})
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].name != result[j].name {
			return result[i].name < result[j].name
		}
		if result[i].address != result[j].address {
			return result[i].address < result[j].address
		}
		return result[i].port < result[j].port
	})
	return result, nil
}

func managedStorageInventory(provisionedRaw string) ([]string, error) {
	provisioned := map[string]struct{}{}
	for _, value := range strings.Split(provisionedRaw, ",") {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if value != "redis" && value != "postgres" && value != "milvus" {
			return nil, fmt.Errorf("managed storage backend inventory is invalid: %s", value)
		}
		provisioned[value] = struct{}{}
	}
	result := make([]string, 0, len(provisioned))
	for backend := range provisioned {
		result = append(result, backend)
	}
	sort.Strings(result)
	return result, nil
}

func requiredManagedStorageList(configData []byte) ([]string, error) {
	required, err := requiredManagedStorageBackends(configData)
	if err != nil {
		return nil, err
	}
	result := make([]string, 0, len(required))
	for backend := range required {
		result = append(result, backend)
	}
	sort.Strings(result)
	return result, nil
}

func activationListeners(data []byte) ([]recipe.ActivationListener, error) {
	topology, err := listenerTopology(data)
	if err != nil {
		return nil, err
	}
	offset, err := runtimePortOffset()
	if err != nil {
		return nil, err
	}
	result := make([]recipe.ActivationListener, 0, len(topology))
	listenerNames := map[string]struct{}{}
	hostBindings := make([]recipe.ActivationListener, 0, len(topology))
	for _, listener := range topology {
		if listener.port < 1 || listener.port+offset > 65535 || !publishableActivationListenerAddress(listener.address) {
			return nil, fmt.Errorf("listener port is outside the managed host range")
		}
		candidate := recipe.ActivationListener{
			Name: listener.name, Address: listener.address, Port: listener.port, HostPort: listener.port + offset,
		}
		if _, ok := listenerNames[listener.name]; ok {
			return nil, fmt.Errorf("listener topology contains duplicate names")
		}
		for _, existing := range hostBindings {
			if recipe.ActivationListenerBindingsConflict(candidate, existing) {
				return nil, fmt.Errorf("listener topology contains conflicting host bindings")
			}
		}
		listenerNames[listener.name] = struct{}{}
		hostBindings = append(hostBindings, candidate)
		result = append(result, candidate)
	}
	return result, nil
}

func publishableActivationListenerAddress(value string) bool {
	switch strings.TrimSpace(value) {
	case "0.0.0.0", "::", "127.0.0.1", "::1":
		return true
	default:
		return false
	}
}

func runtimePortOffset() (int, error) {
	raw := strings.TrimSpace(os.Getenv("VLLM_SR_PORT_OFFSET"))
	if raw == "" {
		return 0, nil
	}
	offset, err := strconv.Atoi(raw)
	if err != nil || offset < 0 || offset > 65535 {
		return 0, fmt.Errorf("invalid runtime port offset")
	}
	return offset, nil
}

func requiredManagedStorageBackends(configData []byte) (map[string]struct{}, error) {
	config, err := routerconfig.ParseYAMLBytesWithoutEnvExpansion(configData)
	if err != nil {
		return nil, err
	}
	return requiredManagedStorageForConfig(config), nil
}

func requiredManagedStorageForConfig(config *routerconfig.RouterConfig) map[string]struct{} {
	required := map[string]struct{}{}
	if config == nil {
		return required
	}
	addResponseAPIStorage(required, config)
	addRouterReplayStorage(required, config)
	addStartupStatusStorage(required, config)
	addSemanticCacheStorage(required, config)
	addMemoryStorage(required, config)
	addVectorStorage(required, config)
	return required
}

func addResponseAPIStorage(required map[string]struct{}, config *routerconfig.RouterConfig) {
	if config.ResponseAPI.Enabled && strings.EqualFold(strings.TrimSpace(config.ResponseAPI.StoreBackend), "redis") &&
		(anyManagedStorageEndpoint(config.ResponseAPI.Redis.ClusterAddresses, "redis") || isManagedStorageEndpoint(config.ResponseAPI.Redis.Address, "redis")) {
		required["redis"] = struct{}{}
	}
}

func addRouterReplayStorage(required map[string]struct{}, config *routerconfig.RouterConfig) {
	if !config.RouterReplay.Enabled {
		return
	}
	switch strings.ToLower(strings.TrimSpace(config.RouterReplay.StoreBackend)) {
	case "redis":
		if config.RouterReplay.Redis != nil && isManagedStorageEndpoint(config.RouterReplay.Redis.Address, "redis") {
			required["redis"] = struct{}{}
		}
	case "postgres":
		if config.RouterReplay.Postgres != nil && isManagedStorageEndpoint(config.RouterReplay.Postgres.Host, "postgres") {
			required["postgres"] = struct{}{}
		}
	case "milvus":
		if config.RouterReplay.Milvus != nil && isManagedStorageEndpoint(config.RouterReplay.Milvus.Address, "milvus") {
			required["milvus"] = struct{}{}
		}
	}
}

func addStartupStatusStorage(required map[string]struct{}, config *routerconfig.RouterConfig) {
	if strings.EqualFold(strings.TrimSpace(config.StartupStatus.StoreBackend), "redis") &&
		config.StartupStatus.Redis != nil && isManagedStorageEndpoint(config.StartupStatus.Redis.Address, "redis") {
		required["redis"] = struct{}{}
	}
}

func addSemanticCacheStorage(required map[string]struct{}, config *routerconfig.RouterConfig) {
	if !config.SemanticCache.Enabled {
		return
	}
	switch strings.ToLower(strings.TrimSpace(config.SemanticCache.BackendType)) {
	case "redis":
		if config.SemanticCache.Redis != nil && isManagedStorageEndpoint(config.SemanticCache.Redis.Connection.Host, "redis") {
			required["redis"] = struct{}{}
		}
	case "milvus":
		if config.SemanticCache.Milvus != nil && isManagedStorageEndpoint(config.SemanticCache.Milvus.Connection.Host, "milvus") {
			required["milvus"] = struct{}{}
		}
	}
}

func addMemoryStorage(required map[string]struct{}, config *routerconfig.RouterConfig) {
	if !memoryStoreEnabled(config) {
		return
	}
	backend := strings.ToLower(strings.TrimSpace(config.Memory.Backend))
	if backend == "" {
		backend = "milvus"
	}
	if backend == "milvus" && isManagedStorageEndpoint(config.Memory.Milvus.Address, "milvus") {
		required["milvus"] = struct{}{}
	}
	if config.Memory.RedisCache != nil && config.Memory.RedisCache.Enabled &&
		isManagedStorageEndpoint(config.Memory.RedisCache.Address, "redis") {
		required["redis"] = struct{}{}
	}
}

func addVectorStorage(required map[string]struct{}, config *routerconfig.RouterConfig) {
	vector := config.VectorStore
	if vector == nil || !vector.Enabled {
		return
	}
	if strings.EqualFold(strings.TrimSpace(vector.BackendType), "milvus") && vector.Milvus != nil &&
		isManagedStorageEndpoint(vector.Milvus.Connection.Host, "milvus") {
		required["milvus"] = struct{}{}
	}
	if strings.EqualFold(strings.TrimSpace(vector.MetadataStore), "postgres") && vector.MetadataPostgres != nil &&
		isManagedStorageEndpoint(vector.MetadataPostgres.Host, "postgres") {
		required["postgres"] = struct{}{}
	}
}

func memoryStoreEnabled(config *routerconfig.RouterConfig) bool {
	if config.Memory.Enabled {
		return true
	}
	for _, decision := range config.AllRoutingDecisions() {
		if decision.HasPlugin(routerconfig.DecisionPluginMemory) {
			return true
		}
	}
	return false
}

func anyManagedStorageEndpoint(endpoints []string, backend string) bool {
	for _, endpoint := range endpoints {
		if isManagedStorageEndpoint(endpoint, backend) {
			return true
		}
	}
	return false
}

func isManagedStorageEndpoint(endpoint, backend string) bool {
	host := managedStorageEndpointHost(endpoint)
	if host == "" {
		return false
	}
	host = strings.ToLower(strings.TrimSuffix(host, "."))
	return host == strings.ToLower(backend) || host == strings.ToLower(managedContainerNameForStorage(backend))
}

func managedStorageEndpointHost(endpoint string) string {
	endpoint = strings.TrimSpace(endpoint)
	if endpoint == "" {
		return ""
	}
	if parsed, err := url.Parse(endpoint); err == nil && parsed.Host != "" {
		return parsed.Hostname()
	}
	if host, _, err := net.SplitHostPort(endpoint); err == nil {
		return strings.Trim(host, "[]")
	}
	return strings.Trim(endpoint, "[]")
}
