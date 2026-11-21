package handlers

import (
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/routercontract"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/k8s/configwriter"
)

// persistedRouterConfigTTL is how long one ConfigMap read answers: the
// requests of a page share it, and a saved change shows within seconds.
const persistedRouterConfigTTL = 3 * time.Second

// persistedRouterConfigClock is time.Now; tests move it past the TTL.
var persistedRouterConfigClock = time.Now

var persistedRouterConfigCache struct {
	sync.Mutex
	target configwriter.ConfigMapTarget
	readAt time.Time
	data   []byte
	err    error
}

// ReadPersistedRouterConfig parses the Router config the Dashboard saves: the
// ConfigMap, through the Kubernetes API, on a deployment that names one, whose
// subPath mount never changes in a running pod (issue #3688), and the config
// file otherwise. The views of what the Router is configured with read it, so
// a saved change shows without a restart. A ConfigMap read answers for
// persistedRouterConfigTTL; the file is read on every call. Each call parses
// its own copy, as routerconfig.Parse did, expanding environment references.
func ReadPersistedRouterConfig(configPath string) (*routerconfig.RouterConfig, error) {
	data, err := readPersistedRouterConfigBytes(configPath)
	if err != nil {
		return nil, err
	}
	return routerconfig.ParseYAMLBytes(data)
}

// ReadToolSelection reads the tools catalog the saved Router config selects.
func ReadToolSelection(configPath string) (routercontract.ToolSelection, error) {
	cfg, err := ReadPersistedRouterConfig(configPath)
	if err != nil {
		return routercontract.ToolSelection{}, err
	}
	return routercontract.ToolSelectionFromConfig(cfg), nil
}

func readPersistedRouterConfigBytes(configPath string) ([]byte, error) {
	target, ok := configwriter.ConfigMapTargetFromEnv()
	if !ok {
		return readPersistedDashboardConfig(configPath)
	}
	cache := &persistedRouterConfigCache
	cache.Lock()
	defer cache.Unlock()
	now := persistedRouterConfigClock()
	if cache.target != target || cache.readAt.IsZero() || now.Sub(cache.readAt) >= persistedRouterConfigTTL {
		cache.data, cache.err = readPersistedDashboardConfig(configPath)
		cache.target, cache.readAt = target, now
	}
	return cache.data, cache.err
}
