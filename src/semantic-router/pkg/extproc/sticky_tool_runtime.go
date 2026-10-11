package extproc

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontools"
)

// stickyToolRuntime is one router generation's sticky tool-set state: the
// local session store and the bounded planner over it. A generation builds
// it only when a decision enables sticky selection, and its resource scope
// closes the store once every request pinned to the generation has drained,
// so a reload hands new requests an empty store while in-flight requests
// finish against the old one.
type stickyToolRuntime struct {
	manager       *sessiontools.Manager
	maxStateBytes int
}

// stickyToolClock is the clock for sticky state lifetimes; tests replace it
// with a synthetic clock. The store and manager read it through a closure,
// so a replacement also reaches runtimes built earlier.
var stickyToolClock = time.Now

// newStickyToolStore builds a generation's local store. Tests replace it to
// observe store lifecycle or inject store failures.
var newStickyToolStore = func(cfg config.ToolSessionStoreConfig) sessiontools.Store {
	return sessiontools.NewMemoryStore(cfg, func() time.Time { return stickyToolClock() })
}

func stickyToolSelectionConfigured(cfg *config.RouterConfig) bool {
	if cfg == nil {
		return false
	}
	for _, decision := range cfg.AllRoutingDecisions() {
		if decision.GetToolSelectionConfig().StickyEnabled() {
			return true
		}
	}
	return false
}

// buildStickyToolRuntime returns nil, allocating nothing, when no decision
// enables sticky selection. Otherwise it rechecks the supported-runtime
// contract and registers the store with resources before anything else can
// fail, so a failed generation build closes it.
func buildStickyToolRuntime(cfg *config.RouterConfig, resources *resourceScope) (*stickyToolRuntime, error) {
	if !stickyToolSelectionConfigured(cfg) {
		return nil, nil
	}
	if err := config.ValidateStickyToolSelectionSupport(cfg); err != nil {
		return nil, err
	}
	storeCfg := config.ToolSessionStoreConfig{}
	if cfg.ToolSessions != nil {
		storeCfg = *cfg.ToolSessions
	}
	store := newStickyToolStore(storeCfg)
	resources.add(store.Close)
	manager, err := sessiontools.NewManager(store, sessiontools.ManagerOptions{
		TTL:     time.Duration(storeCfg.EffectiveTTLSeconds()) * time.Second,
		Timeout: time.Duration(storeCfg.EffectiveTimeoutMs()) * time.Millisecond,
		Clock:   func() time.Time { return stickyToolClock() },
	})
	if err != nil {
		return nil, err
	}
	return &stickyToolRuntime{manager: manager, maxStateBytes: storeCfg.EffectiveMaxStateBytes()}, nil
}
