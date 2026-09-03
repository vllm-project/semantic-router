package config

import (
	"sync"
	"sync/atomic"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

var (
	configUpdateMu          sync.Mutex
	configUpdateSubscribers = map[uint64]chan *RouterConfig{}
	configUpdateNextID      uint64
)

// Replace replaces the globally cached config. It is safe for concurrent readers.
func Replace(newCfg *RouterConfig) {
	decisionNames := make([]string, 0, len(newCfg.Decisions))
	for _, decision := range newCfg.Decisions {
		decisionNames = append(decisionNames, decision.Name)
	}
	logging.ComponentDebugEvent("config", "config_replace_started", map[string]interface{}{
		"decision_count": len(newCfg.Decisions),
		"decision_names": decisionNames,
	})

	configMu.Lock()
	config = newCfg
	configErr = nil
	configMu.Unlock()

	configUpdateMu.Lock()
	subscribers := make(map[uint64]chan *RouterConfig, len(configUpdateSubscribers))
	for id, channel := range configUpdateSubscribers {
		subscribers[id] = channel
	}
	configUpdateMu.Unlock()

	notifyConfigUpdateSubscribers(newCfg, subscribers)
}

func notifyConfigUpdateSubscribers(newCfg *RouterConfig, subscribers map[uint64]chan *RouterConfig) {
	if len(subscribers) == 0 {
		logging.ComponentDebugEvent("config", "config_update_listener_missing", nil)
		return
	}
	for id, channel := range subscribers {
		select {
		case channel <- newCfg:
			logging.ComponentDebugEvent("config", "config_update_notified", map[string]interface{}{
				"subscriber_id":  id,
				"decision_count": len(newCfg.Decisions),
			})
		default:
			logging.ComponentWarnEvent("config", "config_update_notification_skipped", map[string]interface{}{
				"subscriber_id":  id,
				"reason":         "channel_full",
				"decision_count": len(newCfg.Decisions),
			})
		}
	}
}

// Get returns the current configuration.
func Get() *RouterConfig {
	configMu.RLock()
	defer configMu.RUnlock()
	return config
}

type ConfigUpdateSubscription struct {
	ch        chan *RouterConfig
	closeOnce sync.Once
	closeFn   func()
}

func (s *ConfigUpdateSubscription) Updates() <-chan *RouterConfig {
	if s == nil {
		return nil
	}
	return s.ch
}

func (s *ConfigUpdateSubscription) Close() {
	if s == nil {
		return
	}
	s.closeOnce.Do(func() {
		if s.closeFn != nil {
			s.closeFn()
		}
	})
}

func SubscribeConfigUpdates(buffer int) *ConfigUpdateSubscription {
	if buffer <= 0 {
		buffer = 1
	}

	id := atomic.AddUint64(&configUpdateNextID, 1)
	channel := make(chan *RouterConfig, buffer)

	configUpdateMu.Lock()
	configUpdateSubscribers[id] = channel
	configUpdateMu.Unlock()

	return &ConfigUpdateSubscription{
		ch: channel,
		closeFn: func() {
			configUpdateMu.Lock()
			delete(configUpdateSubscribers, id)
			configUpdateMu.Unlock()
			close(channel)
		},
	}
}

// WatchConfigUpdates returns a compatibility channel that receives config
// updates. New code should prefer SubscribeConfigUpdates so callers can
// explicitly release their subscription.
func WatchConfigUpdates() <-chan *RouterConfig {
	return SubscribeConfigUpdates(1).Updates()
}
