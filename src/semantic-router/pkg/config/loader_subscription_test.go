package config

import "testing"

func TestSubscribeConfigUpdatesFanout(t *testing.T) {
	configUpdateMu.Lock()
	previousSubscribers := configUpdateSubscribers
	previousNextID := configUpdateNextID
	configUpdateSubscribers = map[uint64]chan *RouterConfig{}
	configUpdateNextID = 0
	configUpdateMu.Unlock()
	t.Cleanup(func() {
		configUpdateMu.Lock()
		configUpdateSubscribers = previousSubscribers
		configUpdateNextID = previousNextID
		configUpdateMu.Unlock()
	})

	subA := SubscribeConfigUpdates(1)
	subB := SubscribeConfigUpdates(1)
	defer subA.Close()
	defer subB.Close()

	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Decisions: []Decision{{Name: "support"}},
		},
	}

	Replace(cfg)

	select {
	case got := <-subA.Updates():
		if got != cfg {
			t.Fatalf("subA config = %p, want %p", got, cfg)
		}
	default:
		t.Fatal("subA did not receive config update")
	}

	select {
	case got := <-subB.Updates():
		if got != cfg {
			t.Fatalf("subB config = %p, want %p", got, cfg)
		}
	default:
		t.Fatal("subB did not receive config update")
	}
}
