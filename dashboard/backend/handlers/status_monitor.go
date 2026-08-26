package handlers

import (
	"context"
	"log"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/statusstore"
)

const statusObservationInterval = time.Minute

// StatusMonitor owns server-side status sampling. Browser traffic reads the
// durable history but never determines whether an hour was observed.
type StatusMonitor struct {
	routerAPIURL string
	historyStore *statusstore.Store
	interval     time.Duration
	stop         chan struct{}
	done         chan struct{}
	startOnce    sync.Once
	closeOnce    sync.Once
}

// NewStatusMonitor creates a monitor for the public Dashboard status surface.
func NewStatusMonitor(routerAPIURL string, historyStore *statusstore.Store) *StatusMonitor {
	return &StatusMonitor{
		routerAPIURL: routerAPIURL,
		historyStore: historyStore,
		interval:     statusObservationInterval,
		stop:         make(chan struct{}),
		done:         make(chan struct{}),
	}
}

// Start begins immediate sampling followed by bounded periodic observations.
func (m *StatusMonitor) Start() {
	m.startOnce.Do(func() {
		go m.run()
	})
}

// Close stops the sampler before its SQLite store is closed.
func (m *StatusMonitor) Close() error {
	m.closeOnce.Do(func() { close(m.stop) })
	select {
	case <-m.done:
	case <-time.After(10 * time.Second):
		return context.DeadlineExceeded
	}
	return nil
}

func (m *StatusMonitor) run() {
	defer close(m.done)
	m.sample()
	ticker := time.NewTicker(m.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			m.sample()
		case <-m.stop:
			return
		}
	}
}

func (m *StatusMonitor) sample() {
	if m.historyStore == nil {
		return
	}
	status := detectSystemStatus(m.routerAPIURL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := m.historyStore.Record(ctx, statusObservations(status.Services)); err != nil {
		log.Printf("status history observation failed: %v", err)
	}
}

func statusObservations(services []ServiceStatus) []statusstore.Observation {
	observations := make([]statusstore.Observation, 0, len(services))
	for _, service := range services {
		observations = append(observations, statusstore.Observation{
			Service: service.Name,
			State:   statusstore.State(service.Status),
		})
	}
	return observations
}
