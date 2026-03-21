package classification

import (
	"sync"
	"sync/atomic"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

var classifierSignalLoadGates sync.Map

type signalLoadGate struct {
	threshold int64
	slots     chan struct{}
	active    atomic.Int64
}

func newSignalLoadGate(cfg config.BatchClassificationConfig) *signalLoadGate {
	maxConcurrency := cfg.MaxConcurrency
	if maxConcurrency <= 0 {
		return nil
	}

	threshold := cfg.ConcurrencyThreshold
	if threshold <= 0 || threshold > maxConcurrency {
		threshold = maxConcurrency
	}

	bypassAllowance := threshold - 1
	gatedSlots := maxConcurrency - bypassAllowance
	if gatedSlots < 1 {
		gatedSlots = 1
	}

	logging.Infof(
		"Signal evaluation load gate enabled: threshold=%d max_concurrency=%d gated_slots=%d",
		threshold,
		maxConcurrency,
		gatedSlots,
	)

	return &signalLoadGate{
		threshold: int64(threshold),
		slots:     make(chan struct{}, gatedSlots),
	}
}

func (g *signalLoadGate) enter() func() {
	if g == nil {
		return func() {}
	}

	active := g.active.Add(1)
	if active >= g.threshold {
		g.slots <- struct{}{}
		return func() {
			<-g.slots
			g.active.Add(-1)
		}
	}

	return func() {
		g.active.Add(-1)
	}
}

func (c *Classifier) enterSignalEvaluationLoadGate() func() {
	if c == nil || c.Config == nil {
		return func() {}
	}

	if existing, ok := classifierSignalLoadGates.Load(c); ok {
		gate, _ := existing.(*signalLoadGate)
		if gate == nil {
			return func() {}
		}
		return gate.enter()
	}

	gate := newSignalLoadGate(c.Config.API.BatchClassification)
	if gate == nil {
		return func() {}
	}

	actual, _ := classifierSignalLoadGates.LoadOrStore(c, gate)
	loadedGate, _ := actual.(*signalLoadGate)
	if loadedGate == nil {
		return func() {}
	}
	return loadedGate.enter()
}
