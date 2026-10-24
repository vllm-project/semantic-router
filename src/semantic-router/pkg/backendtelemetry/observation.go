package backendtelemetry

import (
	"errors"
	"fmt"
	"strings"
	"time"
)

// SchemaVersion is the Observation contract version this package accepts.
const SchemaVersion = 1

// Engine names the serving engine that runs a replica. It selects the adapter
// that knows how to read the engine's metrics.
type Engine string

const (
	EngineVLLM        Engine = "vllm"
	EngineSGLang      Engine = "sglang"
	EngineATOM        Engine = "atom"
	EngineTensorRTLLM Engine = "tensorrt-llm"
)

var knownEngines = []Engine{EngineVLLM, EngineSGLang, EngineATOM, EngineTensorRTLLM}

// Engines lists the engines this contract recognizes.
func Engines() []Engine { return append([]Engine(nil), knownEngines...) }

// ParseEngine accepts a configured engine name.
func ParseEngine(name string) (Engine, error) {
	for _, engine := range knownEngines {
		if name == string(engine) {
			return engine, nil
		}
	}
	return "", fmt.Errorf("unknown engine %q (want one of %s)", name, joinEngines())
}

func joinEngines() string {
	names := make([]string, len(knownEngines))
	for i, engine := range knownEngines {
		names[i] = string(engine)
	}
	return strings.Join(names, ", ")
}

// Health is an adapter's view of whether a replica can serve. The zero value
// is HealthUnknown.
type Health int

const (
	HealthUnknown Health = iota
	HealthHealthy
	HealthDegraded
	HealthUnhealthy
)

func (h Health) String() string {
	switch h {
	case HealthUnknown:
		return "unknown"
	case HealthHealthy:
		return "healthy"
	case HealthDegraded:
		return "degraded"
	case HealthUnhealthy:
		return "unhealthy"
	}
	return fmt.Sprintf("Health(%d)", int(h))
}

// Identity names one serving replica. Model and BackendRef locate it in the
// router configuration (providers.models[].name and backend_refs[].name);
// Replica distinguishes members when one backend ref resolves to several.
type Identity struct {
	Model      string
	BackendRef string
	Replica    string
	Engine     Engine
}

func (id Identity) String() string {
	s := id.Model + "/" + id.BackendRef
	if id.Replica != "" {
		s += "/" + id.Replica
	}
	return s
}

func (id Identity) validate() error {
	for _, field := range []struct{ name, value string }{
		{"model", id.Model},
		{"backend ref", id.BackendRef},
	} {
		if field.value == "" {
			return fmt.Errorf("%s is required", field.name)
		}
	}
	for _, field := range []struct{ name, value string }{
		{"model", id.Model},
		{"backend ref", id.BackendRef},
		{"replica", id.Replica},
	} {
		if strings.TrimSpace(field.value) != field.value {
			return fmt.Errorf("%s %q must not have surrounding spaces", field.name, field.value)
		}
	}
	_, err := ParseEngine(string(id.Engine))
	return err
}

// Observation is one replica's state as an adapter saw it at CollectedAt.
// Every metric is optional; see Value.
type Observation struct {
	Schema   int
	Identity Identity
	// Source names the adapter and the surface it read, such as
	// "vllm/prometheus", so diagnostics can say where evidence came from.
	Source      string
	CollectedAt time.Time
	// TTL bounds how long the observation stays fresh. Zero uses the store's
	// default.
	TTL time.Duration

	Health Health

	QueueDepth     Count
	ActiveRequests Count
	KVCacheUsage   Ratio
	// PrefixCacheHitRate is the hit fraction over the latency window.
	PrefixCacheHitRate Ratio

	// TTFT, TPOT and E2E are means over LatencyWindow, derived from counter
	// deltas between two scrapes. LatencyWindow is required when any of them
	// is present.
	TTFT          Latency
	TPOT          Latency
	E2E           Latency
	LatencyWindow time.Duration
}

// ErrInvalidObservation wraps every validation failure.
var ErrInvalidObservation = errors.New("invalid backend observation")

// Validate checks the contract. It never repairs a field: an adapter that
// cannot fill a value must leave it absent.
func (o Observation) Validate() error {
	if err := o.validate(); err != nil {
		return fmt.Errorf("%w %s: %w", ErrInvalidObservation, o.Identity, err)
	}
	return nil
}

func (o Observation) validate() error {
	if o.Schema != SchemaVersion {
		return fmt.Errorf("schema %d is not supported (want %d)", o.Schema, SchemaVersion)
	}
	if err := o.Identity.validate(); err != nil {
		return err
	}
	if o.Source == "" {
		return errors.New("source is required")
	}
	if o.CollectedAt.IsZero() {
		return errors.New("collected_at is required")
	}
	if o.TTL < 0 {
		return errors.New("ttl must not be negative")
	}
	if o.Health < HealthUnknown || o.Health > HealthUnhealthy {
		return fmt.Errorf("health %d is not defined", int(o.Health))
	}
	if err := o.validateMetrics(); err != nil {
		return err
	}
	return o.validateLatency()
}

func (o Observation) validateMetrics() error {
	for _, c := range []struct {
		name  string
		value Count
	}{
		{"queue_depth", o.QueueDepth},
		{"active_requests", o.ActiveRequests},
	} {
		if v, ok := c.value.Get(); ok && v < 0 {
			return fmt.Errorf("%s must not be negative", c.name)
		}
	}
	for _, r := range []struct {
		name  string
		value Ratio
	}{
		{"kv_cache_usage", o.KVCacheUsage},
		{"prefix_cache_hit_rate", o.PrefixCacheHitRate},
	} {
		// The negated range also rejects NaN.
		if v, ok := r.value.Get(); ok && !(v >= 0 && v <= 1) {
			return fmt.Errorf("%s must be within [0, 1]", r.name)
		}
	}
	return nil
}

func (o Observation) validateLatency() error {
	present := false
	for _, l := range []struct {
		name  string
		value Latency
	}{
		{"ttft", o.TTFT},
		{"tpot", o.TPOT},
		{"e2e", o.E2E},
	} {
		v, ok := l.value.Get()
		if !ok {
			continue
		}
		if v < 0 {
			return fmt.Errorf("%s must not be negative", l.name)
		}
		present = true
	}
	if o.LatencyWindow < 0 {
		return errors.New("latency_window must not be negative")
	}
	if present && o.LatencyWindow == 0 {
		return errors.New("latency_window is required with a latency")
	}
	return nil
}
