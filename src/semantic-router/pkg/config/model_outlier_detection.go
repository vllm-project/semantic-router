package config

import (
	"fmt"
	"time"
)

// Defaults of a model_runtime deployment's outlier detection.
const (
	DefaultConsecutiveTimeouts = 3
	DefaultBaseEjectionMs      = 30_000
	DefaultMaxEjectionPercent  = 50
	maxEjectionMs              = 3_600_000
)

// DeploymentOutlierDetection takes a model_runtime replica whose exchanges
// keep running out of their deadline, the caller's or timeout_ms, out of
// dispatch for a while, as a provider model's consecutive_5xx does for a
// backend. An unset field keeps its default.
type DeploymentOutlierDetection struct {
	// ConsecutiveTimeouts is how many exchanges in a row that run out of
	// their deadline eject a replica (default 3); 0 turns ejection off. A
	// successful answer starts the count again.
	ConsecutiveTimeouts *int `yaml:"consecutive_timeouts,omitempty" json:"consecutive_timeouts,omitempty"`
	// BaseEjectionMs is how long a first ejection lasts (default 30000). Each
	// further one in a row lasts twice as long, up to MaxEjectionMs (default
	// ten times the base, at most an hour).
	BaseEjectionMs int `yaml:"base_ejection_ms,omitempty" json:"base_ejection_ms,omitempty"`
	MaxEjectionMs  int `yaml:"max_ejection_ms,omitempty" json:"max_ejection_ms,omitempty"`
	// MaxEjectionPercent caps the share of the deployment's replicas ejected
	// at once (default 50); one replica can always be ejected.
	MaxEjectionPercent int `yaml:"max_ejection_percent,omitempty" json:"max_ejection_percent,omitempty"`
}

// ReplicaEjection is a deployment's outlier detection with its defaults applied.
type ReplicaEjection struct {
	ConsecutiveTimeouts int
	BaseEjection        time.Duration
	MaxEjection         time.Duration
	MaxEjectionPercent  int
}

// Ejection returns the outlier detection the deployment's replica pool applies.
func (d ModelDeployment) Ejection() ReplicaEjection {
	ejection := ReplicaEjection{
		ConsecutiveTimeouts: DefaultConsecutiveTimeouts,
		BaseEjection:        DefaultBaseEjectionMs * time.Millisecond,
		MaxEjectionPercent:  DefaultMaxEjectionPercent,
	}
	if o := d.OutlierDetection; o != nil {
		if o.ConsecutiveTimeouts != nil {
			ejection.ConsecutiveTimeouts = *o.ConsecutiveTimeouts
		}
		if o.BaseEjectionMs > 0 {
			ejection.BaseEjection = time.Duration(o.BaseEjectionMs) * time.Millisecond
		}
		ejection.MaxEjection = time.Duration(o.MaxEjectionMs) * time.Millisecond
		if o.MaxEjectionPercent > 0 {
			ejection.MaxEjectionPercent = o.MaxEjectionPercent
		}
	}
	if ejection.MaxEjection == 0 {
		ejection.MaxEjection = min(10*ejection.BaseEjection, maxEjectionMs*time.Millisecond)
	}
	return ejection
}

func (d ModelDeployment) validateOutlierDetection() error {
	o := d.OutlierDetection
	if o == nil {
		return nil
	}
	if o.ConsecutiveTimeouts != nil && (*o.ConsecutiveTimeouts < 0 || *o.ConsecutiveTimeouts > 100) {
		return fmt.Errorf("outlier_detection.consecutive_timeouts must be between 0 and 100")
	}
	for _, field := range []struct {
		name  string
		value int
	}{{"base_ejection_ms", o.BaseEjectionMs}, {"max_ejection_ms", o.MaxEjectionMs}} {
		if field.value < 0 || field.value > maxEjectionMs {
			return fmt.Errorf("outlier_detection.%s must be between 0 and %d", field.name, maxEjectionMs)
		}
	}
	if base := d.Ejection().BaseEjection; o.MaxEjectionMs > 0 && time.Duration(o.MaxEjectionMs)*time.Millisecond < base {
		return fmt.Errorf("outlier_detection.max_ejection_ms must not be below base_ejection_ms (%d)", base.Milliseconds())
	}
	if o.MaxEjectionPercent < 0 || o.MaxEjectionPercent > 100 {
		return fmt.Errorf("outlier_detection.max_ejection_percent must be between 0 and 100")
	}
	return nil
}
