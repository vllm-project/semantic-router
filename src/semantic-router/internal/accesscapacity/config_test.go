package accesscapacity

import (
	"testing"
	"time"
)

func TestConfigValidation(t *testing.T) {
	config := DefaultConfig()
	if err := config.Validate(); err != nil {
		t.Fatalf("DefaultConfig().Validate() error = %v", err)
	}
	for name, mutate := range map[string]func(*Config){
		"one replica":     func(value *Config) { value.Replicas = 1 },
		"low concurrency": func(value *Config) { value.Concurrency = value.Replicas - 1 },
		"short timeout":   func(value *Config) { value.OperationTimeout = time.Second },
		"invalid prefix":  func(value *Config) { value.KeyPrefix = "capacity::bad" },
		"missing output":  func(value *Config) { value.OutputRoot = " " },
		"empty threshold": func(value *Config) { value.Thresholds.MaxAdmissionP99 = 0 },
	} {
		t.Run(name, func(t *testing.T) {
			candidate := config
			mutate(&candidate)
			if err := candidate.Validate(); err == nil {
				t.Fatal("Validate() accepted invalid capacity configuration")
			}
		})
	}
}

func TestNearestRankLatencyIsDeterministic(t *testing.T) {
	values := []time.Duration{
		9 * time.Millisecond, time.Millisecond, 3 * time.Millisecond, 2 * time.Millisecond,
		8 * time.Millisecond, 4 * time.Millisecond, 7 * time.Millisecond, 5 * time.Millisecond,
		10 * time.Millisecond, 6 * time.Millisecond,
	}
	result := latency(values)
	if result.Count != 10 || result.P50MS != 5 || result.P95MS != 10 || result.P99MS != 10 || result.MaxMS != 10 {
		t.Fatalf("latency() = %+v", result)
	}
}
