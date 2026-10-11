package config

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

var admissionDeploymentKeys = map[string]bool{
	"safety":                 true,
	"hazard":                 true,
	"prompt_guard":           true,
	"domain_classifier":      true,
	"pii_classifier":         true,
	"fact_check_classifier":  true,
	"hallucination_detector": true,
	"feedback_detector":      true,
}

func validateModelAdmissionContracts(cfg *RouterConfig) error {
	for key, admission := range cfg.ModelAdmission {
		_, declared := cfg.ModelDeployments[key]
		if !admissionDeploymentKeys[key] && !declared {
			return fmt.Errorf(
				"global.model_catalog.admission: unknown deployment %q; supported deployments: %s",
				key,
				strings.Join(sortedAdmissionDeploymentKeys(), ", "),
			)
		}
		if err := validateAdmissionConfig(key, admission); err != nil {
			return err
		}
	}
	return nil
}

func validateAdmissionConfig(key string, admission AdmissionConfig) error {
	if admission.MaxConcurrency < 1 {
		return fmt.Errorf("global.model_catalog.admission.%s: max_concurrency must be >= 1", key)
	}
	if admission.MaxQueue < 0 {
		return fmt.Errorf("global.model_catalog.admission.%s: max_queue must be >= 0", key)
	}
	if admission.QueueTimeoutMs < 0 {
		return fmt.Errorf("global.model_catalog.admission.%s: queue_timeout_ms must be >= 0", key)
	}
	switch admission.OnOverflow {
	case "", "shed", "wait", "fail_open":
	default:
		return fmt.Errorf(
			"global.model_catalog.admission.%s: on_overflow must be shed, wait, or fail_open",
			key,
		)
	}
	if admission.OnOverflow == "wait" && admission.MaxQueue < 1 {
		return fmt.Errorf(
			"global.model_catalog.admission.%s: on_overflow wait requires max_queue >= 1",
			key,
		)
	}
	return nil
}

func sortedAdmissionDeploymentKeys() []string {
	keys := make([]string, 0, len(admissionDeploymentKeys))
	for key := range admissionDeploymentKeys {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

// maxSignalTimeoutMs is the longest model-signal deadline a recipe may set.
const maxSignalTimeoutMs = 3_600_000

// DefaultSignalTimeout bounds the model-runtime signals of a request that has
// no deadline of its own: a served request, whose gateway timeout (Envoy's
// ext_proc message_timeout, 60 s in the shipped gateways) the Router does not
// see.
const DefaultSignalTimeout = 45 * time.Second

func validateModelSignalTimeoutContracts(cfg *RouterConfig) error {
	if cfg.ModelSignalTimeoutMs < 0 || cfg.ModelSignalTimeoutMs > maxSignalTimeoutMs {
		return fmt.Errorf("global.model_catalog.signal_timeout_ms must be between 0 and %d", maxSignalTimeoutMs)
	}
	return nil
}

// SignalTimeout is the configured deadline of a request's model-runtime
// signals; 0 derives it from the request's deadline.
func (c *RouterConfig) SignalTimeout() time.Duration {
	if c == nil {
		return 0
	}
	return time.Duration(c.ModelSignalTimeoutMs) * time.Millisecond
}
