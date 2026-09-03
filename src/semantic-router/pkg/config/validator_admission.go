package config

import (
	"fmt"
	"sort"
	"strings"
)

var admissionDeploymentKeys = map[string]bool{
	"prompt_guard":            true,
	"domain_classifier":       true,
	"pii_classifier":          true,
	"fact_check_classifier":   true,
	"hallucination_detector":  true,
	"hallucination_explainer": true,
	"feedback_detector":       true,
}

func validateModelAdmissionContracts(cfg *RouterConfig) error {
	for key, admission := range cfg.ModelAdmission {
		if !admissionDeploymentKeys[key] {
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
