package v1alpha1

import "fmt"

func (r *SemanticRouter) validateProbeConfigs() error {
	return r.validateProbes()
}

type namedProbeSpec struct {
	name  string
	probe *ProbeSpec
}

// validateProbes validates probe configurations.
func (r *SemanticRouter) validateProbes() error {
	for _, probe := range r.enabledProbeSpecs() {
		if err := r.validateProbeSpec(probe.name, probe.probe); err != nil {
			return err
		}
	}

	return nil
}

func (r *SemanticRouter) enabledProbeSpecs() []namedProbeSpec {
	probes := make([]namedProbeSpec, 0, 3)
	probes = appendEnabledProbe(probes, "startupProbe", r.Spec.StartupProbe)
	probes = appendEnabledProbe(probes, "livenessProbe", r.Spec.LivenessProbe)
	probes = appendEnabledProbe(probes, "readinessProbe", r.Spec.ReadinessProbe)
	return probes
}

func appendEnabledProbe(
	probes []namedProbeSpec,
	name string,
	probe *ProbeSpec,
) []namedProbeSpec {
	if probe == nil || (probe.Enabled != nil && !*probe.Enabled) {
		return probes
	}
	return append(probes, namedProbeSpec{name: name, probe: probe})
}

// validateProbeSpec validates a single probe specification.
func (r *SemanticRouter) validateProbeSpec(name string, probe *ProbeSpec) error {
	if probe.TimeoutSeconds != nil && *probe.TimeoutSeconds <= 0 {
		return fmt.Errorf("%s.timeoutSeconds must be greater than 0", name)
	}
	if probe.PeriodSeconds != nil && *probe.PeriodSeconds <= 0 {
		return fmt.Errorf("%s.periodSeconds must be greater than 0", name)
	}
	if probe.FailureThreshold != nil && *probe.FailureThreshold <= 0 {
		return fmt.Errorf("%s.failureThreshold must be greater than 0", name)
	}
	return nil
}
