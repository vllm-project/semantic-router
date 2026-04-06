package v1alpha1

import "fmt"

func (r *SemanticRouter) validateProbeConfigs() error {
	return r.validateProbes()
}

// validateProbes validates probe configurations.
func (r *SemanticRouter) validateProbes() error {
	if r.Spec.StartupProbe != nil && (r.Spec.StartupProbe.Enabled == nil || *r.Spec.StartupProbe.Enabled) {
		if err := r.validateProbeSpec("startupProbe", r.Spec.StartupProbe); err != nil {
			return err
		}
	}

	if r.Spec.LivenessProbe != nil && (r.Spec.LivenessProbe.Enabled == nil || *r.Spec.LivenessProbe.Enabled) {
		if err := r.validateProbeSpec("livenessProbe", r.Spec.LivenessProbe); err != nil {
			return err
		}
	}

	if r.Spec.ReadinessProbe != nil && (r.Spec.ReadinessProbe.Enabled == nil || *r.Spec.ReadinessProbe.Enabled) {
		if err := r.validateProbeSpec("readinessProbe", r.Spec.ReadinessProbe); err != nil {
			return err
		}
	}

	return nil
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
