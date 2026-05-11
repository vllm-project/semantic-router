/*
Copyright 2026 vLLM Semantic Router Contributors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1alpha1

import "fmt"

// validateSemanticRouter validates the SemanticRouter resource (platform spec:
// autoscaling, persistence, probes, ingress). Config contract semantic checks
// belong in dedicated helpers as the config surface grows.
func (r *SemanticRouter) validateSemanticRouter() error {
	if r.Spec.Autoscaling.Enabled != nil && *r.Spec.Autoscaling.Enabled {
		if err := r.validateAutoscaling(); err != nil {
			return err
		}
	}

	if err := r.validatePersistence(); err != nil {
		return err
	}

	if err := r.validateProbes(); err != nil {
		return err
	}

	if r.Spec.Ingress.Enabled != nil && *r.Spec.Ingress.Enabled {
		if err := r.validateIngress(); err != nil {
			return err
		}
	}

	return nil
}

func (r *SemanticRouter) validateAutoscaling() error {
	if r.Spec.Autoscaling.MinReplicas != nil && r.Spec.Autoscaling.MaxReplicas != nil {
		if *r.Spec.Autoscaling.MinReplicas > *r.Spec.Autoscaling.MaxReplicas {
			return fmt.Errorf("autoscaling.minReplicas (%d) must be less than or equal to maxReplicas (%d)",
				*r.Spec.Autoscaling.MinReplicas, *r.Spec.Autoscaling.MaxReplicas)
		}
	}

	if r.Spec.Autoscaling.TargetCPUUtilizationPercentage == nil &&
		r.Spec.Autoscaling.TargetMemoryUtilizationPercentage == nil {
		return fmt.Errorf("autoscaling requires at least one metric (targetCPUUtilizationPercentage or targetMemoryUtilizationPercentage)")
	}

	return nil
}

func (r *SemanticRouter) validatePersistence() error {
	if r.Spec.Persistence.Enabled != nil && *r.Spec.Persistence.Enabled {
		if r.Spec.Persistence.ExistingClaim != "" && r.Spec.Persistence.StorageClassName != "" {
			return fmt.Errorf("cannot specify both persistence.existingClaim and persistence.storageClassName")
		}
	}
	return nil
}

func probeValidationActive(p *ProbeSpec) bool {
	return p != nil && (p.Enabled == nil || *p.Enabled)
}

func (r *SemanticRouter) validateProbes() error {
	probes := []struct {
		name  string
		probe *ProbeSpec
	}{
		{name: "startupProbe", probe: r.Spec.StartupProbe},
		{name: "livenessProbe", probe: r.Spec.LivenessProbe},
		{name: "readinessProbe", probe: r.Spec.ReadinessProbe},
	}
	for _, item := range probes {
		if !probeValidationActive(item.probe) {
			continue
		}
		if err := r.validateProbeSpec(item.name, item.probe); err != nil {
			return err
		}
	}
	return nil
}

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

func (r *SemanticRouter) validateIngress() error {
	if len(r.Spec.Ingress.Hosts) == 0 {
		return fmt.Errorf("ingress.hosts must be specified when ingress is enabled")
	}

	for i, host := range r.Spec.Ingress.Hosts {
		if host.Host == "" {
			return fmt.Errorf("ingress.hosts[%d].host cannot be empty", i)
		}
		if len(host.Paths) == 0 {
			return fmt.Errorf("ingress.hosts[%d].paths must have at least one path", i)
		}
	}

	return nil
}
