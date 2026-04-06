package v1alpha1

import "fmt"

func (r *SemanticRouter) validateIngressConfig() error {
	if r.Spec.Ingress.Enabled == nil || !*r.Spec.Ingress.Enabled {
		return nil
	}
	return r.validateIngress()
}

// validateIngress validates ingress configuration.
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
