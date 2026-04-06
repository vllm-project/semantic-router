package v1alpha1

import "fmt"

func (r *SemanticRouter) validateAutoscalingConfig() error {
	if r.Spec.Autoscaling.Enabled == nil || !*r.Spec.Autoscaling.Enabled {
		return nil
	}
	return r.validateAutoscaling()
}

// validateAutoscaling validates HPA configuration.
func (r *SemanticRouter) validateAutoscaling() error {
	if r.Spec.Autoscaling.MinReplicas != nil && r.Spec.Autoscaling.MaxReplicas != nil {
		if *r.Spec.Autoscaling.MinReplicas > *r.Spec.Autoscaling.MaxReplicas {
			return fmt.Errorf(
				"autoscaling.minReplicas (%d) must be less than or equal to maxReplicas (%d)",
				*r.Spec.Autoscaling.MinReplicas,
				*r.Spec.Autoscaling.MaxReplicas,
			)
		}
	}

	if r.Spec.Autoscaling.TargetCPUUtilizationPercentage == nil &&
		r.Spec.Autoscaling.TargetMemoryUtilizationPercentage == nil {
		return fmt.Errorf(
			"autoscaling requires at least one metric (targetCPUUtilizationPercentage or targetMemoryUtilizationPercentage)",
		)
	}

	return nil
}
