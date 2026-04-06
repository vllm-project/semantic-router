package v1alpha1

import "fmt"

func (r *SemanticRouter) validatePersistenceConfig() error {
	return r.validatePersistence()
}

// validatePersistence validates PVC configuration.
func (r *SemanticRouter) validatePersistence() error {
	if r.Spec.Persistence.Enabled != nil && *r.Spec.Persistence.Enabled {
		if r.Spec.Persistence.ExistingClaim != "" && r.Spec.Persistence.StorageClassName != "" {
			return fmt.Errorf("cannot specify both persistence.existingClaim and persistence.storageClassName")
		}
	}
	return nil
}
