package evaluationplane

import "fmt"

func validateControlledPairRegistryTargets(
	registry *Registry,
	baseline RunManifest,
	candidate RunManifest,
) error {
	contracts := registry.executionContracts()
	for _, arm := range []struct {
		role     string
		manifest RunManifest
	}{
		{role: "baseline", manifest: baseline},
		{role: "candidate", manifest: candidate},
	} {
		if _, err := contracts.resolve(arm.manifest); err != nil {
			return fmt.Errorf(
				"%w: controlled pair %s target no longer matches the active deployment registry: %w",
				ErrConflict,
				arm.role,
				err,
			)
		}
	}
	return nil
}
