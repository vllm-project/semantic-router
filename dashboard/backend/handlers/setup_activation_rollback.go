package handlers

import "fmt"

func restoreSetupRuntimeAfterFailure(configPath string, previous configFileSnapshot) error {
	if err := restoreConfigFileSnapshot(configPath, previous); err != nil {
		return fmt.Errorf("restore source config: %w", err)
	}
	if !previous.existed || len(previous.data) == 0 {
		return nil
	}

	effectiveConfigPath, err := syncRuntimeConfigForCurrentRuntime(configPath)
	if err != nil {
		return fmt.Errorf("source config restored, but previous runtime config sync failed: %w", err)
	}
	if err := restartSetupRuntimeServices(configPath, effectiveConfigPath); err != nil {
		return fmt.Errorf("source and runtime config restored, but previous runtime restart failed: %w", err)
	}
	return nil
}

func formatSetupActivationFailure(prefix string, activationErr error, restoreErr error) string {
	if restoreErr != nil {
		return fmt.Sprintf("%s: %v (failed to restore previous config: %v)", prefix, activationErr, restoreErr)
	}
	return fmt.Sprintf("%s: %v (previous config restored)", prefix, activationErr)
}
