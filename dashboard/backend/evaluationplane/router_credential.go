package evaluationplane

import (
	"crypto/subtle"
	"errors"
	"fmt"
	"os"
	"strings"
)

func resolveRouterAuthentication(
	evaluationEnv string,
	provider CredentialProvider,
) (bool, error) {
	evaluationEnv = strings.TrimSpace(evaluationEnv)
	if evaluationEnv != "" {
		if !secretEnvPattern.MatchString(evaluationEnv) {
			return false, fmt.Errorf("evaluation Router credential reference must be an uppercase environment variable name")
		}
		if evaluationEnv == routerManagementCredentialEnv {
			return false, fmt.Errorf("evaluation Router credential cannot reuse the Dashboard management credential")
		}
	}

	managementToken, managementErr := managementCredential(provider)
	authRequired := managementToken != "" || (managementErr != nil && !errors.Is(managementErr, os.ErrNotExist))
	if evaluationEnv == "" {
		return authRequired, nil
	}
	evaluationToken, present := os.LookupEnv(evaluationEnv)
	evaluationToken = strings.TrimSpace(evaluationToken)
	if !present || evaluationToken == "" {
		return false, fmt.Errorf("dedicated Router evaluation credential is unavailable")
	}
	if managementErr != nil && !errors.Is(managementErr, os.ErrNotExist) {
		return false, fmt.Errorf("verify dedicated Router evaluation credential isolation: %w", managementErr)
	}
	if managementToken != "" && subtle.ConstantTimeCompare([]byte(evaluationToken), []byte(managementToken)) == 1 {
		return false, fmt.Errorf("evaluation Router credential must be distinct from the Dashboard management credential")
	}
	return authRequired, nil
}

func managementCredential(provider CredentialProvider) (string, error) {
	if provider == nil {
		return "", os.ErrNotExist
	}
	token, err := provider.ManagementCredential()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(token), nil
}
