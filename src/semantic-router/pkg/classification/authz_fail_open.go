package classification

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// applyAuthzFailOpenOnClassifyError clears Classify error when authz.fail_open is true and user ID
// is empty (e.g. Envoy stripped identity headers before ext_proc). Returns anonymous authz result.
func applyAuthzFailOpenOnClassifyError(failOpen bool, userID string, result *AuthzResult, err error) (*AuthzResult, error) {
	if err == nil {
		return result, nil
	}
	if failOpen && strings.TrimSpace(userID) == "" {
		logging.Warnf("[Authz Signal] empty user identity with authz.fail_open=true — continuing with no matched roles (anonymous): %v", err)
		return &AuthzResult{}, nil
	}
	return result, err
}
