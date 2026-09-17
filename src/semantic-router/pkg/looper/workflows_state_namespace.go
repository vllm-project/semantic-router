package looper

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// workflowStateFileNameMaxBytes is Linux NAME_MAX. Redis keys stay unhashed.
const workflowStateFileNameMaxBytes = 255

var errWorkflowStateUnscoped = errors.New("workflow tool state is unscoped and cannot be resumed")

func normalizeWorkflowRecipeName(recipe config.RecipeName) config.RecipeName {
	name := strings.TrimSpace(string(recipe))
	if name == "" {
		return config.DefaultRecipeName
	}
	return config.RecipeName(name)
}

func workflowStateNamespace(recipe config.RecipeName) string {
	name := string(normalizeWorkflowRecipeName(recipe))
	return base64.RawURLEncoding.EncodeToString([]byte(name))
}

func workflowNamespacedStateID(recipe config.RecipeName, id string) (string, error) {
	if !validWorkflowStateID(id) {
		return "", fmt.Errorf("invalid workflow state id %q", id)
	}
	return workflowStateNamespace(recipe) + "__" + id, nil
}

// workflowStateStoreFileName is the on-disk name for one namespaced state.
// Namespaced IDs always contain "__"; hashed names never do, so the two
// forms cannot collide. Redis keys keep the namespaced ID unhashed.
func workflowStateStoreFileName(namespaced string) string {
	name := namespaced + ".json"
	if len(name) <= workflowStateFileNameMaxBytes {
		return name
	}
	sum := sha256.Sum256([]byte(namespaced))
	return "sha256-" + hex.EncodeToString(sum[:]) + ".json"
}

func workflowStateClaimable(state *workflowPendingToolState, recipe config.RecipeName) error {
	if state == nil {
		return fmt.Errorf("workflow tool state missing")
	}
	if strings.TrimSpace(state.RecipeName) == "" {
		return errWorkflowStateUnscoped
	}
	stored := normalizeWorkflowRecipeName(config.RecipeName(state.RecipeName))
	requested := normalizeWorkflowRecipeName(recipe)
	if stored != requested {
		return fmt.Errorf("workflow tool state belongs to recipe %q, not %q", stored, requested)
	}
	return nil
}
