package looper

import (
	"errors"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

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
	var b strings.Builder
	b.Grow(len(name))
	for _, ch := range name {
		if (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '-' || ch == '_' {
			b.WriteRune(ch)
			continue
		}
		b.WriteByte('_')
	}
	if b.Len() == 0 {
		return string(config.DefaultRecipeName)
	}
	return b.String()
}

func workflowNamespacedStateID(recipe config.RecipeName, id string) (string, error) {
	if !validWorkflowStateID(id) {
		return "", fmt.Errorf("invalid workflow state id %q", id)
	}
	return workflowStateNamespace(recipe) + "__" + id, nil
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
