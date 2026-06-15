package classification

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func newContrastivePreferenceClassifier(
	rules []config.PreferenceRule,
	resolvedLocalCfg config.PreferenceModelConfig,
) (*PreferenceClassifier, error) {
	contrastive, err := NewContrastivePreferenceClassifierWithConfig(
		rules,
		resolvedLocalCfg.EmbeddingModel,
		resolvedLocalCfg.PrototypeScoring,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize contrastive preference classifier: %w", err)
	}

	return &PreferenceClassifier{
		preferenceRules: rules,
		contrastive:     contrastive,
		useContrastive:  true,
		timeout:         30 * time.Second,
	}, nil
}

// classifyContrastive runs few-shot contrastive routing using embeddings.
func (p *PreferenceClassifier) classifyContrastive(conversationJSON string) (*PreferenceResult, error) {
	if p.contrastive == nil {
		return nil, fmt.Errorf("contrastive classifier is not initialized")
	}

	start := time.Now()
	result, err := p.contrastive.Classify(preferenceConversationText(conversationJSON))
	if err != nil {
		return nil, err
	}

	logging.Infof("Preference contrastive classification: preference=%s, latency=%.3fs",
		result.Preference, time.Since(start).Seconds())
	return result, nil
}

func preferenceConversationText(conversationJSON string) string {
	text := conversationJSON

	var messages []struct {
		Content string `json:"content"`
	}

	if err := json.Unmarshal([]byte(conversationJSON), &messages); err == nil && len(messages) > 0 {
		var sb strings.Builder
		for _, msg := range messages {
			if strings.TrimSpace(msg.Content) == "" {
				continue
			}
			if sb.Len() > 0 {
				sb.WriteString("\n")
			}
			sb.WriteString(msg.Content)
		}

		if sb.Len() > 0 {
			text = sb.String()
		}
	}

	return text
}
