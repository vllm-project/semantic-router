package modelservice

import (
	"context"
	"fmt"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// DecisionMedia contains validated data URLs for one deployment. It is ephemeral
// request data and must never be included in diagnostics or replay records.
type (
	DecisionMedia    struct{ Images, Videos []string }
	decisionMediaKey struct{}
)

// WithDecisionMedia scopes attachments to the routing stage. Callers must not
// reuse this context for unrelated response-time or tool judgments.
func WithDecisionMedia(ctx context.Context, media map[string]DecisionMedia) context.Context {
	if len(media) == 0 {
		return ctx
	}
	copied := make(map[string]DecisionMedia, len(media))
	for name, value := range media {
		copied[name] = DecisionMedia{slices.Clone(value.Images), slices.Clone(value.Videos)}
	}
	return context.WithValue(ctx, decisionMediaKey{}, copied)
}

func attachDecisionMedia(ctx context.Context, deployment string, request Request) Request {
	media, _ := ctx.Value(decisionMediaKey{}).(map[string]DecisionMedia)
	if selected, ok := media[deployment]; ok {
		request.Images, request.Videos = slices.Clone(selected.Images), slices.Clone(selected.Videos)
	}
	return request
}

// ValidateDecisionMedia checks opt-in against the actual prepared runtime card.
func ValidateDecisionMedia(deployment config.ModelDeployment, card ModelCard) error {
	if err := deployment.ValidateDecisionMedia(); err != nil {
		return err
	}
	policy := deployment.Input.Media
	if policy == nil {
		return nil
	}
	if !card.Serves("decisions") {
		return fmt.Errorf("input.media requires the decisions surface")
	}
	for _, entry := range []struct {
		kind  string
		limit *config.DecisionMediaLimit
		bytes int
	}{
		{"image", policy.Images, card.ImageMaxBytes}, {"video", policy.Videos, card.VideoMaxBytes},
	} {
		if entry.limit == nil {
			continue
		}
		if !slices.Contains(card.Modalities, entry.kind) {
			return fmt.Errorf("input.media: model card does not support %s", entry.kind)
		}
		if entry.bytes <= 0 {
			return fmt.Errorf("input.media: model card has no %s byte limit", entry.kind)
		}
		if entry.limit.MaxBytes > entry.bytes {
			return fmt.Errorf("input.media: %s max_bytes exceeds model card limit %d", entry.kind, entry.bytes)
		}
	}
	return nil
}
