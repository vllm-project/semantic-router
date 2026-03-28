package extproc

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// resolveImageGenConfig returns the image generation plugin config used by the
// selected modality decision. Canonical decision-level image_gen plugin config
// is authoritative; legacy model_config.image_gen_backend remains as a runtime
// fallback for older modality-routing fixtures. Prompt prefixes continue to come
// from the router-level modality detector config; the decision plugin only owns
// image-generation backend behavior.
func resolveImageGenConfig(cfg *config.RouterConfig, decision *config.Decision, diffusionModel string) (*config.ImageGenPluginConfig, []string, error) {
	if decision != nil {
		if pluginCfg := decision.GetImageGenConfig(); pluginCfg != nil && pluginCfg.Enabled {
			return pluginCfg, cfg.ModalityDetector.PromptPrefixes, nil
		}
	}

	backendEntry, err := resolveDiffusionBackend(cfg, diffusionModel)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to resolve diffusion backend: %w", err)
	}

	return backendEntry.ToPluginConfig(), cfg.ModalityDetector.PromptPrefixes, nil
}
