package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

const (
	metaRoutingFeedbackRecorderName        = "meta_routing_feedback"
	metaRoutingFeedbackDefaultMaxRecords   = 1000
	metaRoutingFeedbackDefaultMaxBodyBytes = 64 * 1024
)

func createMetaRoutingFeedbackRecorder(cfg *config.RouterConfig) *routerreplay.Recorder {
	if cfg == nil || !cfg.MetaRouting.Enabled() {
		return nil
	}

	pluginCfg := &config.RouterReplayPluginConfig{
		Enabled:             true,
		MaxRecords:          metaRoutingFeedbackDefaultMaxRecords,
		CaptureRequestBody:  true,
		CaptureResponseBody: false,
		MaxBodyBytes:        metaRoutingFeedbackDefaultMaxBodyBytes,
	}

	recorder, err := createReplayRecorder(metaRoutingFeedbackRecorderName, pluginCfg, &cfg.RouterReplay)
	if err != nil {
		logging.Errorf("Failed to initialize meta-routing feedback recorder: %v", err)
		return nil
	}
	return recorder
}
