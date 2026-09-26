package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"sync"
	"time"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// vLLM does not answer until its model has loaded, which can take longer than
// the Router's own startup, so an unreachable backend is retried for about ten
// minutes.
var (
	servedContextWindowRetryInterval = 30 * time.Second
	servedContextWindowAttempts      = 20
)

type servedContextWindowTarget struct {
	model         string
	card          string
	source        string
	endpoint      string
	baseURL       string
	listPath      string
	upstreamModel string
	declared      int
	profile       *config.ProviderProfile
}

type servedModelList struct {
	Data []struct {
		ID          string `json:"id"`
		MaxModelLen int    `json:"max_model_len"`
	} `json:"data"`
}

// startServedContextWindowCheck reports vLLM backends that serve a smaller
// max_model_len than the context window routing uses. The model card stays the
// routing contract, so a mismatch is logged as a configuration error rather
// than substituted into eligibility or budgets.
func (r *OpenAIRouter) startServedContextWindowCheck() {
	if r == nil {
		return
	}
	targets := servedContextWindowTargets(r.Config)
	if len(targets) == 0 {
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	var probes sync.WaitGroup
	for _, target := range targets {
		probes.Go(func() { r.checkServedContextWindow(ctx, target) })
	}
	r.resources.add(func() error {
		cancel()
		probes.Wait()
		return nil
	})
}

func servedContextWindowTargets(cfg *config.RouterConfig) []servedContextWindowTarget {
	if cfg == nil {
		return nil
	}
	registry, err := modelcatalog.BuiltIn()
	if err != nil {
		return nil
	}
	var targets []servedContextWindowTarget
	for _, endpoint := range cfg.VLLMEndpoints {
		params, ok := cfg.ModelConfig[endpoint.Model]
		if !ok || params.ContextWindowSize <= 0 {
			continue
		}
		profile, err := cfg.GetProviderProfileForEndpoint(endpoint.Name)
		if err != nil || profile == nil {
			continue
		}
		provider, err := profile.ProviderType()
		if err != nil || provider != "vllm" {
			continue
		}
		base, err := url.Parse(profile.BaseURL)
		if err != nil {
			continue
		}
		listPath, err := registry.ResolveOperationPath(provider, profile.Protocol, "list_models", base.Path)
		if err != nil {
			continue
		}
		address, err := endpoint.ResolveAddress(cfg.ProviderProfiles)
		if err != nil {
			continue
		}
		card, source := endpoint.Model, ""
		if params.Catalog != "" {
			card = params.Catalog
		}
		if cfg.EffectiveModelRegistry != nil {
			if effective, found := cfg.EffectiveModelRegistry.Model(endpoint.Model); found {
				source = effective.Card.Provenance["limits.context_window_size"]
			}
		}
		targets = append(targets, servedContextWindowTarget{
			model:         endpoint.Model,
			card:          card,
			source:        source,
			endpoint:      endpoint.Name,
			baseURL:       providerEndpointScheme(cfg, endpoint.Name, profile) + "://" + address,
			listPath:      listPath,
			upstreamModel: cfg.ResolveExternalModelID(endpoint.Model, endpoint.Name),
			declared:      params.ContextWindowSize,
			profile:       profile,
		})
	}
	return targets
}

func (r *OpenAIRouter) checkServedContextWindow(ctx context.Context, target servedContextWindowTarget) {
	authorize, err := configuredProviderAuthorizer(r.Config, target.profile, target.model)
	if err != nil {
		logServedContextWindowUnverified(target, err)
		return
	}
	client, err := connector.New(target.baseURL, authorize, connector.Options{
		AttemptTimeout:   5 * time.Second,
		MaxRequestBytes:  1,
		MaxResponseBytes: 1 << 20,
		MaxErrorBytes:    1 << 10,
	})
	if err != nil {
		logServedContextWindowUnverified(target, err)
		return
	}
	defer func() { _ = client.Close() }()
	operation := connector.Operation{
		Name:              "served_context_window",
		Method:            http.MethodGet,
		Path:              target.listPath,
		SuccessStatusCode: http.StatusOK,
		RetrySafe:         true,
	}
	for attempt := 1; ; attempt++ {
		result, err := client.DoRequest(ctx, operation, connector.Request{Headers: target.profile.ExtraHeaders})
		if err == nil {
			reportServedContextWindow(target, result.Body)
			return
		}
		var failure *connector.Error
		if ctx.Err() != nil || !errors.As(err, &failure) || !failure.Retryable || attempt >= servedContextWindowAttempts {
			logServedContextWindowUnverified(target, err)
			return
		}
		timer := time.NewTimer(servedContextWindowRetryInterval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-timer.C:
		}
	}
}

func reportServedContextWindow(target servedContextWindowTarget, body []byte) {
	var list servedModelList
	if err := json.Unmarshal(body, &list); err != nil {
		logServedContextWindowUnverified(target, err)
		return
	}
	for _, served := range list.Data {
		if served.ID != target.upstreamModel {
			continue
		}
		if served.MaxModelLen > 0 && served.MaxModelLen < target.declared {
			logging.ComponentWarnEvent("extproc", "served_context_window_below_model_card", map[string]interface{}{
				"model":                 target.model,
				"model_card":            target.card,
				"context_window_size":   target.declared,
				"context_window_source": target.source,
				"endpoint":              target.endpoint,
				"upstream_model":        target.upstreamModel,
				"served_max_model_len":  served.MaxModelLen,
				"message": fmt.Sprintf(
					"vLLM at %s serves %s with max_model_len %d, below the %d-token context window that routing uses for model %q, so it can reject requests the Router admits. Align the deployment and model card: start vLLM with --max-model-len %d, or set context_window_size: %d on the routing.modelCards entry named %q.",
					target.baseURL, target.upstreamModel, served.MaxModelLen, target.declared, target.model,
					target.declared, served.MaxModelLen, target.card,
				),
			})
		}
		return
	}
	logServedContextWindowUnverified(target, fmt.Errorf("backend does not list model %q", target.upstreamModel))
}

func logServedContextWindowUnverified(target servedContextWindowTarget, err error) {
	logging.ComponentDebugEvent("extproc", "served_context_window_unverified", map[string]interface{}{
		"model":    target.model,
		"endpoint": target.endpoint,
		"error":    err.Error(),
	})
}
