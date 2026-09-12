package services

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func domainResultService(t *testing.T, handler http.Handler, configure func(*config.RouterConfig)) *ClassificationService {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	host, portText, err := net.SplitHostPort(server.Listener.Addr().String())
	require.NoError(t, err)
	port, err := strconv.Atoi(portText)
	require.NoError(t, err)
	cfg := &config.RouterConfig{}
	cfg.CategoryModel.Backend = &config.RemoteClassifierBackend{Protocol: config.RemoteClassifierProtocolHTTPClassify, Contract: config.RemoteClassifierContractLabelDistribution, Model: "domain-test"}
	cfg.CategoryModel.CategoryMappingPath = "injected-mapping"
	cfg.CategoryModel.Threshold = 0.9
	cfg.ExternalModels = []config.ExternalModelConfig{{Name: "domain-test", ModelRole: config.ModelRoleClassification, ModelName: "domain-test", ModelEndpoint: config.ClassifierVLLMEndpoint{Address: host, Port: port, Protocol: "http"}}}
	cfg.Categories = []config.Category{{CategoryMetadata: config.CategoryMetadata{Name: "math"}}, {CategoryMetadata: config.CategoryMetadata{Name: "other"}}}
	cfg.Decisions = []config.Decision{{Name: "math-route", Rules: config.RuleNode{Type: config.SignalTypeDomain, Name: "math"}}}
	if configure != nil {
		configure(cfg)
	}
	mapping := &classification.CategoryMapping{CategoryToIdx: map[string]int{"math": 0, "other": 1}, IdxToCategory: map[string]string{"0": "math", "1": "other"}}
	classifier, err := classification.NewClassifier(cfg, mapping, nil, nil)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, classifier.Close()) })
	service := NewClassificationService(classifier, cfg)
	t.Cleanup(func() { require.NoError(t, service.Close()) })
	return service
}

func writeDomainDistribution(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`[{"label":"math","score":0.75},{"label":"other","score":0.25}]`))
}

func assertIntentScore(t *testing.T, response *IntentResponse, category string, scored bool) {
	t.Helper()
	require.NotNil(t, response)
	require.Equal(t, category, response.Classification.Category)
	raw, err := json.Marshal(response)
	require.NoError(t, err)
	var decoded map[string]interface{}
	require.NoError(t, json.Unmarshal(raw, &decoded))
	result := decoded["classification"].(map[string]interface{})
	require.Equal(t, scored, decoded["probabilities_available"], string(raw))
	require.Equal(t, scored, result["confidence_available"], string(raw))
	if scored {
		require.Equal(t, 0.75, result["confidence"])
		require.Equal(t, map[string]interface{}{category: 0.75}, decoded["probabilities"])
	} else {
		require.Nil(t, result["confidence"], string(raw))
		require.NotContains(t, decoded, "probabilities", string(raw))
	}
}

func TestIntentReusesEvaluatedDomainWithoutRetry(t *testing.T) {
	for _, test := range []struct {
		name    string
		failure bool
		policy  config.UnknownPolicy
	}{
		{name: "success_below_match_threshold"},
		{name: "error_default", failure: true},
		{name: "error_no_match", failure: true, policy: config.RuleOnUnknownNoMatch},
		{name: "error_match", failure: true, policy: config.RuleOnUnknownMatch},
		{name: "error_fail_request", failure: true, policy: config.RuleOnUnknownFailRequest},
	} {
		t.Run(test.name, func(t *testing.T) {
			var calls atomic.Int32
			var failing atomic.Bool
			service := domainResultService(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				if failing.Load() {
					http.Error(w, "unavailable", http.StatusBadRequest)
					return
				}
				writeDomainDistribution(w)
			}), func(cfg *config.RouterConfig) { cfg.Decisions[0].Rules.OnUnknown = test.policy })
			calls.Store(0)
			failing.Store(test.failure)
			response, err := service.ClassifyIntent(context.Background(), IntentRequest{Text: "one evaluation", Options: &IntentOptions{ReturnProbabilities: true}})
			assert.Equal(t, int32(1), calls.Load(), "domain was classified again after signal evaluation")
			if test.policy == config.RuleOnUnknownFailRequest {
				require.ErrorIs(t, err, decision.ErrDecisionUnresolved)
				require.Nil(t, response)
				return
			}
			require.NoError(t, err)
			category := "math"
			if test.failure {
				category = "other"
				if test.policy == config.RuleOnUnknownMatch {
					category = "math-route"
				}
				require.Equal(t, "domain_evaluation_failed", response.SignalErrors["domain:math"])
			}
			assertIntentScore(t, response, category, !test.failure)
		})
	}
}

func TestIntentQueueFullDoesNotRetryOrInventScore(t *testing.T) {
	started, finish := make(chan struct{}), make(chan struct{})
	var finishOnce sync.Once
	defer finishOnce.Do(func() { close(finish) })
	var holding atomic.Bool
	service := domainResultService(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if holding.Load() {
			close(started)
			select {
			case <-finish:
			case <-r.Context().Done():
			}
		}
		writeDomainDistribution(w)
	}), func(cfg *config.RouterConfig) {
		cfg.ModelAdmission = map[string]config.AdmissionConfig{"domain_classifier": {MaxConcurrency: 1, OnOverflow: "shed"}}
	})
	holding.Store(true)
	done := make(chan error, 1)
	go func() {
		_, err := service.ClassifyIntent(context.Background(), IntentRequest{Text: "hold the slot"})
		done <- err
	}()
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("model request did not acquire the slot")
	}
	counter := metrics.ModelAdmissionOutcomes.WithLabelValues("domain_classifier", "shed")
	before := testutil.ToFloat64(counter)
	response, err := service.ClassifyIntent(context.Background(), IntentRequest{Text: "shed this request", Options: &IntentOptions{ReturnProbabilities: true}})
	holding.Store(false)
	finishOnce.Do(func() { close(finish) })
	require.NoError(t, <-done)
	require.NoError(t, err)
	assert.Equal(t, float64(1), testutil.ToFloat64(counter)-before, "one failed evaluation must not reacquire admission")
	assertIntentScore(t, response, "other", false)
}

func TestIntentKeepsFallbackWhenDomainWasNotEvaluated(t *testing.T) {
	for _, failure := range []bool{false, true} {
		t.Run(strconv.FormatBool(failure), func(t *testing.T) {
			var calls atomic.Int32
			var failing atomic.Bool
			service := domainResultService(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				if failing.Load() {
					http.Error(w, "unavailable", http.StatusBadRequest)
					return
				}
				writeDomainDistribution(w)
			}), func(cfg *config.RouterConfig) {
				cfg.Decisions = nil
				cfg.CategoryModel.Threshold = 0
			})
			calls.Store(0)
			failing.Store(failure)
			response, err := service.ClassifyIntent(context.Background(), IntentRequest{Text: "fallback", Options: &IntentOptions{ReturnProbabilities: true}})
			require.NoError(t, err)
			require.Equal(t, int32(1), calls.Load())
			category := "math"
			if failure {
				category = "other"
			}
			assertIntentScore(t, response, category, !failure)
		})
	}
}

func TestIntentUnrelatedSignalErrorPreservesDecisionScore(t *testing.T) {
	service := &ClassificationService{}
	signals := &classification.SignalResults{SignalErrors: map[string]string{"pii:email": "pii_evaluation_failed"}}
	result := &decision.DecisionResult{Decision: &config.Decision{Name: "math-route"}, Confidence: 0.75, ConfidenceScored: true}
	category := resolveIntentCategory(context.Background(), nil, result, signals, "")
	response := service.buildIntentResponseFromSignals(signals, result, category, IntentRequest{Options: &IntentOptions{ReturnProbabilities: true}}, nil, nil)
	assertIntentScore(t, response, "math-route", true)
	require.Equal(t, signals.SignalErrors, response.SignalErrors)
}
