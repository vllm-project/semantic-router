package services

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func TestClassificationServiceRefreshRuntimeConfigRefreshesClassifierConfig(t *testing.T) {
	oldConfig := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{Name: "old_route"}},
		},
	}
	newConfig := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{Name: "new_route"}},
		},
	}

	service := &ClassificationService{
		classifier: &classification.Classifier{Config: oldConfig},
		config:     oldConfig,
	}

	service.RefreshRuntimeConfig(newConfig)

	if service.config != newConfig {
		t.Fatalf("expected service config to be updated")
	}
	if service.classifier == nil {
		t.Fatalf("expected classifier to remain available")
	}
	if got := service.classifier.Config.Decisions; len(got) != 1 || got[0].Name != "new_route" {
		t.Fatalf("expected new policy in prepared classifier, got %+v", got)
	}
	if oldConfig.Decisions[0].Name != "old_route" {
		t.Fatal("refresh mutated the previous config snapshot")
	}
	t.Cleanup(func() { _ = service.Close() })
}

func TestClassificationServiceRefreshRuntimeConfigDoesNotReplaceGlobalConfig(t *testing.T) {
	globalConfig := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{Name: "global_route"}},
		},
	}
	oldConfig := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{Name: "old_route"}},
		},
	}
	newConfig := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{Name: "new_route"}},
		},
	}

	restoreGlobalConfig := replaceGlobalConfigForServiceTest(globalConfig)
	t.Cleanup(restoreGlobalConfig)

	service := NewClassificationService(nil, oldConfig)
	service.RefreshRuntimeConfig(newConfig)

	if got := service.GetConfig(); got != newConfig {
		t.Fatalf("service.GetConfig() = %p, want %p", got, newConfig)
	}
	if !service.HasClassifier() {
		t.Fatal("reload did not activate a classifier for placeholder service")
	}
	if got := config.Get(); got != globalConfig {
		t.Fatalf("config.Get() = %p, want unchanged global config %p", got, globalConfig)
	}
}

func TestClassificationServiceRefreshRuntimeConfigRetainsSnapshotOnFailure(t *testing.T) {
	oldConfig := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{Name: "old_route"}},
		},
	}
	oldClassifier := &classification.Classifier{Config: oldConfig}
	service := &ClassificationService{
		classifier: oldClassifier,
		config:     oldConfig,
	}

	service.RefreshRuntimeConfig(nil)

	if service.config != oldConfig {
		t.Fatal("failed reload replaced the service config")
	}
	if service.classifier != oldClassifier {
		t.Fatal("failed reload replaced the classifier")
	}
}

func TestClassificationServiceConcurrentClassifyAndRefresh(t *testing.T) {
	oldConfig := &config.RouterConfig{}
	newConfig := &config.RouterConfig{}
	service := &ClassificationService{
		classifier: &classification.Classifier{Config: oldConfig},
		config:     oldConfig,
	}

	var wg sync.WaitGroup
	for range 8 {
		wg.Add(2)
		go func() {
			defer wg.Done()
			for range 20 {
				_, _ = service.ClassifyIntentForEval(context.Background(), IntentRequest{Text: "hello"})
			}
		}()
		go func() {
			defer wg.Done()
			for range 20 {
				service.RefreshRuntimeConfig(newConfig)
			}
		}()
	}
	wg.Wait()
}

func TestUpdateConfigDoesNotWaitForRemoteClassification(t *testing.T) {
	previousGlobal := config.Get()
	t.Cleanup(func() {
		if previousGlobal != nil {
			config.Replace(previousGlobal)
			return
		}
		config.Replace(&config.RouterConfig{})
	})
	started := make(chan struct{})
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, _ *http.Request) {
			close(started)
			<-release
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprint(w, `{
				"id":"classification",
				"choices":[{
					"message":{
						"content":"{\"scores\":{\"SAFE\":0.1,\"RISKY\":0.9},\"rationale\":\"test\"}"
					}
				}]
			}`)
		},
	))
	defer server.Close()
	host, portString, err := net.SplitHostPort(
		strings.TrimPrefix(server.URL, "http://"),
	)
	if err != nil {
		t.Fatalf("split server address: %v", err)
	}
	var port int
	if _, scanErr := fmt.Sscanf(portString, "%d", &port); scanErr != nil {
		t.Fatalf("parse server port: %v", scanErr)
	}
	threshold := 0.5
	cfg := &config.RouterConfig{
		ExternalModels: []config.ExternalModelConfig{{
			Name:      "judge",
			ModelRole: config.ModelRoleClassification,
			ModelName: "judge",
			ModelEndpoint: config.ClassifierVLLMEndpoint{
				Address: host, Port: port, Protocol: "http",
			},
			TimeoutSeconds: 5,
		}},
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{ClassifierRules: []config.ClassifierSignalRule{{
				Name: "risk", Type: "llm", Model: "judge",
				Labels: []string{"SAFE", "RISKY"}, Instructions: "Classify.",
			}}},
			Decisions: []config.Decision{{
				Name: "risk-route",
				Rules: config.RuleNode{
					Type: "classifier", Name: "risk", Label: "RISKY",
					Predicate: &config.NumericPredicate{GTE: &threshold},
				},
			}},
		},
	}
	classifier, err := classification.NewClassifier(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("new classifier: %v", err)
	}
	service := NewClassificationService(classifier, cfg)
	classifyDone := make(chan struct{})
	go func() {
		defer close(classifyDone)
		_, _ = service.ClassifyIntent(context.Background(), IntentRequest{Text: "classify me"})
	}()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("remote classifier did not start")
	}

	updateDone := make(chan struct{})
	go func() {
		service.UpdateConfig(&config.RouterConfig{})
		close(updateDone)
	}()
	select {
	case <-updateDone:
	case <-time.After(250 * time.Millisecond):
		t.Fatal("config update waited for remote classification")
	}
	close(release)
	select {
	case <-classifyDone:
	case <-time.After(time.Second):
		t.Fatal("classification did not complete")
	}
}

func replaceGlobalConfigForServiceTest(newCfg *config.RouterConfig) func() {
	previous := config.Get()
	config.Replace(newCfg)
	return func() {
		if previous != nil {
			config.Replace(previous)
			return
		}
		config.Replace(&config.RouterConfig{})
	}
}

type serviceTestOwner struct {
	closed atomic.Int32
	close  func() error
}

func (o *serviceTestOwner) Close() error {
	o.closed.Add(1)
	return o.close()
}

func TestStandaloneRefreshDrainsRealCallAndReleasesOnlyOwnedRuntime(t *testing.T) {
	started, release := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		close(started)
		<-release
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"choices":[{"message":{"content":"{\"scores\":{\"SAFE\":0.1,\"RISKY\":0.9},\"rationale\":\"test\"}"}}]}`)
	}))
	defer func() { unblock(); server.Close() }()
	host, portString, err := net.SplitHostPort(strings.TrimPrefix(server.URL, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	var port int
	if _, scanErr := fmt.Sscanf(portString, "%d", &port); scanErr != nil {
		t.Fatal(scanErr)
	}
	threshold := 0.5
	cfg := &config.RouterConfig{
		ExternalModels: []config.ExternalModelConfig{{Name: "judge", ModelRole: config.ModelRoleClassification, ModelName: "judge", ModelEndpoint: config.ClassifierVLLMEndpoint{Address: host, Port: port, Protocol: "http"}, TimeoutSeconds: 5}},
		IntelligentRouting: config.IntelligentRouting{
			Signals:   config.Signals{ClassifierRules: []config.ClassifierSignalRule{{Name: "risk", Type: "llm", Model: "judge", Labels: []string{"SAFE", "RISKY"}, Instructions: "Classify."}}},
			Decisions: []config.Decision{{Name: "old-risk-route", Rules: config.RuleNode{Type: "classifier", Name: "risk", Label: "RISKY", Predicate: &config.NumericPredicate{GTE: &threshold}}}},
		},
	}
	classifier, err := classification.NewLegacyClassifierFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	service := NewClassificationService(classifier, cfg)
	owner := &serviceTestOwner{close: classifier.Close}
	service.runtimeOwner = owner
	service.unifiedClassifier = &classification.UnifiedClassifier{}
	t.Cleanup(func() { _ = service.Close() })
	callDone := make(chan error, 1)
	go func() {
		_, err := service.ClassifyIntent(context.Background(), IntentRequest{Text: "classify me"})
		callDone <- err
	}()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("remote inference did not start")
	}
	refreshDone := make(chan error, 1)
	go func() { refreshDone <- service.TryRefreshRuntimeConfig(&config.RouterConfig{}) }()
	select {
	case err := <-refreshDone:
		t.Fatalf("refresh retired an active request: %v", err)
	case <-time.After(25 * time.Millisecond):
	}
	if owner.closed.Load() != 0 {
		t.Fatal("old classifier closed under its HTTP call")
	}
	unblock()
	if err := <-callDone; err != nil {
		t.Fatal(err)
	}
	if err := <-refreshDone; err != nil {
		t.Fatal(err)
	}
	if owner.closed.Load() != 1 || service.unifiedClassifier != nil {
		t.Fatal("refresh did not release old ownership and replace its unified view")
	}
	current := service.GetClassifier()
	if err := service.TryRefreshRuntimeConfig(nil); err == nil || service.GetClassifier() != current {
		t.Fatal("failed preparation replaced the usable snapshot")
	}
	if err := service.Close(); err != nil {
		t.Fatal(err)
	}
	if err := service.TryRefreshRuntimeConfig(cfg); !errors.Is(err, binding.ErrClosed) {
		t.Fatalf("closed service accepted a new candidate: %v", err)
	}
	if owner.closed.Load() != 1 {
		t.Fatal("retired classifier was closed twice")
	}
}
