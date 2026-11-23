package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fakeclientset "k8s.io/client-go/kubernetes/fake"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/k8s/configwriter"
)

const (
	persistedConfigNamespace = "vllm-semantic-router-system"
	persistedConfigName      = "semantic-router-config"
)

// persistedConfigDocument is a canonical config whose one model, decision
// priority, preview timeout and tools catalog say which save a reader saw.
func persistedConfigDocument(model string, priority, previewTimeoutSeconds int) string {
	return fmt.Sprintf(`version: v0.5
providers:
  defaults: {model: %[1]s}
  models:
    - name: %[1]s
      description: The %[1]s model
      backend_refs: [{name: %[1]s, endpoint: "%[1]s.example:8000", protocol: http, weight: 1}]
recipes:
  - name: default
    decisions:
      - name: route
        priority: %[2]d
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: %[1]s}]
        plugins:
          - type: tools
            configuration:
              enabled: true
              select: {mode: add, tools_db_path: "/tools/%[1]s.json"}
global:
  services:
    api:
      routing_preview:
        request_timeout_seconds: %[3]d
`, model, priority, previewTimeoutSeconds)
}

var healthyModelBackend = modelVerificationRoundTripper(func(*http.Request) (*http.Response, error) {
	return modelVerificationHTTPResponse(http.StatusOK, `{"data":[]}`), nil
})

func resetPersistedRouterConfigCache() {
	cache := &persistedRouterConfigCache
	cache.Lock()
	defer cache.Unlock()
	cache.target, cache.readAt, cache.data, cache.err = configwriter.ConfigMapTarget{}, time.Time{}, nil, nil
}

// usePersistedConfigClock moves the persisted config's clock by hand.
func usePersistedConfigClock(t *testing.T) *modelHealthTestClock {
	t.Helper()
	clock := &modelHealthTestClock{now: time.Unix(1_800_000_000, 0)}
	previous := persistedRouterConfigClock
	persistedRouterConfigClock = clock.Now
	t.Cleanup(func() { persistedRouterConfigClock = previous })
	return clock
}

// usePersistedConfigMap declares a ConfigMap, as a Kubernetes deployment
// does, served by a fake clientset that holds document when seeded, over a
// subPath mount that keeps document whatever is saved later.
func usePersistedConfigMap(t *testing.T, document string, seeded bool) (string, *fakeclientset.Clientset) {
	t.Helper()
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	require.NoError(t, os.WriteFile(configPath, []byte(document), 0o400))
	t.Setenv(configwriter.ConfigMapNameEnv, persistedConfigName)
	t.Setenv(configwriter.ConfigMapNamespaceEnv, persistedConfigNamespace)
	clientset := fakeclientset.NewSimpleClientset()
	if seeded {
		createPersistedConfigMap(t, clientset, document)
	}
	t.Cleanup(stubInClusterConfigMapWriter(t, configwriter.NewConfigMapWriter(clientset)))
	return configPath, clientset
}

func createPersistedConfigMap(t *testing.T, clientset *fakeclientset.Clientset, document string) {
	t.Helper()
	_, err := clientset.CoreV1().ConfigMaps(persistedConfigNamespace).Create(context.Background(), &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Namespace: persistedConfigNamespace, Name: persistedConfigName},
		Data:       map[string]string{"config.yaml": document},
	}, metav1.CreateOptions{})
	require.NoError(t, err)
}

func savePersistedConfigMap(t *testing.T, clientset *fakeclientset.Clientset, document string) {
	t.Helper()
	configMaps := clientset.CoreV1().ConfigMaps(persistedConfigNamespace)
	current, err := configMaps.Get(context.Background(), persistedConfigName, metav1.GetOptions{})
	require.NoError(t, err)
	current.Data["config.yaml"] = document
	_, err = configMaps.Update(context.Background(), current, metav1.UpdateOptions{})
	require.NoError(t, err)
}

func checkedModelNames(t *testing.T, service *modelHealthService) []string {
	t.Helper()
	response, err := service.Check(context.Background(), nil, false)
	require.NoError(t, err)
	names := make([]string, 0, len(response.Models))
	for _, entry := range response.Models {
		names = append(names, entry.Model)
	}
	return names
}

func verifiableModel(service *modelVerificationService, models ...string) string {
	for _, model := range models {
		if _, err := service.resolveTarget(model); err == nil {
			return model
		}
	}
	return ""
}

func evaluatedRoutePriority(configPath string) int {
	result := &TestQueryResult{EvalTrace: json.RawMessage(`[{"decision_name":"route","state":"matched","matched":true}]`)}
	appendEvaluatedRulesFromTrace(result, configPath, "")
	if len(result.EvaluatedRules) != 1 {
		return -1
	}
	return result.EvaluatedRules[0].Priority
}

func selectedToolsDB(t *testing.T, configPath string) string {
	t.Helper()
	selection, err := ReadToolSelection(configPath)
	require.NoError(t, err)
	return selection.ToolsDBPath
}

// A save to the ConfigMap reaches model health, model verification, the
// topology preview and the tools catalog once the last read is older than
// the TTL, while the pod's subPath mount still holds the first document.
func TestPersistedRouterConfigViewsFollowTheConfigMap(t *testing.T) {
	for _, view := range []struct {
		name string
		// open builds the view once, as route registration does, and returns
		// what it shows.
		open         func(configPath string) func(*testing.T) any
		first, saved any
	}{
		{
			name: "model health",
			open: func(configPath string) func(*testing.T) any {
				service := newModelHealthService(configPath, modelHealthOptions{client: healthyModelBackend})
				return func(t *testing.T) any { return checkedModelNames(t, service) }
			},
			first: []string{"alpha"}, saved: []string{"beta"},
		},
		{
			name: "model verification",
			open: func(configPath string) func(*testing.T) any {
				service := newModelVerificationService(configPath, modelVerificationOptions{client: healthyModelBackend})
				return func(*testing.T) any { return verifiableModel(service, "alpha", "beta") }
			},
			first: "alpha", saved: "beta",
		},
		{
			name: "topology preview timeout",
			open: func(configPath string) func(*testing.T) any {
				return func(*testing.T) any { return topologyPreviewTimeout(configPath) }
			},
			first: 605 * time.Second, saved: 905 * time.Second,
		},
		{
			name: "topology decision priority",
			open: func(configPath string) func(*testing.T) any {
				return func(*testing.T) any { return evaluatedRoutePriority(configPath) }
			},
			first: 10, saved: 20,
		},
		{
			name: "tools catalog",
			open: func(configPath string) func(*testing.T) any {
				return func(t *testing.T) any { return selectedToolsDB(t, configPath) }
			},
			first: "/tools/alpha.json", saved: "/tools/beta.json",
		},
	} {
		t.Run(view.name, func(t *testing.T) {
			clock := usePersistedConfigClock(t)
			configPath, clientset := usePersistedConfigMap(t, persistedConfigDocument("alpha", 10, 600), true)
			observe := view.open(configPath)
			require.Equal(t, view.first, observe(t))

			savePersistedConfigMap(t, clientset, persistedConfigDocument("beta", 20, 900))
			clock.Advance(persistedRouterConfigTTL - time.Millisecond)
			require.Equal(t, view.first, observe(t), "a read within the TTL answers from the last one")

			clock.Advance(time.Millisecond)
			require.Equal(t, view.saved, observe(t), "the saved ConfigMap shows after the TTL")
			mounted, err := os.ReadFile(configPath)
			require.NoError(t, err)
			require.Equal(t, persistedConfigDocument("alpha", 10, 600), string(mounted))
		})
	}
}

// The Models page's health read stops listing a model the saved ConfigMap
// removed once the TTL has passed, without a Dashboard restart.
func TestModelHealthRouteFollowsTheSavedConfigMap(t *testing.T) {
	clock := usePersistedConfigClock(t)
	configPath, clientset := usePersistedConfigMap(t, persistedConfigDocument("alpha", 10, 600), true)
	handler := newModelHealthHandler(newModelHealthService(configPath, modelHealthOptions{client: healthyModelBackend}))
	read := func(query string) (int, ModelHealthResponse) {
		t.Helper()
		recorder := httptest.NewRecorder()
		handler(recorder, httptest.NewRequest(http.MethodGet, modelHealthPath+query, nil))
		var response ModelHealthResponse
		require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &response))
		return recorder.Code, response
	}
	status, response := read("")
	require.Equal(t, http.StatusOK, status)
	require.Len(t, response.Models, 1)
	require.Equal(t, "alpha", response.Models[0].Model)

	savePersistedConfigMap(t, clientset, persistedConfigDocument("beta", 20, 900))
	clock.Advance(persistedRouterConfigTTL)
	status, response = read("")
	require.Equal(t, http.StatusOK, status)
	require.Len(t, response.Models, 1)
	require.Equal(t, "beta", response.Models[0].Model)
	status, _ = read("?model=alpha")
	require.Equal(t, http.StatusNotFound, status)
}

// Without a declared ConfigMap the views read the config file on every call,
// as they always did, and no Kubernetes client is built.
func TestPersistedRouterConfigReadsTheLocalFileEveryTime(t *testing.T) {
	t.Setenv(configwriter.ConfigMapNameEnv, "")
	t.Setenv(configwriter.ConfigMapNamespaceEnv, "")
	configMapWriterMu.Lock()
	previousFactory, previousResolved := newInClusterConfigMapWriter, configMapWriterResolved
	newInClusterConfigMapWriter = func() (*configwriter.ConfigMapWriter, error) {
		t.Error("the local config file needs no Kubernetes client")
		return nil, errors.New("not in a cluster")
	}
	configMapWriterResolved = false
	configMapWriterMu.Unlock()
	t.Cleanup(func() {
		configMapWriterMu.Lock()
		newInClusterConfigMapWriter, configMapWriterResolved = previousFactory, previousResolved
		configMapWriterMu.Unlock()
	})
	resetPersistedRouterConfigCache()
	usePersistedConfigClock(t)

	configPath := filepath.Join(t.TempDir(), "config.yaml")
	require.NoError(t, os.WriteFile(configPath, []byte(persistedConfigDocument("alpha", 10, 600)), 0o600))
	service := newModelHealthService(configPath, modelHealthOptions{client: healthyModelBackend})
	require.Equal(t, []string{"alpha"}, checkedModelNames(t, service))
	require.Equal(t, "/tools/alpha.json", selectedToolsDB(t, configPath))

	require.NoError(t, os.WriteFile(configPath, []byte(persistedConfigDocument("beta", 20, 900)), 0o600))
	require.Equal(t, []string{"beta"}, checkedModelNames(t, service))
	require.Equal(t, "/tools/beta.json", selectedToolsDB(t, configPath))
	require.Equal(t, 905*time.Second, topologyPreviewTimeout(configPath))
	require.True(t, persistedRouterConfigCache.readAt.IsZero(), "the local file is not cached")
}

// A failed ConfigMap read fails a view as a failed file read does, and the
// first read after the TTL asks the Kubernetes API again.
func TestPersistedRouterConfigReportsAConfigMapReadFailure(t *testing.T) {
	clock := usePersistedConfigClock(t)
	document := persistedConfigDocument("alpha", 10, 600)
	configPath, clientset := usePersistedConfigMap(t, document, false)
	service := newModelHealthService(configPath, modelHealthOptions{client: healthyModelBackend})

	_, err := service.Check(context.Background(), nil, false)
	var failure *modelVerificationError
	require.ErrorAs(t, err, &failure)
	require.Equal(t, "config_unavailable", failure.code)
	_, err = ReadToolSelection(configPath)
	require.Error(t, err)

	createPersistedConfigMap(t, clientset, document)
	clock.Advance(persistedRouterConfigTTL)
	require.Equal(t, []string{"alpha"}, checkedModelNames(t, service))
}
