package testcases

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func TestResponseAPIRedisRequiresDeclaredTopology(t *testing.T) {
	for _, tc := range []struct {
		name, profile, selector string
		listError, wantError    bool
		wantFound, wantCluster  bool
	}{
		{"standalone", "response-api-redis", "app=redis", false, false, true, false},
		{"cluster", "response-api-redis-cluster", "app=redis-cluster", false, false, true, true},
		{"standalone missing", "response-api-redis", "", false, true, false, false},
		{"cluster missing", "response-api-redis-cluster", "", false, true, false, false},
		{"standalone replaced by cluster", "response-api-redis", "app=redis-cluster", false, true, false, false},
		{"cluster replaced by standalone", "response-api-redis-cluster", "app=redis", false, true, false, false},
		{"memory without Redis", "response-api", "", false, false, false, false},
		{"gateway memory without Redis", "envoy-ai-gateway", "", false, false, false, false},
		{"Redis list unavailable", "response-api-redis", "", true, true, false, false},
		{"memory list unavailable", "response-api", "", true, true, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := responseAPIRedisTestClient(t, tc.selector, tc.listError)
			_, cluster, found, err := getResponseAPIRedisPod(context.Background(), client, tc.profile)
			if (err != nil) != tc.wantError {
				t.Fatalf("error = %v, wantError = %v", err, tc.wantError)
			}
			if found != tc.wantFound || cluster != tc.wantCluster {
				t.Fatalf("found=%t cluster=%t, want found=%t cluster=%t", found, cluster, tc.wantFound, tc.wantCluster)
			}
		})
	}
}

func TestResponseAPIRedisAssertionsRejectMissingRequiredFixture(t *testing.T) {
	checks := map[string]func(context.Context, *kubernetes.Clientset, string, pkgtestcases.TestCaseOptions) error{
		"stored":  assertRedisResponseStored,
		"deleted": assertRedisResponseDeleted,
		"TTL":     assertRedisResponseTTLSet,
		"restart TTL extension": func(ctx context.Context, client *kubernetes.Clientset, responseID string, opts pkgtestcases.TestCaseOptions) error {
			return extendRedisKeyTTL(ctx, client, responseID, restartKeyTTLSeconds, opts)
		},
	}
	for name, check := range checks {
		for _, profile := range []string{"response-api-redis", "response-api-redis-cluster", "response-api", "envoy-ai-gateway"} {
			t.Run(name+"/"+profile, func(t *testing.T) {
				client := responseAPIRedisTestClient(t, "", false)
				err := check(context.Background(), client, "resp_test", pkgtestcases.TestCaseOptions{Profile: profile})
				required := profile == "response-api-redis" || profile == "response-api-redis-cluster"
				if (err != nil) != required {
					t.Fatalf("error = %v, Redis required = %v", err, required)
				}
			})
		}
	}
}

func responseAPIRedisTestClient(t *testing.T, selector string, listError bool) *kubernetes.Clientset {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/namespaces/default/pods" {
			t.Errorf("unexpected Kubernetes request: %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if listError {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		pods := corev1.PodList{TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "PodList"}}
		if selector != "" && r.URL.Query().Get("labelSelector") == selector {
			pods.Items = []corev1.Pod{{ObjectMeta: metav1.ObjectMeta{Name: "redis-fixture"}, Status: corev1.PodStatus{Phase: corev1.PodRunning}}}
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(pods); err != nil {
			t.Errorf("encode PodList: %v", err)
		}
	}))
	t.Cleanup(server.Close)
	client, err := kubernetes.NewForConfig(&rest.Config{Host: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	return client
}
