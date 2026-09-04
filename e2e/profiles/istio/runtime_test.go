package istio

import (
	"bytes"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"k8s.io/client-go/util/jsonpath"
)

func TestSidecarJSONPaths(t *testing.T) {
	tests := []struct {
		name            string
		podList         string
		wantContainers  []string
		wantReadyStates []string
	}{
		{
			name: "classic sidecar",
			podList: `{
				"apiVersion": "v1",
				"kind": "PodList",
				"items": [{
					"metadata": {"name": "classic-sidecar"},
					"spec": {
						"containers": [
							{"name": "semantic-router"},
							{"name": "istio-proxy"}
						],
						"initContainers": [{"name": "istio-init"}]
					},
					"status": {
						"containerStatuses": [
							{"name": "semantic-router", "ready": true},
							{"name": "istio-proxy", "ready": true}
						],
						"initContainerStatuses": [{"name": "istio-init", "ready": false}]
					}
				}]
			}`,
			wantContainers:  []string{"semantic-router", "istio-proxy", "istio-init"},
			wantReadyStates: []string{"true", "true"},
		},
		{
			name: "native sidecar",
			podList: `{
				"apiVersion": "v1",
				"kind": "PodList",
				"items": [{
					"metadata": {"name": "native-sidecar"},
					"spec": {
						"containers": [{"name": "semantic-router"}],
						"initContainers": [
							{"name": "istio-init"},
							{"name": "istio-proxy", "restartPolicy": "Always"}
						]
					},
					"status": {
						"containerStatuses": [{"name": "semantic-router", "ready": true}],
						"initContainerStatuses": [
							{"name": "istio-init", "ready": false},
							{"name": "istio-proxy", "ready": true}
						]
					}
				}]
			}`,
			wantContainers:  []string{"semantic-router", "istio-init", "istio-proxy"},
			wantReadyStates: []string{"true", "true"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var podList any
			if err := json.Unmarshal([]byte(tt.podList), &podList); err != nil {
				t.Fatalf("unmarshal PodList fixture: %v", err)
			}

			if got := executeJSONPath(t, sidecarContainerJSONPath, podList); !slices.Equal(got, tt.wantContainers) {
				t.Errorf("sidecar containers = %v, want %v", got, tt.wantContainers)
			}
			if got := executeJSONPath(t, sidecarReadinessJSONPath, podList); !slices.Equal(got, tt.wantReadyStates) {
				t.Errorf("sidecar readiness = %v, want %v", got, tt.wantReadyStates)
			}
		})
	}
}

func executeJSONPath(t *testing.T, expression string, input any) []string {
	t.Helper()

	parser := jsonpath.New(t.Name())
	if err := parser.Parse(expression); err != nil {
		t.Fatalf("parse JSONPath %q: %v", expression, err)
	}

	var output bytes.Buffer
	if err := parser.Execute(&output, input); err != nil {
		t.Fatalf("execute JSONPath %q: %v", expression, err)
	}
	return strings.Fields(output.String())
}
