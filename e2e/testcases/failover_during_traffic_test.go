package testcases

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

func TestFailoverRequiresSuccessfulPodDeletion(t *testing.T) {
	for _, scenario := range []struct {
		name         string
		listStatus   int
		emptyList    bool
		deleteStatus int
		wantDelete   bool
		wantError    string
	}{
		{name: "list forbidden", listStatus: http.StatusForbidden, wantError: "list backend pods"},
		{name: "no pods", listStatus: http.StatusOK, emptyList: true, wantError: "no backend pod"},
		{name: "delete forbidden", listStatus: http.StatusOK, deleteStatus: http.StatusForbidden, wantDelete: true, wantError: "delete backend pod"},
		{name: "pod already deleted", listStatus: http.StatusOK, deleteStatus: http.StatusNotFound, wantDelete: true, wantError: "delete backend pod"},
		{name: "deleted exact pod", listStatus: http.StatusOK, deleteStatus: http.StatusOK, wantDelete: true},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			deletions := make(chan string, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				switch r.Method {
				case http.MethodGet:
					if r.URL.Path != "/api/v1/namespaces/default/pods" || r.URL.Query().Get("labelSelector") != "app=vllm-llama3-8b-instruct" {
						t.Errorf("unexpected pod list request: %s", r.URL)
					}
					w.WriteHeader(scenario.listStatus)
					if scenario.listStatus != http.StatusOK {
						_, _ = fmt.Fprint(w, `{"kind":"Status","apiVersion":"v1","status":"Failure","reason":"Forbidden","code":403}`)
						return
					}
					items := `[{"metadata":{"name":"backend-a"}},{"metadata":{"name":"backend-b"}}]`
					if scenario.emptyList {
						items = "[]"
					}
					_, _ = fmt.Fprintf(w, `{"kind":"PodList","apiVersion":"v1","items":%s}`, items)
				case http.MethodDelete:
					deletions <- r.URL.Path
					w.WriteHeader(scenario.deleteStatus)
					if scenario.deleteStatus != http.StatusOK {
						_, _ = fmt.Fprintf(w, `{"kind":"Status","apiVersion":"v1","status":"Failure","code":%d}`, scenario.deleteStatus)
						return
					}
					_, _ = fmt.Fprint(w, `{"kind":"Status","apiVersion":"v1","status":"Success"}`)
				default:
					t.Errorf("unexpected request method: %s", r.Method)
					w.WriteHeader(http.StatusMethodNotAllowed)
				}
			}))
			t.Cleanup(server.Close)
			client, clientErr := kubernetes.NewForConfig(&rest.Config{Host: server.URL})
			if clientErr != nil {
				t.Fatal(clientErr)
			}
			name, err := deleteProductionStackBackendPod(context.Background(), client.CoreV1().Pods("default"))
			if scenario.wantError == "" {
				if err != nil || name != "backend-a" {
					t.Fatalf("delete returned pod %q and error %v, want backend-a and no error", name, err)
				}
			} else if err == nil || !strings.Contains(err.Error(), scenario.wantError) || name != "" {
				t.Fatalf("delete returned pod %q and error %v, want no pod and error containing %q", name, err, scenario.wantError)
			}
			wantPath := ""
			if scenario.wantDelete {
				wantPath = "/api/v1/namespaces/default/pods/backend-a"
			}
			var deleted string
			select {
			case deleted = <-deletions:
			default:
			}
			if deleted != wantPath {
				t.Fatalf("deleted %q, want %q", deleted, wantPath)
			}
		})
	}
}
