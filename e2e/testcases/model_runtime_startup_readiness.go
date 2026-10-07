package testcases

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"sort"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/modelruntime"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("model-runtime-startup-readiness", pkgtestcases.TestCase{
		Description: "The Router reports ready only once every Router-managed deployment is ready: /startup-status lists each one ready, without the attached endpoints, and startup completed after the last of them",
		Tags:        []string{"model-runtime", "lifecycle", "readiness", "managed"},
		Fn:          testModelRuntimeStartupReadiness,
	})
}

// mrStartupStatus is the part of GET /startup-status the case reads.
type mrStartupStatus struct {
	Phase            string   `json:"phase"`
	Ready            bool     `json:"ready"`
	PendingModels    []string `json:"pending_models"`
	ReadyModels      int      `json:"ready_models"`
	TotalModels      int      `json:"total_models"`
	ModelDeployments []struct {
		Name     string `json:"name"`
		Artifact string `json:"artifact"`
		Process  string `json:"process"`
		State    string `json:"state"`
		Ready    bool   `json:"ready"`
	} `json:"model_deployments"`
}

func testModelRuntimeStartupReadiness(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := openRouterRuntimeSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	status, err := session.startupStatus(ctx)
	if err != nil {
		return err
	}
	if status.Phase != "ready" || !status.Ready || len(status.PendingModels) != 0 ||
		status.TotalModels != len(mrManagedDeployments) || status.ReadyModels != len(mrManagedDeployments) {
		return fmt.Errorf("startup status of a ready Router: phase %q ready %v pending %v, %d of %d deployments ready, want all %d",
			status.Phase, status.Ready, status.PendingModels, status.ReadyModels, status.TotalModels, len(mrManagedDeployments))
	}
	listed := make([]string, 0, len(status.ModelDeployments))
	for _, deployment := range status.ModelDeployments {
		listed = append(listed, deployment.Name)
		process := mrDeviceProcess
		if deployment.Name == mrDecisionDeployment {
			process = mrDecisionsProcess
		}
		if !deployment.Ready || deployment.State != "ready" || deployment.Process != process ||
			!strings.HasPrefix(deployment.Artifact, "/tmp/vsr-fixtures/") {
			return fmt.Errorf("startup status lists %s as %+v, want it ready in process %s with its fixture artifact", deployment.Name, deployment, process)
		}
	}
	managed := slices.Clone(mrManagedDeployments)
	sort.Strings(managed)
	if !slices.Equal(listed, managed) {
		return fmt.Errorf("startup status lists %v, want exactly the managed deployments %v (attached endpoints are not the Router's to wait for)", listed, managed)
	}

	order, err := startupLogOrder(ctx, client, session.pod)
	if err != nil {
		return err
	}
	for _, name := range mrManagedDeployments {
		readyAt, ok := order.firstReady[name]
		if !ok || readyAt > order.startupComplete {
			return fmt.Errorf("startup completed (log line %d) before %s was ready (line %d, found %v): the Router served before its model", order.startupComplete, name, readyAt, ok)
		}
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"model_deployments": listed, "startup_complete_line": order.startupComplete, "first_ready_lines": order.firstReady,
		})
	}
	return nil
}

func (s *modelRuntimeSession) startupStatus(ctx context.Context) (mrStartupStatus, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, s.api.URL("/startup-status"), nil)
	if err != nil {
		return mrStartupStatus{}, err
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return mrStartupStatus{}, fmt.Errorf("GET /startup-status: %w", err)
	}
	defer response.Body.Close()
	var status mrStartupStatus
	if err := json.NewDecoder(response.Body).Decode(&status); err != nil {
		return mrStartupStatus{}, fmt.Errorf("decode /startup-status (%d): %w", response.StatusCode, err)
	}
	if response.StatusCode != http.StatusOK {
		return mrStartupStatus{}, fmt.Errorf("GET /startup-status: %d %+v", response.StatusCode, status)
	}
	return status, nil
}

// mrStartupOrder holds, by line number of the Router container's log, when
// each deployment first reported ready and when startup completed.
type mrStartupOrder struct {
	firstReady      map[string]int
	startupComplete int
}

func startupLogOrder(ctx context.Context, client kubernetes.Interface, pod modelruntime.PodTarget) (mrStartupOrder, error) {
	logs, err := client.CoreV1().Pods(pod.Namespace).GetLogs(pod.Pod, &corev1.PodLogOptions{Container: pod.Container}).DoRaw(ctx)
	if err != nil {
		return mrStartupOrder{}, fmt.Errorf("read the Router's log: %w", err)
	}
	order := mrStartupOrder{firstReady: map[string]int{}}
	scanner := bufio.NewScanner(bytes.NewReader(logs))
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for line := 1; scanner.Scan(); line++ {
		var event struct {
			Msg         string   `json:"msg"`
			Ready       bool     `json:"ready"`
			Deployments []string `json:"deployments"`
		}
		if json.Unmarshal(scanner.Bytes(), &event) != nil {
			continue
		}
		switch {
		case event.Msg == "startup_complete" && order.startupComplete == 0:
			order.startupComplete = line
		case event.Msg == "deployment_readiness_changed" && event.Ready:
			for _, name := range event.Deployments {
				if _, seen := order.firstReady[name]; !seen {
					order.firstReady[name] = line
				}
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return mrStartupOrder{}, fmt.Errorf("scan the Router's log: %w", err)
	}
	if order.startupComplete == 0 {
		return mrStartupOrder{}, fmt.Errorf("the Router's log has no startup_complete line")
	}
	return order, nil
}
