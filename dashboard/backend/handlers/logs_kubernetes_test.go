package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	fakeclientset "k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
)

const podLogTestNamespace = "vllm-sr"

func podLogTestPod(name, app string, phase corev1.PodPhase, containers ...string) *corev1.Pod {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: podLogTestNamespace, Labels: map[string]string{"app": app}},
		Status:     corev1.PodStatus{Phase: phase},
	}
	for _, container := range containers {
		pod.Spec.Containers = append(pod.Spec.Containers, corev1.Container{Name: container})
	}
	return pod
}

// podLogTestStreams serves a fixed tail or error per pod and records the
// options of every read.
type podLogTestStreams struct {
	mu      sync.Mutex
	tails   map[string]string
	errs    map[string]error
	options map[string]corev1.PodLogOptions
}

func (streams *podLogTestStreams) stream(_ context.Context, namespace, pod string, options *corev1.PodLogOptions) (io.ReadCloser, error) {
	streams.mu.Lock()
	defer streams.mu.Unlock()
	if namespace != podLogTestNamespace {
		return nil, errors.New("read from the wrong namespace")
	}
	if streams.options == nil {
		streams.options = map[string]corev1.PodLogOptions{}
	}
	streams.options[pod] = *options
	if err := streams.errs[pod]; err != nil {
		return nil, err
	}
	return io.NopCloser(strings.NewReader(streams.tails[pod])), nil
}

func newPodLogTestSource(env map[string]string, streams *podLogTestStreams, objects ...runtime.Object) (*podLogSource, *fakeclientset.Clientset) {
	client := fakeclientset.NewSimpleClientset(objects...)
	source := newPodLogSource(client, podLogTestNamespace, func(name string) string { return env[name] })
	if streams != nil {
		source.stream = streams.stream
	}
	return source, client
}

func servePodLogs(t *testing.T, spoolRoot string, source *podLogSource, query string) LogsResponse {
	t.Helper()
	request := httptest.NewRequest(http.MethodGet, "/api/logs?"+query, nil)
	response := httptest.NewRecorder()
	newLogsHandler(spoolRoot, source).ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if response.Body.Len() > maxLogsResponseBytes {
		t.Fatalf("response bytes = %d, limit = %d", response.Body.Len(), maxLogsResponseBytes)
	}
	var payload LogsResponse
	if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if payload.Count != len(payload.Logs) {
		t.Fatalf("count = %d, entries = %d", payload.Count, len(payload.Logs))
	}
	return payload
}

func podLogLines(payload LogsResponse) []string {
	lines := make([]string, 0, len(payload.Logs))
	for _, entry := range payload.Logs {
		lines = append(lines, entry.Pod+" "+entry.Line)
	}
	return lines
}

func TestPodLogsMergeReplicasByTimestampAndRedact(t *testing.T) {
	t.Parallel()
	secret := "pod-log-secret" //nolint:gosec // Canary verifies that credentials are redacted from pod logs.
	streams := &podLogTestStreams{tails: map[string]string{
		"router-a": strings.Join([]string{
			"2026-11-18T02:40:00.000000001Z [info] router ready",
			"2026-11-18T02:40:02Z Authorization: Bearer " + secret,
			"2026-11-18T02:40:03Z -----BEGIN PRIVATE KEY-----",
			"2026-11-18T02:40:03Z cHJpdmF0ZS1rZXktbWF0ZXJpYWwtdGhhdC1tdXN0LW5ldmVyLWJlLXJldHVybmVkLXRvLXVzZXJz",
			"2026-11-18T02:40:03Z -----END PRIVATE KEY-----",
		}, "\n") + "\n",
		"router-b": "2026-11-18T02:40:01Z [info] replica b ready\n2026-11-18T02:40:04Z [warn] slow upstream\n",
	}}
	source, _ := newPodLogTestSource(nil, streams,
		podLogTestPod("router-b", "router", corev1.PodRunning, "router"),
		podLogTestPod("router-a", "router", corev1.PodRunning, "router"),
		podLogTestPod("dashboard-0", "dashboard", corev1.PodRunning, "dashboard"),
		podLogTestPod("other-0", "other", corev1.PodRunning, "other"),
	)

	payload := servePodLogs(t, "", source, "component=router&lines=50")

	if !payload.Supported || payload.DeploymentType != "kubernetes" || payload.Namespace != podLogTestNamespace || payload.Error != "" {
		t.Fatalf("unexpected Kubernetes contract: %#v", payload)
	}
	if strings.Join(payload.Components, ",") != "router,dashboard" || strings.Join(payload.Pods, ",") != "router-a,router-b" {
		t.Fatalf("components = %v, pods = %v", payload.Components, payload.Pods)
	}
	want := []string{
		"router-a [info] router ready",
		"router-b [info] replica b ready",
		"router-a Authorization: [REDACTED]",
		"router-a [REDACTED PRIVATE KEY]",
		"router-b [warn] slow upstream",
	}
	if got := podLogLines(payload); strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Fatalf("merged lines:\n%s\nwant:\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
	}
	first := payload.Logs[0]
	if first.Service != "router" || first.Timestamp != "2026-11-18T02:40:00.000000001Z" {
		t.Fatalf("first entry = %#v", first)
	}
	options := streams.options["router-a"]
	if options.Container != "router" || !options.Timestamps || options.TailLines == nil || *options.TailLines != 50 || options.LimitBytes == nil {
		t.Fatalf("pod log options = %#v", options)
	}
	if _, read := streams.options["dashboard-0"]; read {
		t.Fatal("a router request read the dashboard pod")
	}
}

func TestPodLogsKeepTheNewestLinesAcrossComponents(t *testing.T) {
	t.Parallel()
	streams := &podLogTestStreams{tails: map[string]string{
		"router-a":    "2026-11-18T02:40:00Z r1\n2026-11-18T02:40:03Z r2\n",
		"dashboard-0": "2026-11-18T02:40:01Z d1\n2026-11-18T02:40:02Z d2\n2026-11-18T02:40:04Z d3\n",
	}}
	source, _ := newPodLogTestSource(nil, streams,
		podLogTestPod("router-a", "router", corev1.PodRunning, "router"),
		podLogTestPod("dashboard-0", "dashboard", corev1.PodRunning, "dashboard"),
	)

	payload := servePodLogs(t, "", source, "component=all&lines=3")

	if got := strings.Join(podLogLines(payload), ","); got != "dashboard-0 d2,router-a r2,dashboard-0 d3" {
		t.Fatalf("newest lines = %s", got)
	}
	if payload.Logs[0].Service != "dashboard" || payload.Logs[1].Service != "router" {
		t.Fatalf("services = %#v", payload.Logs)
	}
}

func TestPodLogsReadSelectorsFromTheEnvironment(t *testing.T) {
	t.Parallel()
	env := map[string]string{
		"VLLM_SR_K8S_ROUTER_LOG_SELECTOR": "app.kubernetes.io/component=router",
		"VLLM_SR_K8S_ENVOY_LOG_SELECTOR":  "app=gateway",
	}
	gateway := podLogTestPod("gateway-0", "gateway", corev1.PodRunning, "sidecar", "envoy")
	chartRouter := podLogTestPod("semantic-router-0", "", corev1.PodRunning, "semantic-router")
	chartRouter.Labels = map[string]string{"app.kubernetes.io/component": "router"}
	streams := &podLogTestStreams{tails: map[string]string{
		"gateway-0":         "2026-11-18T02:40:00Z envoy started\n",
		"semantic-router-0": "2026-11-18T02:40:01Z router started\n",
	}}
	source, _ := newPodLogTestSource(env, streams,
		gateway,
		chartRouter,
		podLogTestPod("router-a", "router", corev1.PodRunning, "router"),
	)

	payload := servePodLogs(t, "", source, "component=all")

	if strings.Join(payload.Components, ",") != "router,envoy,dashboard" {
		t.Fatalf("components = %v", payload.Components)
	}
	if got := strings.Join(payload.Pods, ","); got != "semantic-router-0,gateway-0" {
		t.Fatalf("pods = %s", got)
	}
	if streams.options["gateway-0"].Container != "envoy" || streams.options["semantic-router-0"].Container != "semantic-router" {
		t.Fatalf("containers = %#v", streams.options)
	}
	if !strings.Contains(payload.Message, "No pods match app=dashboard") {
		t.Fatalf("message = %q", payload.Message)
	}
}

func TestPodLogsNameAMissingListPermission(t *testing.T) {
	t.Parallel()
	source, client := newPodLogTestSource(nil, &podLogTestStreams{})
	client.PrependReactor("list", "pods", func(k8stesting.Action) (bool, runtime.Object, error) {
		return true, nil, apierrors.NewForbidden(schema.GroupResource{Resource: "pods"}, "", errors.New("RBAC: access denied"))
	})

	payload := servePodLogs(t, "", source, "component=all")

	if payload.Supported || payload.DeploymentType != "kubernetes" || payload.Error != "" || len(payload.Logs) != 0 {
		t.Fatalf("unexpected forbidden contract: %#v", payload)
	}
	for _, want := range []string{"cannot list pods", `namespace "vllm-sr"`, "list verb on pods"} {
		if !strings.Contains(payload.Message, want) {
			t.Fatalf("message %q does not contain %q", payload.Message, want)
		}
	}
}

func TestPodLogsNameAMissingLogPermission(t *testing.T) {
	t.Parallel()
	streams := &podLogTestStreams{errs: map[string]error{
		"router-a": apierrors.NewForbidden(schema.GroupResource{Resource: "pods/log"}, "router-a", errors.New("RBAC: access denied")),
	}}
	source, _ := newPodLogTestSource(nil, streams, podLogTestPod("router-a", "router", corev1.PodRunning, "router"))

	payload := servePodLogs(t, "", source, "component=router")

	if payload.Supported || payload.Error != "" || len(payload.Logs) != 0 || len(payload.Pods) != 0 {
		t.Fatalf("unexpected forbidden contract: %#v", payload)
	}
	for _, want := range []string{"cannot read pod logs", "get verb on pods/log"} {
		if !strings.Contains(payload.Message, want) {
			t.Fatalf("message %q does not contain %q", payload.Message, want)
		}
	}
}

func TestPodLogsReadThroughClientGo(t *testing.T) {
	t.Parallel()
	source, client := newPodLogTestSource(nil, nil, podLogTestPod("router-a", "router", corev1.PodRunning, "istio-proxy", "router"))

	payload := servePodLogs(t, "", source, "component=router&lines=20")

	if len(payload.Logs) != 1 || payload.Logs[0].Line != "fake logs" || payload.Logs[0].Pod != "router-a" {
		t.Fatalf("logs = %#v", payload.Logs)
	}
	var options *corev1.PodLogOptions
	for _, action := range client.Actions() {
		if generic, ok := action.(k8stesting.GenericAction); ok && action.GetSubresource() == "log" {
			options, _ = generic.GetValue().(*corev1.PodLogOptions)
		}
	}
	if options == nil || options.Container != "router" || !options.Timestamps || options.TailLines == nil || *options.TailLines != 20 {
		t.Fatalf("GetLogs options = %#v", options)
	}
}

func TestPodLogsExplainWhatTheyCannotRead(t *testing.T) {
	t.Parallel()
	source, _ := newPodLogTestSource(
		map[string]string{"VLLM_SR_K8S_DASHBOARD_LOG_SELECTOR": "app in (dashboard"},
		&podLogTestStreams{},
		podLogTestPod("router-a", "router", corev1.PodPending, "router"),
	)

	router := servePodLogs(t, "", source, "component=router")
	if !router.Supported || len(router.Logs) != 0 || router.Message != "1 router pod has not started yet." {
		t.Fatalf("pending router = %#v", router)
	}
	dashboard := servePodLogs(t, "", source, "component=dashboard")
	if dashboard.Message != "VLLM_SR_K8S_DASHBOARD_LOG_SELECTOR is not a valid label selector." {
		t.Fatalf("invalid selector = %#v", dashboard)
	}
	envoy := servePodLogs(t, "", source, "component=envoy")
	if !envoy.Supported || !strings.Contains(envoy.Message, "VLLM_SR_K8S_ENVOY_LOG_SELECTOR") {
		t.Fatalf("unconfigured envoy = %#v", envoy)
	}

	empty, _ := newPodLogTestSource(nil, &podLogTestStreams{})
	none := servePodLogs(t, "", empty, "component=router")
	if none.Message != `No pods match app=router in namespace "vllm-sr".` {
		t.Fatalf("no pods = %#v", none)
	}
}

func TestPodLogsReportFailedPods(t *testing.T) {
	t.Parallel()
	waiting := apierrors.NewBadRequest(`container "router" in pod "router-b" is waiting to start: ContainerCreating`)
	streams := &podLogTestStreams{
		tails: map[string]string{"router-a": "2026-11-18T02:40:00Z ready\n"},
		errs:  map[string]error{"router-b": waiting},
	}
	source, _ := newPodLogTestSource(nil, streams,
		podLogTestPod("router-a", "router", corev1.PodRunning, "router"),
		podLogTestPod("router-b", "router", corev1.PodRunning, "router"),
	)

	partial := servePodLogs(t, "", source, "component=router")
	if partial.Error != "" || len(partial.Logs) != 1 || !strings.Contains(partial.Message, "router-b") {
		t.Fatalf("partial failure = %#v", partial)
	}

	streams.errs["router-a"] = errors.New("connection reset")
	failed := servePodLogs(t, "", source, "component=router")
	if len(failed.Logs) != 0 || !strings.Contains(failed.Error, "router-a") || !strings.Contains(failed.Error, "waiting to start") {
		t.Fatalf("total failure = %#v", failed)
	}
}

func TestPodLogsBoundLinesAndTheResponse(t *testing.T) {
	t.Parallel()
	oversized := "2026-11-18T02:40:00Z " + strings.Repeat("x", maxRuntimeLogLineBytes*2) + "\n"
	long, _ := newPodLogTestSource(nil,
		&podLogTestStreams{tails: map[string]string{"router-a": oversized}},
		podLogTestPod("router-a", "router", corev1.PodRunning, "router"),
	)
	single := servePodLogs(t, "", long, "component=router")
	if len(single.Logs) != 1 {
		t.Fatalf("lines = %d", len(single.Logs))
	}
	if line := single.Logs[0].Line; len(line) != maxRuntimeLogLineBytes || !strings.HasSuffix(line, runtimeLogTruncatedMarker) {
		t.Fatalf("bounded line bytes = %d", len(line))
	}

	escaped := strings.Repeat("2026-11-18T02:40:01Z "+strings.Repeat(`\"`, maxRuntimeLogLineBytes/2)+"\n", 40)
	escaped += "2026-11-18T02:40:02Z newest\n"
	large, _ := newPodLogTestSource(nil,
		&podLogTestStreams{tails: map[string]string{"router-a": escaped}},
		podLogTestPod("router-a", "router", corev1.PodRunning, "router"),
	)
	bounded := servePodLogs(t, "", large, "component=router&lines=1000")
	if len(bounded.Logs) == 0 || len(bounded.Logs) > 40 || bounded.Logs[len(bounded.Logs)-1].Line != "newest" {
		t.Fatalf("bounded response kept %d lines", len(bounded.Logs))
	}
}

func TestLogsHandlerPrefersAMountedSpoolOverPods(t *testing.T) {
	t.Parallel()
	root := newTestLogSpool(t)
	if err := os.WriteFile(filepath.Join(root, "router.log"), []byte("[info] from the spool\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	streams := &podLogTestStreams{}
	source, _ := newPodLogTestSource(nil, streams, podLogTestPod("router-a", "router", corev1.PodRunning, "router"))

	payload := servePodLogs(t, root, source, "component=router")

	if payload.DeploymentType != "local-spool" || len(payload.Logs) != 1 || payload.Logs[0].Pod != "" {
		t.Fatalf("spool answer = %#v", payload)
	}
	if strings.Join(payload.Components, ",") != "router,envoy,dashboard" {
		t.Fatalf("spool components = %v", payload.Components)
	}
	if len(streams.options) != 0 {
		t.Fatal("pods were read while a spool is mounted")
	}
}
