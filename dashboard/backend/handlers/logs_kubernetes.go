package handlers

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/k8s/configwriter"
)

const (
	serviceAccountNamespaceFile = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
	podLogsTimeout              = 10 * time.Second
	maxPodLogPodsPerComponent   = 8
	maxConcurrentPodLogStreams  = 4
	// The API counts LimitBytes from the start of the tail, so this caps the
	// transfer of oversized lines rather than selecting the newest bytes.
	maxPodLogStreamBytes = 8 << 20
	// Room for the RFC 3339 timestamp the kubelet puts before each line.
	podLogTimestampAllowance = 64
)

// podLogComponents name the label selector variable and default of each
// component. Envoy has no default, so it is read only from pods a deployment
// names.
var podLogComponents = []struct {
	name, selectorEnv, defaultSelector string
}{
	{name: "router", selectorEnv: "VLLM_SR_K8S_ROUTER_LOG_SELECTOR", defaultSelector: "app=router"},
	{name: "envoy", selectorEnv: "VLLM_SR_K8S_ENVOY_LOG_SELECTOR"},
	{name: "dashboard", selectorEnv: "VLLM_SR_K8S_DASHBOARD_LOG_SELECTOR", defaultSelector: "app=dashboard"},
}

// podLogStreamer opens the log tail of one pod container.
type podLogStreamer func(ctx context.Context, namespace, pod string, options *corev1.PodLogOptions) (io.ReadCloser, error)

type podLogComponent struct {
	name        string
	selectorEnv string
	selector    string
	invalid     bool
}

type podLogTarget struct {
	component string
	pod       string
	container string
}

type podLogLine struct {
	at    time.Time
	entry LogEntry
}

// podLogSource reads component logs from pods through the Kubernetes API,
// with the service account of the Dashboard pod.
type podLogSource struct {
	client     kubernetes.Interface
	namespace  string
	components []podLogComponent
	stream     podLogStreamer
}

// newInClusterPodLogSource returns nil outside a Kubernetes cluster.
func newInClusterPodLogSource() *podLogSource {
	config, err := rest.InClusterConfig()
	if err != nil {
		return nil
	}
	client, err := kubernetes.NewForConfig(config)
	if err != nil {
		log.Printf("Logs: the Kubernetes client could not be built: %v", err)
		return nil
	}
	source := newPodLogSource(client, inClusterNamespace(), os.Getenv)
	log.Printf("Logs: without a log spool, pod logs are read from namespace %q", source.namespace)
	return source
}

// inClusterNamespace is the namespace of the Router ConfigMap, else the
// Dashboard pod's own.
func inClusterNamespace() string {
	if namespace := strings.TrimSpace(os.Getenv(configwriter.ConfigMapNamespaceEnv)); namespace != "" {
		return namespace
	}
	data, err := os.ReadFile(serviceAccountNamespaceFile)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

func newPodLogSource(client kubernetes.Interface, namespace string, getenv func(string) string) *podLogSource {
	source := &podLogSource{
		client:    client,
		namespace: namespace,
		stream: func(ctx context.Context, namespace, pod string, options *corev1.PodLogOptions) (io.ReadCloser, error) {
			return client.CoreV1().Pods(namespace).GetLogs(pod, options).Stream(ctx)
		},
	}
	for _, component := range podLogComponents {
		selector := strings.TrimSpace(getenv(component.selectorEnv))
		if selector == "" {
			selector = component.defaultSelector
		}
		if selector == "" {
			continue
		}
		_, err := labels.Parse(selector)
		source.components = append(source.components, podLogComponent{
			name:        component.name,
			selectorEnv: component.selectorEnv,
			selector:    selector,
			invalid:     err != nil,
		})
	}
	return source
}

func (source *podLogSource) read(ctx context.Context, component string, lines int) LogsResponse {
	response := LogsResponse{
		DeploymentType: "kubernetes",
		Service:        component,
		Supported:      true,
		Components:     source.componentNames(),
		Namespace:      source.namespace,
		Logs:           []LogEntry{},
	}
	if source.namespace == "" {
		response.Supported = false
		response.Message = fmt.Sprintf("Set %s to the namespace of the Router pods.", configwriter.ConfigMapNamespaceEnv)
		return response
	}
	selected := source.selectComponents(component)
	if len(selected) == 0 {
		response.Message = fmt.Sprintf("No pod selector is set for %s. Set %s to the label selector of its pods.", component, podLogSelectorEnv(component))
		return response
	}

	ctx, cancel := context.WithTimeout(ctx, podLogsTimeout)
	defer cancel()

	var notes []string
	var targets []podLogTarget
	for _, candidate := range selected {
		if candidate.invalid {
			notes = append(notes, fmt.Sprintf("%s is not a valid label selector.", candidate.selectorEnv))
			continue
		}
		list, err := source.client.CoreV1().Pods(source.namespace).List(ctx, metav1.ListOptions{LabelSelector: candidate.selector})
		if err != nil {
			return source.failed(response, err, "list pods", "list", "pods")
		}
		found, waiting, omitted := podLogTargets(candidate.name, list.Items)
		switch {
		case len(list.Items) == 0:
			notes = append(notes, fmt.Sprintf("No pods match %s in namespace %q.", candidate.selector, source.namespace))
		case waiting > 0:
			notes = append(notes, fmt.Sprintf("%s not started yet.", countPods(waiting, candidate.name, "has", "have")))
		}
		if omitted > 0 {
			notes = append(notes, fmt.Sprintf("Reading %d of %s.", len(found), countPods(len(found)+omitted, candidate.name, "", "")))
		}
		targets = append(targets, found...)
	}

	results := source.readPods(ctx, targets, lines)
	var merged []podLogLine
	var failures []string
	for index, result := range results {
		response.Pods = append(response.Pods, targets[index].pod)
		if result.err != nil {
			if apierrors.IsForbidden(result.err) {
				return source.failed(response, result.err, "read pod logs", "get", "pods/log")
			}
			failures = append(failures, podLogFailureNote(targets[index].pod, result.err))
		}
		merged = append(merged, result.lines...)
	}
	sort.SliceStable(merged, func(left, right int) bool {
		return merged[left].at.Before(merged[right].at)
	})
	if len(merged) > lines {
		merged = merged[len(merged)-lines:]
	}
	for _, line := range merged {
		response.Logs = append(response.Logs, line.entry)
	}

	if len(failures) > 0 && len(failures) == len(targets) {
		response.Error = strings.Join(failures, " ")
	} else {
		notes = append(notes, failures...)
	}
	if len(notes) == 0 && len(response.Logs) == 0 && len(targets) > 0 {
		notes = append(notes, "The pods have no recent log lines.")
	}
	response.Message = strings.Join(notes, " ")
	return response
}

func (source *podLogSource) componentNames() []string {
	names := make([]string, 0, len(source.components))
	for _, component := range source.components {
		names = append(names, component.name)
	}
	return names
}

func (source *podLogSource) selectComponents(component string) []podLogComponent {
	if component == "all" {
		return source.components
	}
	for _, candidate := range source.components {
		if candidate.name == component {
			return []podLogComponent{candidate}
		}
	}
	return nil
}

func podLogSelectorEnv(component string) string {
	for _, candidate := range podLogComponents {
		if candidate.name == component {
			return candidate.selectorEnv
		}
	}
	return ""
}

// failed answers a Kubernetes API error. A permission the service account
// lacks is a deployment fault the answer names; anything else is an error.
func (source *podLogSource) failed(response LogsResponse, err error, action, verb, resource string) LogsResponse {
	response.Logs = []LogEntry{}
	response.Pods = nil
	switch {
	case apierrors.IsForbidden(err):
		response.Supported = false
		response.Message = fmt.Sprintf(
			"The Dashboard's service account cannot %s in namespace %q. Grant it the %s verb on %s in a Role bound to that service account.",
			action, source.namespace, verb, resource,
		)
	case apierrors.IsUnauthorized(err):
		response.Error = "The Kubernetes API rejected the Dashboard's service account credentials."
	case errors.Is(err, context.DeadlineExceeded):
		response.Error = fmt.Sprintf("The Kubernetes API did not answer within %d seconds.", int(podLogsTimeout/time.Second))
	default:
		log.Printf("Logs: Kubernetes API request failed: %s", redactURLsForLog(err.Error()))
		response.Error = "The Kubernetes API could not be reached to read pod logs."
	}
	return response
}

// podLogTargets picks the started pods of one component, running pods first,
// at most maxPodLogPodsPerComponent of them.
func podLogTargets(component string, pods []corev1.Pod) (targets []podLogTarget, waiting, omitted int) {
	started := make([]corev1.Pod, 0, len(pods))
	for _, pod := range pods {
		if pod.Status.Phase == corev1.PodPending {
			waiting++
			continue
		}
		started = append(started, pod)
	}
	sort.SliceStable(started, func(left, right int) bool {
		leftRunning := started[left].Status.Phase == corev1.PodRunning
		rightRunning := started[right].Status.Phase == corev1.PodRunning
		if leftRunning != rightRunning {
			return leftRunning
		}
		return started[left].Name < started[right].Name
	})
	if len(started) > maxPodLogPodsPerComponent {
		omitted = len(started) - maxPodLogPodsPerComponent
		started = started[:maxPodLogPodsPerComponent]
	}
	for _, pod := range started {
		targets = append(targets, podLogTarget{component: component, pod: pod.Name, container: podLogContainer(pod, component)})
	}
	return targets, waiting, omitted
}

// podLogContainer prefers the container named after the component, else the
// pod's first container.
func podLogContainer(pod corev1.Pod, component string) string {
	for _, container := range pod.Spec.Containers {
		if container.Name == component {
			return container.Name
		}
	}
	if len(pod.Spec.Containers) > 0 {
		return pod.Spec.Containers[0].Name
	}
	return ""
}

func countPods(count int, component, singularVerb, pluralVerb string) string {
	text := fmt.Sprintf("%d %s pods", count, component)
	verb := pluralVerb
	if count == 1 {
		text = fmt.Sprintf("1 %s pod", component)
		verb = singularVerb
	}
	if verb == "" {
		return text
	}
	return text + " " + verb
}

type podLogResult struct {
	lines []podLogLine
	err   error
}

func (source *podLogSource) readPods(ctx context.Context, targets []podLogTarget, lines int) []podLogResult {
	results := make([]podLogResult, len(targets))
	slots := make(chan struct{}, maxConcurrentPodLogStreams)
	var wait sync.WaitGroup
	for index, target := range targets {
		wait.Add(1)
		go func() {
			defer wait.Done()
			slots <- struct{}{}
			defer func() { <-slots }()
			results[index].lines, results[index].err = source.readPod(ctx, target, lines)
		}()
	}
	wait.Wait()
	return results
}

func (source *podLogSource) readPod(ctx context.Context, target podLogTarget, lines int) ([]podLogLine, error) {
	tail := int64(lines)
	limit := int64(maxPodLogStreamBytes)
	stream, err := source.stream(ctx, source.namespace, target.pod, &corev1.PodLogOptions{
		Container:  target.container,
		TailLines:  &tail,
		Timestamps: true,
		LimitBytes: &limit,
	})
	if err != nil {
		return nil, err
	}
	defer stream.Close()
	return readPodLogLines(stream, target)
}

// readPodLogLines parses one pod's timestamped tail and redacts it in order.
// A line without a timestamp keeps the one before it.
func readPodLogLines(stream io.Reader, target podLogTarget) ([]podLogLine, error) {
	reader := bufio.NewReaderSize(stream, 32<<10)
	var redactor runtimeLogRedactor
	var at time.Time
	stamp := ""
	lines := []podLogLine{}
	for {
		record, err := readPodLogRecord(reader, maxRuntimeLogLineBytes+podLogTimestampAllowance)
		if len(record) > 0 {
			message := record
			if prefix, rest, found := bytes.Cut(record, []byte{' '}); found {
				if parsed, parseErr := time.Parse(time.RFC3339Nano, string(prefix)); parseErr == nil {
					at, stamp, message = parsed, string(prefix), rest
				}
			}
			if line := boundRuntimeLogLine(message); line != "" {
				if safe, ok := redactor.redact(line); ok && safe != "" {
					lines = append(lines, podLogLine{
						at:    at,
						entry: LogEntry{Line: safe, Service: target.component, Pod: target.pod, Timestamp: stamp},
					})
				}
			}
		}
		if errors.Is(err, io.EOF) {
			return lines, nil
		}
		if err != nil {
			return lines, err
		}
	}
}

// readPodLogRecord reads one line and keeps at most limit bytes of it.
func readPodLogRecord(reader *bufio.Reader, limit int) ([]byte, error) {
	var record []byte
	for {
		chunk, err := reader.ReadSlice('\n')
		if room := limit - len(record); room > 0 {
			record = append(record, chunk[:min(len(chunk), room)]...)
		}
		if !errors.Is(err, bufio.ErrBufferFull) {
			return bytes.TrimSuffix(record, []byte{'\n'}), err
		}
	}
}

func podLogFailureNote(pod string, err error) string {
	switch {
	case apierrors.IsBadRequest(err) || apierrors.IsNotFound(err):
		// The API's own text says which container has not started or that
		// the pod is gone.
		return strings.TrimSuffix(err.Error(), ".") + "."
	case errors.Is(err, context.DeadlineExceeded):
		return fmt.Sprintf("%s did not answer within %d seconds.", pod, int(podLogsTimeout/time.Second))
	default:
		log.Printf("Logs: reading pod %s failed: %s", pod, redactURLsForLog(err.Error()))
		return fmt.Sprintf("The logs of %s could not be read.", pod)
	}
}
