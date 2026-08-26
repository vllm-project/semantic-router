package framework

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

const defaultRouterDiagnosticsNamespace = "vllm-semantic-router-system"

type routerDiagnosticsNamespaceProvider interface {
	RouterDiagnosticsNamespace() string
}

func (r *Runner) routerDiagnosticsNamespace() string {
	provider, ok := r.profile.(routerDiagnosticsNamespaceProvider)
	if !ok {
		return defaultRouterDiagnosticsNamespace
	}
	namespace := strings.TrimSpace(provider.RouterDiagnosticsNamespace())
	if namespace == "" {
		return defaultRouterDiagnosticsNamespace
	}
	return namespace
}

// collectSemanticRouterLogs snapshots Router logs and namespace events before cleanup.
func (r *Runner) collectSemanticRouterLogs(
	ctx context.Context,
	client *kubernetes.Clientset,
	namespace string,
) error {
	pods, err := client.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("failed to list semantic-router pods: %w", err)
	}

	var allLogs strings.Builder
	allLogs.WriteString("========================================\n")
	fmt.Fprintf(&allLogs, "Semantic Router Diagnostics: %s\n", namespace)
	fmt.Fprintf(&allLogs, "Captured: %s\n", time.Now().UTC().Format(time.RFC3339))
	allLogs.WriteString("========================================\n\n")
	if len(pods.Items) == 0 {
		allLogs.WriteString("(no pods found)\n\n")
	}
	for _, pod := range pods.Items {
		r.appendPodLogs(ctx, client, &allLogs, pod)
	}
	r.appendNamespaceEvents(ctx, client, &allLogs, namespace)
	return r.appendRouterDiagnostics(allLogs.String())
}

func (r *Runner) appendRouterDiagnostics(content string) error {
	logFile, err := os.OpenFile(
		"semantic-router-logs.txt",
		os.O_CREATE|os.O_WRONLY|os.O_APPEND,
		0o644,
	)
	if err != nil {
		return fmt.Errorf("open semantic-router diagnostics: %w", err)
	}
	if _, err := logFile.WriteString(content); err != nil {
		_ = logFile.Close()
		return fmt.Errorf("append semantic-router diagnostics: %w", err)
	}
	if err := logFile.Close(); err != nil {
		return fmt.Errorf("close semantic-router diagnostics: %w", err)
	}
	r.log("✅ Semantic router logs saved to: semantic-router-logs.txt")
	return nil
}

func (r *Runner) appendPodLogs(
	ctx context.Context,
	client *kubernetes.Clientset,
	allLogs *strings.Builder,
	pod corev1.Pod,
) {
	fmt.Fprintf(allLogs, "=== Pod: %s (Namespace: %s) ===\n", pod.Name, pod.Namespace)
	fmt.Fprintf(allLogs, "Status: %s\n", pod.Status.Phase)
	fmt.Fprintf(allLogs, "Node: %s\n", pod.Spec.NodeName)
	if pod.Status.StartTime != nil {
		fmt.Fprintf(allLogs, "Started: %s\n", pod.Status.StartTime.Format(time.RFC3339))
	}
	allLogs.WriteString("\n")
	for _, container := range pod.Spec.Containers {
		r.appendContainerLogs(ctx, client, allLogs, pod, container.Name)
	}
	allLogs.WriteString("\n")
}

func (r *Runner) appendContainerLogs(
	ctx context.Context,
	client *kubernetes.Clientset,
	allLogs *strings.Builder,
	pod corev1.Pod,
	containerName string,
) {
	r.appendContainerLogSnapshot(ctx, client, allLogs, pod, containerName, false)
	if containerRestartCount(pod, containerName) > 0 {
		r.appendContainerLogSnapshot(ctx, client, allLogs, pod, containerName, true)
	}
}

func (r *Runner) appendContainerLogSnapshot(
	ctx context.Context,
	client *kubernetes.Clientset,
	allLogs *strings.Builder,
	pod corev1.Pod,
	containerName string,
	previous bool,
) {
	label := "Current"
	if previous {
		label = "Previous"
	}
	fmt.Fprintf(allLogs, "--- %s logs: %s ---\n", label, containerName)

	logOptions := &corev1.PodLogOptions{Container: containerName, Previous: previous}
	logs, err := client.CoreV1().Pods(pod.Namespace).GetLogs(pod.Name, logOptions).Stream(ctx)
	if err != nil {
		fmt.Fprintf(allLogs, "Error getting logs: %v\n\n", err)
		return
	}
	logBytes, readErr := io.ReadAll(logs)
	if closeErr := logs.Close(); closeErr != nil {
		r.log("Warning: failed to close log stream for %s/%s: %v", pod.Name, containerName, closeErr)
	}
	if readErr != nil {
		fmt.Fprintf(allLogs, "Error reading logs: %v\n\n", readErr)
		return
	}
	if len(logBytes) == 0 {
		allLogs.WriteString("(no logs available)\n\n")
		return
	}
	allLogs.Write(logBytes)
	allLogs.WriteString("\n\n")
}

func containerRestartCount(pod corev1.Pod, containerName string) int32 {
	for _, status := range pod.Status.ContainerStatuses {
		if status.Name == containerName {
			return status.RestartCount
		}
	}
	return 0
}

func (r *Runner) appendNamespaceEvents(
	ctx context.Context,
	client *kubernetes.Clientset,
	allLogs *strings.Builder,
	namespace string,
) {
	allLogs.WriteString("=== Namespace events ===\n")
	events, err := client.CoreV1().Events(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		fmt.Fprintf(allLogs, "Error getting events: %v\n\n", err)
		return
	}
	if len(events.Items) == 0 {
		allLogs.WriteString("(no events found)\n\n")
		return
	}
	for _, event := range events.Items {
		fmt.Fprintf(
			allLogs,
			"%s %s/%s %s: %s\n",
			event.LastTimestamp.UTC().Format(time.RFC3339),
			event.InvolvedObject.Kind,
			event.InvolvedObject.Name,
			event.Reason,
			event.Message,
		)
	}
	allLogs.WriteString("\n")
}
