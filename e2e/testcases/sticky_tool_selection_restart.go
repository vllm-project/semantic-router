package testcases

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/remotecommand"
)

const stickySemanticRouterContainer = "semantic-router"

type stickyRouterContainerIdentity struct {
	podUID       types.UID
	containerID  string
	restartCount int32
}

func restartStickySemanticRouterContainer(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	if opts.RestConfig == nil {
		return fmt.Errorf("restart semantic-router container: Kubernetes REST config is required")
	}
	restartCtx, cancel := context.WithTimeout(ctx, restartRecoveryTimeout)
	defer cancel()

	pods, err := client.CoreV1().Pods(semanticRouterNamespace).List(restartCtx, metav1.ListOptions{
		LabelSelector: semanticRouterPodLabel,
	})
	if err != nil {
		return fmt.Errorf("list semantic-router pods: %w", err)
	}
	if len(pods.Items) != 1 {
		return fmt.Errorf("found %d semantic-router pods, want 1", len(pods.Items))
	}

	pod := &pods.Items[0]
	baseline, err := stickyRouterContainerBaseline(pod)
	if err != nil {
		return err
	}
	if opts.Verbose {
		fmt.Printf("[Test] Restarting semantic-router container in pod %s (restart count=%d)\n", pod.Name, baseline.restartCount)
	}

	req := client.CoreV1().RESTClient().Post().
		Resource("pods").
		Namespace(semanticRouterNamespace).
		Name(pod.Name).
		SubResource("exec")
	req.VersionedParams(&corev1.PodExecOptions{
		Container: stickySemanticRouterContainer,
		Command:   []string{"/bin/bash", "-c", "kill -TERM 1"},
		Stderr:    true,
	}, scheme.ParameterCodec)
	executor, err := remotecommand.NewSPDYExecutor(opts.RestConfig, http.MethodPost, req.URL())
	if err != nil {
		return fmt.Errorf("create semantic-router restart executor: %w", err)
	}

	var stderr bytes.Buffer
	streamErr := executor.StreamWithContext(restartCtx, remotecommand.StreamOptions{Stderr: &stderr})
	restartErr := waitForStickyRouterContainerRestart(restartCtx, client, baseline)
	if restartErr != nil {
		if streamErr != nil {
			detail := strings.TrimSpace(stderr.String())
			if detail != "" {
				streamErr = fmt.Errorf("%w: %s", streamErr, detail)
			}
			return errors.Join(restartErr, fmt.Errorf("signal semantic-router container: %w", streamErr))
		}
		return restartErr
	}
	if streamErr != nil && opts.Verbose {
		fmt.Printf("[Test] Container restart closed the exec stream: %v\n", streamErr)
	}
	return nil
}

func stickyRouterContainerBaseline(pod *corev1.Pod) (stickyRouterContainerIdentity, error) {
	status, found := stickyRouterContainerStatus(pod)
	if !found {
		return stickyRouterContainerIdentity{}, fmt.Errorf("semantic-router container status is missing from pod %s", pod.Name)
	}
	if status.ContainerID == "" || status.State.Running == nil || !status.Ready {
		return stickyRouterContainerIdentity{}, fmt.Errorf("semantic-router container in pod %s is not ready", pod.Name)
	}
	return stickyRouterContainerIdentity{
		podUID:       pod.UID,
		containerID:  status.ContainerID,
		restartCount: status.RestartCount,
	}, nil
}

func waitForStickyRouterContainerRestart(
	ctx context.Context,
	client *kubernetes.Clientset,
	baseline stickyRouterContainerIdentity,
) error {
	timer := time.NewTimer(restartRecoveryTimeout)
	defer timer.Stop()
	ticker := time.NewTicker(restartRecoveryInterval)
	defer ticker.Stop()

	var lastState string
	for {
		pods, err := client.CoreV1().Pods(semanticRouterNamespace).List(ctx, metav1.ListOptions{
			LabelSelector: semanticRouterPodLabel,
		})
		if err != nil {
			lastState = err.Error()
		} else if len(pods.Items) != 1 {
			lastState = fmt.Sprintf("found %d semantic-router pods, want 1", len(pods.Items))
		} else {
			restarted, state, stateErr := stickyRouterContainerRestarted(&pods.Items[0], baseline)
			lastState = state
			if stateErr != nil {
				return stateErr
			}
			if restarted {
				return nil
			}
		}

		select {
		case <-ctx.Done():
			return errors.Join(ctx.Err(), fmt.Errorf("last semantic-router container state: %s", lastState))
		case <-timer.C:
			return fmt.Errorf("semantic-router container did not restart after %s: %s", restartRecoveryTimeout, lastState)
		case <-ticker.C:
		}
	}
}

func stickyRouterContainerRestarted(
	pod *corev1.Pod,
	baseline stickyRouterContainerIdentity,
) (bool, string, error) {
	if pod.UID != baseline.podUID {
		return false, "", fmt.Errorf("semantic-router pod was replaced: got UID %s, want %s", pod.UID, baseline.podUID)
	}
	status, found := stickyRouterContainerStatus(pod)
	if !found {
		return false, "container status is missing", nil
	}
	state := fmt.Sprintf(
		"pod_uid=%s container_id=%s restart_count=%d ready=%t running=%t",
		pod.UID,
		status.ContainerID,
		status.RestartCount,
		status.Ready,
		status.State.Running != nil,
	)
	restarted := status.RestartCount > baseline.restartCount &&
		status.ContainerID != "" && status.ContainerID != baseline.containerID &&
		status.Ready && status.State.Running != nil
	return restarted, state, nil
}

func stickyRouterContainerStatus(pod *corev1.Pod) (corev1.ContainerStatus, bool) {
	for _, status := range pod.Status.ContainerStatuses {
		if status.Name == stickySemanticRouterContainer {
			return status, true
		}
	}
	return corev1.ContainerStatus{}, false
}
