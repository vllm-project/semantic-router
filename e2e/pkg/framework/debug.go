package framework

import (
	"context"
	"fmt"
	"io"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

// PrintAllPodsStatus prints detailed status and logs for all pods in all namespaces
// This is useful for debugging when tests fail
func PrintAllPodsStatus(ctx context.Context, client *kubernetes.Clientset) {
	fmt.Printf("\n========== All Pods Status and Logs ==========\n")

	// Get all pods from all namespaces
	pods, err := client.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	if err != nil {
		fmt.Printf("Failed to list pods: %v\n", err)
		return
	}

	fmt.Printf("Total pods: %d\n", len(pods.Items))

	for _, pod := range pods.Items {
		fmt.Printf("\n--- Pod: %s/%s ---\n", pod.Namespace, pod.Name)
		fmt.Printf("Phase: %s\n", pod.Status.Phase)
		fmt.Printf("Node: %s\n", pod.Spec.NodeName)
		fmt.Printf("Start Time: %v\n", pod.Status.StartTime)

		// Print conditions
		fmt.Printf("\nConditions:\n")
		for _, condition := range pod.Status.Conditions {
			fmt.Printf("  - Type: %s, Status: %s, Reason: %s, Message: %s\n",
				condition.Type, condition.Status, condition.Reason, condition.Message)
		}

		// Print container statuses
		fmt.Printf("\nContainer Statuses:\n")
		for _, containerStatus := range pod.Status.ContainerStatuses {
			fmt.Printf("  - Container: %s\n", containerStatus.Name)
			fmt.Printf("    Ready: %v\n", containerStatus.Ready)
			fmt.Printf("    RestartCount: %d\n", containerStatus.RestartCount)
			fmt.Printf("    Image: %s\n", containerStatus.Image)

			if containerStatus.State.Waiting != nil {
				fmt.Printf("    State: Waiting\n")
				fmt.Printf("    Reason: %s\n", containerStatus.State.Waiting.Reason)
				fmt.Printf("    Message: %s\n", containerStatus.State.Waiting.Message)
			} else if containerStatus.State.Running != nil {
				fmt.Printf("    State: Running\n")
				fmt.Printf("    Started At: %v\n", containerStatus.State.Running.StartedAt)
			} else if containerStatus.State.Terminated != nil {
				fmt.Printf("    State: Terminated\n")
				fmt.Printf("    Reason: %s\n", containerStatus.State.Terminated.Reason)
				fmt.Printf("    Message: %s\n", containerStatus.State.Terminated.Message)
				fmt.Printf("    Exit Code: %d\n", containerStatus.State.Terminated.ExitCode)
			}

			if containerStatus.LastTerminationState.Terminated != nil {
				fmt.Printf("    Last Termination:\n")
				fmt.Printf("      Reason: %s\n", containerStatus.LastTerminationState.Terminated.Reason)
				fmt.Printf("      Message: %s\n", containerStatus.LastTerminationState.Terminated.Message)
				fmt.Printf("      Exit Code: %d\n", containerStatus.LastTerminationState.Terminated.ExitCode)
			}
		}

		// Print init container statuses if any
		if len(pod.Status.InitContainerStatuses) > 0 {
			fmt.Printf("\nInit Container Statuses:\n")
			for _, containerStatus := range pod.Status.InitContainerStatuses {
				fmt.Printf("  - Container: %s\n", containerStatus.Name)
				fmt.Printf("    Ready: %v\n", containerStatus.Ready)
				fmt.Printf("    RestartCount: %d\n", containerStatus.RestartCount)

				if containerStatus.State.Waiting != nil {
					fmt.Printf("    State: Waiting - %s: %s\n",
						containerStatus.State.Waiting.Reason,
						containerStatus.State.Waiting.Message)
				} else if containerStatus.State.Running != nil {
					fmt.Printf("    State: Running\n")
				} else if containerStatus.State.Terminated != nil {
					fmt.Printf("    State: Terminated - %s: %s (Exit Code: %d)\n",
						containerStatus.State.Terminated.Reason,
						containerStatus.State.Terminated.Message,
						containerStatus.State.Terminated.ExitCode)
				}
			}
		}

		// Print events for this pod
		fmt.Printf("\nRecent Events:\n")
		events, err := client.CoreV1().Events(pod.Namespace).List(ctx, metav1.ListOptions{
			FieldSelector: fmt.Sprintf("involvedObject.name=%s,involvedObject.kind=Pod", pod.Name),
		})
		if err == nil && len(events.Items) > 0 {
			// Sort events by last timestamp (most recent first)
			for i := len(events.Items) - 1; i >= 0 && i >= len(events.Items)-10; i-- {
				event := events.Items[i]
				fmt.Printf("  - [%s] %s: %s (Count: %d)\n",
					event.LastTimestamp.Format("15:04:05"),
					event.Reason,
					event.Message,
					event.Count)
			}
		} else if err != nil {
			fmt.Printf("  Failed to get events: %v\n", err)
		} else {
			fmt.Printf("  No events found\n")
		}

		// Print container logs
		fmt.Printf("\nContainer Logs:\n")
		for _, container := range pod.Spec.Containers {
			fmt.Printf("\n  --- Logs for container: %s ---\n", container.Name)
			logOptions := &corev1.PodLogOptions{
				Container: container.Name,
				TailLines: int64Ptr(50), // Last 50 lines
			}

			req := client.CoreV1().Pods(pod.Namespace).GetLogs(pod.Name, logOptions)
			logs, err := req.Stream(ctx)
			if err != nil {
				fmt.Printf("  Failed to get logs: %v\n", err)
				continue
			}
			defer logs.Close()

			logBytes, err := io.ReadAll(logs)
			if err != nil {
				fmt.Printf("  Failed to read logs: %v\n", err)
				continue
			}

			if len(logBytes) == 0 {
				fmt.Printf("  (no logs available)\n")
			} else {
				fmt.Printf("%s\n", string(logBytes))
			}
		}

		// Print init container logs if any failed
		for _, containerStatus := range pod.Status.InitContainerStatuses {
			if !containerStatus.Ready {
				fmt.Printf("\n  --- Logs for init container: %s ---\n", containerStatus.Name)
				logOptions := &corev1.PodLogOptions{
					Container: containerStatus.Name,
					TailLines: int64Ptr(50),
				}

				req := client.CoreV1().Pods(pod.Namespace).GetLogs(pod.Name, logOptions)
				logs, err := req.Stream(ctx)
				if err != nil {
					fmt.Printf("  Failed to get logs: %v\n", err)
					continue
				}
				defer logs.Close()

				logBytes, err := io.ReadAll(logs)
				if err != nil {
					fmt.Printf("  Failed to read logs: %v\n", err)
					continue
				}

				if len(logBytes) == 0 {
					fmt.Printf("  (no logs available)\n")
				} else {
					fmt.Printf("%s\n", string(logBytes))
				}
			}
		}
	}
	fmt.Printf("\n========================================\n\n")
}

// int64Ptr returns a pointer to an int64 value
func int64Ptr(i int64) *int64 {
	return &i
}
