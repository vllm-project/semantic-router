package testcases

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

func TestStickyRouterContainerRestarted(t *testing.T) {
	baseline := stickyRouterContainerIdentity{
		podUID:       types.UID("router-pod"),
		containerID:  "containerd://before",
		restartCount: 2,
	}
	tests := []struct {
		name      string
		pod       *corev1.Pod
		wantReady bool
		wantErr   bool
	}{
		{
			name:      "same container",
			pod:       stickyRouterRestartTestPod("router-pod", "containerd://before", 2, true, true),
			wantReady: false,
		},
		{
			name:      "restarted but not ready",
			pod:       stickyRouterRestartTestPod("router-pod", "containerd://after", 3, false, true),
			wantReady: false,
		},
		{
			name:      "container ID changed without restart count",
			pod:       stickyRouterRestartTestPod("router-pod", "containerd://after", 2, true, true),
			wantReady: false,
		},
		{
			name:      "restart count changed without container ID",
			pod:       stickyRouterRestartTestPod("router-pod", "containerd://before", 3, true, true),
			wantReady: false,
		},
		{
			name:      "restarted and ready",
			pod:       stickyRouterRestartTestPod("router-pod", "containerd://after", 3, true, true),
			wantReady: true,
		},
		{
			name:      "pod replaced",
			pod:       stickyRouterRestartTestPod("replacement-pod", "containerd://after", 0, true, true),
			wantReady: false,
			wantErr:   true,
		},
		{
			name: "container status missing",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{UID: types.UID("router-pod")},
			},
			wantReady: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ready, _, err := stickyRouterContainerRestarted(test.pod, baseline)
			if (err != nil) != test.wantErr {
				t.Fatalf("stickyRouterContainerRestarted() error = %v, wantErr %t", err, test.wantErr)
			}
			if ready != test.wantReady {
				t.Fatalf("stickyRouterContainerRestarted() ready = %t, want %t", ready, test.wantReady)
			}
		})
	}
}

func stickyRouterRestartTestPod(
	uid types.UID,
	containerID string,
	restartCount int32,
	ready bool,
	running bool,
) *corev1.Pod {
	state := corev1.ContainerState{}
	if running {
		state.Running = &corev1.ContainerStateRunning{}
	}
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: uid},
		Status: corev1.PodStatus{ContainerStatuses: []corev1.ContainerStatus{{
			Name:         stickySemanticRouterContainer,
			ContainerID:  containerID,
			RestartCount: restartCount,
			Ready:        ready,
			State:        state,
		}}},
	}
}
