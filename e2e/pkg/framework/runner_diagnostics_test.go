package framework

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
)

func TestRouterDiagnosticsNamespaceUsesProfileOverride(t *testing.T) {
	runner := &Runner{profile: &diagnosticsProfile{}}
	if got := runner.routerDiagnosticsNamespace(); got != "custom-router-system" {
		t.Fatalf("routerDiagnosticsNamespace() = %q", got)
	}
}

func TestContainerRestartCount(t *testing.T) {
	pod := corev1.Pod{Status: corev1.PodStatus{ContainerStatuses: []corev1.ContainerStatus{
		{Name: "semantic-router", RestartCount: 3},
	}}}
	if got := containerRestartCount(pod, "semantic-router"); got != 3 {
		t.Fatalf("containerRestartCount() = %d", got)
	}
	if got := containerRestartCount(pod, "missing"); got != 0 {
		t.Fatalf("missing container restart count = %d", got)
	}
}

type diagnosticsProfile struct{ stubProfile }

func (*diagnosticsProfile) RouterDiagnosticsNamespace() string {
	return "custom-router-system"
}
