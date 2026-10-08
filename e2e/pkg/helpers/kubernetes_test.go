package helpers

import (
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestFirstReadyPodSkipsUnavailablePods(t *testing.T) {
	terminating := portForwardTestPod("old-terminating", corev1.PodRunning, corev1.ConditionTrue)
	deletedAt := metav1.Now()
	terminating.DeletionTimestamp = &deletedAt

	for _, unavailable := range []corev1.Pod{
		terminating,
		portForwardTestPod("running-not-ready", corev1.PodRunning, corev1.ConditionFalse),
		portForwardTestPod("running-ready-unknown", corev1.PodRunning, corev1.ConditionUnknown),
		portForwardTestPod("running-ready-missing", corev1.PodRunning, ""),
		portForwardTestPod("pending", corev1.PodPending, corev1.ConditionTrue),
		portForwardTestPod("completed", corev1.PodSucceeded, corev1.ConditionTrue),
	} {
		t.Run(unavailable.Name, func(t *testing.T) {
			ready := portForwardTestPod("new-ready", corev1.PodRunning, corev1.ConditionTrue)
			pod, err := firstReadyPod([]corev1.Pod{unavailable, ready}, "default", "provider-mocker")
			if err != nil {
				t.Fatal(err)
			}
			if pod.Name != ready.Name {
				t.Fatalf("selected %q instead of ready replacement %q", pod.Name, ready.Name)
			}
		})
	}
}

func TestFirstReadyPodRejectsUnavailableServices(t *testing.T) {
	terminating := portForwardTestPod("terminating", corev1.PodRunning, corev1.ConditionTrue)
	deletedAt := metav1.Now()
	terminating.DeletionTimestamp = &deletedAt
	for _, tc := range []struct {
		name string
		pods []corev1.Pod
	}{
		{name: "empty"},
		{name: "terminating-only", pods: []corev1.Pod{terminating}},
		{name: "no-ready-pods", pods: []corev1.Pod{
			portForwardTestPod("running", corev1.PodRunning, corev1.ConditionFalse),
			portForwardTestPod("pending", corev1.PodPending, corev1.ConditionTrue),
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pod, err := firstReadyPod(tc.pods, "fixtures", "provider-mocker")
			if pod != nil || err == nil || !strings.Contains(err.Error(), "fixtures/provider-mocker") {
				t.Fatalf("expected unavailable service error, got pod=%v err=%v", pod, err)
			}
		})
	}
}

func TestFirstReadyPodPreservesOrderOfReadyPods(t *testing.T) {
	first := portForwardTestPod("first-ready", corev1.PodRunning, corev1.ConditionTrue)
	second := portForwardTestPod("second-ready", corev1.PodRunning, corev1.ConditionTrue)
	pod, err := firstReadyPod([]corev1.Pod{first, second}, "default", "provider-mocker")
	if err != nil || pod == nil || pod.Name != first.Name {
		t.Fatalf("expected first ready pod, got pod=%v err=%v", pod, err)
	}
}

func portForwardTestPod(name string, phase corev1.PodPhase, ready corev1.ConditionStatus) corev1.Pod {
	pod := corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Status:     corev1.PodStatus{Phase: phase},
	}
	if ready != "" {
		pod.Status.Conditions = []corev1.PodCondition{{Type: corev1.PodReady, Status: ready}}
	}
	return pod
}
