package helpers

import (
	"strings"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestCheckDeploymentReadyReplicasRequiresCompleteObservedRollout(t *testing.T) {
	replicas := int32(2)
	ready := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: "semantic-router", Namespace: "router-system", Generation: 3,
		},
		Spec: appsv1.DeploymentSpec{Replicas: &replicas},
		Status: appsv1.DeploymentStatus{
			ObservedGeneration: 3,
			Replicas:           2,
			UpdatedReplicas:    2,
			ReadyReplicas:      2,
			AvailableReplicas:  2,
		},
	}
	tests := []struct {
		name    string
		mutate  func(*appsv1.Deployment)
		wantErr string
	}{
		{name: "complete rollout"},
		{
			name: "stale generation",
			mutate: func(deployment *appsv1.Deployment) {
				deployment.Status.ObservedGeneration = 2
			},
			wantErr: "has not been observed",
		},
		{
			name: "one replica not ready",
			mutate: func(deployment *appsv1.Deployment) {
				deployment.Status.ReadyReplicas = 1
				deployment.Status.AvailableReplicas = 1
				deployment.Status.UnavailableReplicas = 1
			},
			wantErr: "rollout is incomplete",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			candidate := ready.DeepCopy()
			if test.mutate != nil {
				test.mutate(candidate)
			}
			err := checkDeploymentReadyReplicas(candidate, replicas)
			if test.wantErr == "" && err != nil {
				t.Fatalf("CheckDeploymentReadyReplicas() error = %v", err)
			}
			if test.wantErr != "" && (err == nil || !strings.Contains(err.Error(), test.wantErr)) {
				t.Fatalf("CheckDeploymentReadyReplicas() error = %v, want %q", err, test.wantErr)
			}
		})
	}
}
