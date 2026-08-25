package accessruntime

import (
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

func TestHeartbeatUsesOnlyTheOpaqueAdmissionPlan(t *testing.T) {
	runtime, issued, _, engine := testRuntime(t)
	authentication := authenticate(t, runtime, issued)
	lease := time.Minute
	admission, err := runtime.Admit(context.Background(), AdmissionRequest{
		Session: authentication.Session,
		Target: Target{
			ResourceType: accesscontrol.GrantResourceEntrypoint,
			ResourceID:   "entry-chat",
			Permission:   accesscontrol.GrantPermissionInvoke,
		},
		AdmissionID: "heartbeat-admission", RequestDigest: "heartbeat-request", LeaseDuration: lease,
	})
	if err != nil || !admission.Result.Allowed() {
		t.Fatalf("Admit() = (%+v, %v)", admission, err)
	}
	deadline := time.Now().UTC().Add(lease)
	engine.heartbeatResult = quotaruntime.AdmissionHeartbeatResult{Deadline: deadline}
	result, err := runtime.Heartbeat(context.Background(), admission)
	if err != nil || !result.Deadline.Equal(deadline) {
		t.Fatalf("Heartbeat() = (%+v, %v)", result, err)
	}
	request := engine.heartbeatRequest
	if request == nil || request.Partition != admission.Tenant.QuotaPartition ||
		request.AdmissionID != admission.Tenant.AdmissionID ||
		request.AdmissionDigest != admission.RequestDigest ||
		request.PlanDigest != admission.Result.PlanDigest ||
		request.LeaseDuration != lease || len(request.Rules) != len(admission.Rules) {
		t.Fatalf("opaque heartbeat request = %+v", request)
	}

	modified := admission
	modified.RequestDigest = "different-request"
	if _, err := runtime.Heartbeat(context.Background(), modified); err == nil {
		t.Fatal("Heartbeat() accepted a modified admission")
	}
}
