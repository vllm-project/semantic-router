package testcases

import (
	"strings"
	"testing"
)

func TestCheckDashboardCapabilities(t *testing.T) {
	valid := func() dashboardCapabilities {
		response := dashboardCapabilities{}
		for _, id := range dashboardCapabilityIDs {
			response.Capabilities = append(response.Capabilities, dashboardCapability{
				ID: id, Enabled: true, Configured: true, Reachable: true, Authorized: true, State: "available",
			})
		}
		return response
	}

	config, err := checkDashboardCapabilities(valid())
	if err != nil || config.ID != "config" {
		t.Fatalf("valid response: config=%+v err=%v", config, err)
	}

	for name, test := range map[string]struct {
		mutate func(*dashboardCapabilities)
		want   string
	}{
		"missing capability": {func(r *dashboardCapabilities) { r.Capabilities = r.Capabilities[1:] }, "omit"},
		"unknown state":      {func(r *dashboardCapabilities) { r.Capabilities[0].State = "broken" }, "unknown state"},
		"available with reason": {func(r *dashboardCapabilities) {
			r.Capabilities[0].Reason = "probe_pending"
		}, "disagrees"},
		"admin unauthorized": {func(r *dashboardCapabilities) {
			r.Capabilities[0].Authorized = false
			r.Capabilities[0].State = "unauthorized"
			r.Capabilities[0].Reason = "permission_required"
		}, "not authorized"},
	} {
		t.Run(name, func(t *testing.T) {
			response := valid()
			test.mutate(&response)
			if _, err := checkDashboardCapabilities(response); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("err = %v, want %q", err, test.want)
			}
		})
	}
}
