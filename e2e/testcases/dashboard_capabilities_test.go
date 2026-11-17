package testcases

import (
	"strings"
	"testing"
)

func TestCheckDashboardCapabilities(t *testing.T) {
	checkedAt := "2026-10-14T19:24:46Z"
	valid := func() dashboardCapabilities {
		response := dashboardCapabilities{}
		for _, id := range dashboardCapabilityIDs {
			response.Capabilities = append(response.Capabilities, dashboardCapability{
				ID: id, Enabled: true, Configured: true, Reachable: true, Authorized: true, State: "available",
			})
		}
		response.Capabilities[1].CheckedAt = &checkedAt
		return response
	}

	config, err := checkDashboardCapabilities(valid())
	if err != nil || config.ID != "config" {
		t.Fatalf("valid response: config=%+v err=%v", config, err)
	}

	unreachable := valid()
	unreachable.Capabilities[1].Reachable = false
	unreachable.Capabilities[1].State = "unreachable"
	unreachable.Capabilities[1].Reason = "router_unreachable"
	if _, err := checkDashboardCapabilities(unreachable); err != nil {
		t.Fatalf("a deployment that hides the Router management API is valid: %v", err)
	}

	for name, test := range map[string]struct {
		mutate func(*dashboardCapabilities)
		want   string
	}{
		"missing capability": {func(r *dashboardCapabilities) { r.Capabilities = r.Capabilities[2:] }, "omit"},
		"unknown state":      {func(r *dashboardCapabilities) { r.Capabilities[0].State = "broken" }, "unknown state"},
		"available with reason": {func(r *dashboardCapabilities) {
			r.Capabilities[0].Reason = "probe_pending"
		}, "disagrees"},
		"admin unauthorized": {func(r *dashboardCapabilities) {
			r.Capabilities[0].Authorized = false
			r.Capabilities[0].State = "unauthorized"
			r.Capabilities[0].Reason = "permission_required"
		}, "not authorized"},
		"config never checked": {func(r *dashboardCapabilities) { r.Capabilities[1].CheckedAt = nil }, "no completed Router check"},
		"config probe pending": {func(r *dashboardCapabilities) {
			r.Capabilities[1].CheckedAt = nil
			r.Capabilities[1].Reachable = false
			r.Capabilities[1].State = "unreachable"
			r.Capabilities[1].Reason = "probe_pending"
		}, "no completed Router check"},
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
