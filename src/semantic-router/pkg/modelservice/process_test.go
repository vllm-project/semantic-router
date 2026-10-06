package modelservice

import (
	"fmt"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// planned describes managed plans as "<process> <model>@<device>,... threads=<n>", sorted.
func planned(plans []*processPlan) []string {
	var lines []string
	for _, plan := range plans {
		if plan.endpoint != "" {
			continue
		}
		models := make([]string, 0, len(plan.models))
		for _, model := range plan.models {
			models = append(models, model.Name+"@"+model.Device)
		}
		sort.Strings(models)
		lines = append(lines, fmt.Sprintf("%s %s threads=%d", plan.name, strings.Join(models, ","), plan.threads))
	}
	sort.Strings(lines)
	return lines
}

func TestPlanProcessesPlansAutoDeploymentsOnTheResolvedDevice(t *testing.T) {
	encoder := func(name, device, process string) config.ModelDeployment {
		return config.ModelDeployment{Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Vela-1.0-Encoder-307M-" + name, Device: device, Process: process}
	}
	kai := config.ModelDeployment{Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Decision-2.0-Kai-0.6B", Device: "rocm:0"}
	attached := config.ModelDeployment{Provider: config.ModelRuntimeProvider, Endpoint: "http://shared:8100"}
	cases := []struct {
		name        string
		auto        string
		cores       int
		deployments map[string]config.ModelDeployment
		want        []string
		// shared maps a deployment to the deployment whose loaded model it calls.
		shared map[string]string
	}{
		{
			name: "auto on a CPU-only host spreads like cpu with thread shares", auto: "cpu", cores: 16,
			deployments: map[string]config.ModelDeployment{
				// An empty device defaults to auto.
				"domain": encoder("Domain", "", ""), "guard": encoder("Guard", "auto", ""), "pii": encoder("PII", "auto", ""),
				"domain-cpu": encoder("Domain", "cpu", ""), "remote": attached,
			},
			want:   []string{"cpu-0 domain@cpu threads=6", "cpu-1 guard@cpu threads=6", "cpu-2 pii@cpu threads=6"},
			shared: map[string]string{"domain-cpu": "domain"},
		},
		{
			name: "auto on a GPU host shares the GPU's process and keeps auto placement", auto: "rocm:0", cores: 16,
			deployments: map[string]config.ModelDeployment{
				"domain": encoder("Domain", "auto", ""), "guard": encoder("Guard", "rocm:0", ""), "kai": {Provider: config.ModelRuntimeProvider, Artifact: kai.Artifact},
				"pii": encoder("PII", "cpu", ""),
			},
			want: []string{"cpu pii@cpu threads=16", "rocm:0 domain@auto,guard@rocm:0,kai@auto threads=0"},
		},
		{
			name: "an explicit process wins and on the CPU still gets a thread share", auto: "cpu", cores: 8,
			deployments: map[string]config.ModelDeployment{
				"decider": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Decision-2.0-Eos-0.8B", Process: "decisions"},
				"domain":  encoder("Domain", "auto", ""), "guard": encoder("Guard", "cpu", ""), "kai": kai,
			},
			want: []string{"cpu-0 domain@cpu threads=3", "cpu-1 guard@cpu threads=3", "decisions decider@cpu threads=3", "rocm:0 kai@rocm:0 threads=0"},
		},
		{
			name: "an explicit process and device on a GPU host", auto: "rocm:0", cores: 8,
			deployments: map[string]config.ModelDeployment{
				"decider": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Decision-2.0-Eos-0.8B", Process: "decisions"},
				"domain":  encoder("Domain", "auto", ""), "guard": encoder("Guard", "cpu", ""), "kai": kai,
			},
			want: []string{"cpu guard@cpu threads=8", "decisions decider@auto threads=0", "rocm:0 domain@auto,kai@rocm:0 threads=0"},
		},
		{
			name: "an unresolved auto keeps one auto process", auto: "", cores: 16,
			deployments: map[string]config.ModelDeployment{
				"domain": encoder("Domain", "auto", ""), "guard": encoder("Guard", "", ""), "pii": encoder("PII", "cpu", ""),
			},
			want: []string{"auto domain@auto,guard@auto threads=0", "cpu pii@cpu threads=16"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			plans := planProcesses(tc.deployments, []string{"vllm-srun"}, "", tc.cores, tc.auto)
			if got := planned(plans); !slices.Equal(got, tc.want) {
				t.Fatalf("plans\n got %q\nwant %q", got, tc.want)
			}
			for deployment, model := range tc.shared {
				found := false
				for _, plan := range plans {
					if served, ok := plan.members[deployment]; ok {
						found = served == model
					}
				}
				if !found {
					t.Fatalf("%s must call %s's loaded model", deployment, model)
				}
			}
		})
	}
}
