package modelservice

import (
	"errors"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestProcessExitsCountAsRestartsOfEveryDeployment(t *testing.T) {
	plan := planProcesses(map[string]config.ModelDeployment{
		"domain": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Vela-1.0-Encoder-307M-Domain", Device: "cpu"},
		"guard":  {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Vela-1.0-Encoder-307M-Guard", Device: "cpu"},
	}, []string{"vllm-srun"}, "", 4, "")[0]
	client, err := NewClient("http://127.0.0.1:1")
	if err != nil {
		t.Fatal(err)
	}
	g := newGroup(plan, client, true)
	g.processExited(errors.New("killed"), time.Minute)
	g.processExited(errors.New("killed"), time.Minute)

	statuses := g.status()
	if len(statuses) != 1 {
		t.Fatalf("statuses = %+v", statuses)
	}
	for _, status := range statuses {
		if status.Restarts != 2 || status.State != "restarting" || status.Ready {
			t.Fatalf("%s after two exits: %+v", status.Name, status)
		}
	}
}
