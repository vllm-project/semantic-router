package testcases

import (
	"context"
	"fmt"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/modelruntime"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("model-runtime-lifecycle", pkgtestcases.TestCase{
		Description: "The Router starts its managed runtimes in process groups, attaches to an external runtime by served name, and reports each deployment's readiness",
		Tags:        []string{"model-runtime", "lifecycle", "managed", "attached"},
		Fn:          testModelRuntimeLifecycle,
	})
}

func testModelRuntimeLifecycle(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := openModelRuntimeSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	if err = session.waitReady(ctx, append(append([]string(nil), mrManagedDeployments...), mrAttachedDeployments...)...); err != nil {
		return err
	}
	metrics, err := session.routerMetrics(ctx)
	if err != nil {
		return err
	}
	if _, reported := metrics.Value(modelruntime.RouterReadyMetric, map[string]string{"deployment": mrOfflineDeployment}); !reported || metrics.DeploymentReady(mrOfflineDeployment) {
		return fmt.Errorf("%s must be reported and not ready: it has no runtime", mrOfflineDeployment)
	}

	processes, err := checkManagedProcesses(ctx, session)
	if err != nil {
		return err
	}
	attached, err := session.attachedRuntime().Models(ctx)
	if err != nil {
		return fmt.Errorf("attached runtime: %w", err)
	}
	served := modelIDs(attached.Data)
	if strings.Join(served, ",") != "decision-a,feedback-a" {
		return fmt.Errorf("the attached runtime serves %v, want decision-a and feedback-a", served)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"managed_processes": processes, "attached_models": served})
	}
	return nil
}

// checkManagedProcesses requires the "decisions" process group to serve the
// decision fixture alone and the device's CPU models to run in one process
// ("cpu") or spread over several ("cpu-0" …), each deployment in exactly one
// process, ready, with the family and surfaces its fixture declares.
func checkManagedProcesses(ctx context.Context, session *modelRuntimeSession) (map[string][]string, error) {
	var runtimes map[string][]string
	err := modelruntime.Eventually(ctx, mrReadyTimeout, func(ctx context.Context) error {
		var err error
		runtimes, err = session.pod.ManagedRuntimes(ctx, mrSocketDir)
		if err != nil {
			return err
		}
		served := 0
		for _, models := range runtimes {
			served += len(models)
		}
		if served != len(mrManagedDeployments) {
			return fmt.Errorf("managed runtime processes %v serve %d models, want the %d managed deployments", runtimes, served, len(mrManagedDeployments))
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	processes := map[string][]string{}
	owner := map[string]string{}
	for socket, models := range runtimes {
		process := strings.SplitN(filepath.Base(socket), "-", 2)[0]
		sort.Strings(models)
		processes[filepath.Base(socket)] = models
		for _, model := range models {
			if previous, twice := owner[model]; twice {
				return nil, fmt.Errorf("%s runs in two processes, %s and %s", model, previous, socket)
			}
			owner[model] = socket
			switch {
			case process == mrDecisionsProcess && model == mrDecisionDeployment:
			case process == mrDeviceProcess && slices.Contains(mrDeviceGroup, model):
			default:
				return nil, fmt.Errorf("process %s serves %s, which belongs to another group", socket, model)
			}
		}
		listed, err := modelruntime.NewClient(modelruntime.SocketTransport{Target: session.pod, Socket: socket}).Models(ctx)
		if err != nil {
			return nil, err
		}
		for _, card := range listed.Data {
			if err := checkFixtureCard(card); err != nil {
				return nil, fmt.Errorf("%s: %w", socket, err)
			}
		}
	}
	return processes, nil
}

func checkFixtureCard(card modelruntime.ModelCard) error {
	if !card.Ready || card.Device != "cpu" {
		return fmt.Errorf("model %s is ready=%v on %q, want ready on cpu", card.ID, card.Ready, card.Device)
	}
	switch card.ID {
	case mrDecisionDeployment:
		if card.Family != "decision2" || !card.HasSurface("decisions") {
			return fmt.Errorf("model %s is %s serving %v, want decision2 decisions", card.ID, card.Family, card.Surfaces)
		}
	case mrEmbeddingDeployment, mrRerankerDeployment:
		surface := map[string]string{mrEmbeddingDeployment: "embeddings", mrRerankerDeployment: "rerank"}[card.ID]
		if card.Family != "task_heads" || !card.HasSurface(surface) {
			return fmt.Errorf("model %s is %s serving %v, want task_heads %s", card.ID, card.Family, card.Surfaces, surface)
		}
	default:
		if card.Family != "task_heads" || !card.HasSurface("classify") || len(card.Heads) == 0 {
			return fmt.Errorf("model %s is %s serving %v with heads %v, want a task_heads classifier", card.ID, card.Family, card.Surfaces, card.Heads)
		}
	}
	return nil
}

func modelIDs(cards []modelruntime.ModelCard) []string {
	ids := make([]string, 0, len(cards))
	for _, card := range cards {
		ids = append(ids, card.ID)
	}
	sort.Strings(ids)
	return ids
}
