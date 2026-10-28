package testcases

import (
	"context"
	"fmt"
	"net/http"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/modelruntime"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("model-runtime-lifecycle", pkgtestcases.TestCase{
		Description: "The frontend starts an independent worker for each managed logical deployment, resolves auto placement on the host, attaches to externally owned multi-model runtimes by served name, and reports readiness",
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
	if strings.Join(served, ",") != "decision-a,feedback-a,vela2-a" {
		return fmt.Errorf("the attached runtime serves %v, want decision-a, feedback-a and vela2-a", served)
	}
	if err = checkLivenessAndReadiness(ctx, session.attachedRuntime(), attached.APIVersion); err != nil {
		return fmt.Errorf("attached runtime: %w", err)
	}
	inventory, err := checkInventory(ctx, session)
	if err != nil {
		return fmt.Errorf("model inventory: %w", err)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"managed_processes": processes, "attached_models": served, "inventory_states": inventory})
	}
	return nil
}

// mrInventoryDeployment is the part of one GET /api/v1/inventory/model-runtime
// entry the lifecycle case reads.
type mrInventoryDeployment struct {
	Name       string `json:"name"`
	Managed    bool   `json:"managed"`
	Process    string `json:"process"`
	ServedName string `json:"served_name"`
	Endpoint   string `json:"endpoint"`
	Ready      bool   `json:"ready"`
	State      string `json:"state"`
	Family     string `json:"family"`
	Heads      []struct {
		Labels []string `json:"labels"`
	} `json:"heads"`
}

// checkLivenessAndReadiness requires a ready runtime to answer GET /health/live
// as alive and GET /health as ready, both under the contract version its
// /v1/models reports.
func checkLivenessAndReadiness(ctx context.Context, runtime *modelruntime.Client, apiVersion string) error {
	live, err := runtime.Liveness(ctx)
	if err != nil {
		return err
	}
	health, status, err := runtime.Health(ctx)
	if err != nil {
		return err
	}
	if apiVersion == "" || live.Status != "alive" || live.APIVersion != apiVersion ||
		status != http.StatusOK || health.Status != "ready" || health.APIVersion != apiVersion {
		return fmt.Errorf("/health/live is %+v and /health %d %+v, want alive and ready under api_version %q", live, status, health, apiVersion)
	}
	return nil
}

// checkInventory requires the Router's model inventory to list every
// deployment once: managed ones ready in independent workers with the card
// their runtime serves, attached ones by served name and transport only, and
// the one without a runtime not ready. It returns each deployment's state.
func checkInventory(ctx context.Context, session *modelRuntimeSession) (map[string]string, error) {
	var inventory struct {
		Deployments []mrInventoryDeployment `json:"deployments"`
		Count       int                     `json:"count"`
	}
	if err := session.getAPI(ctx, "/api/v1/inventory/model-runtime", &inventory); err != nil {
		return nil, err
	}
	listed := map[string]mrInventoryDeployment{}
	for _, deployment := range inventory.Deployments {
		if _, twice := listed[deployment.Name]; twice {
			return nil, fmt.Errorf("%s is listed twice", deployment.Name)
		}
		listed[deployment.Name] = deployment
	}
	want := append(append(append([]string(nil), mrManagedDeployments...), mrAttachedDeployments...), mrOfflineDeployment)
	if inventory.Count != len(inventory.Deployments) || len(listed) != len(want) {
		return nil, fmt.Errorf("lists %d deployments (count %d), want %v", len(listed), inventory.Count, want)
	}
	states := map[string]string{}
	for _, name := range want {
		deployment, ok := listed[name]
		if !ok {
			return nil, fmt.Errorf("%s is not listed", name)
		}
		states[name] = deployment.State
		if strings.Contains(deployment.Endpoint, mrSocketDir) {
			return nil, fmt.Errorf("%s shows its socket path %q", name, deployment.Endpoint)
		}
	}
	processOwners := map[string]string{}
	for _, name := range mrManagedDeployments {
		deployment := listed[name]
		wantFamily := "task_heads"
		if name == mrDecisionDeployment {
			wantFamily = "decision2"
		}
		if !deployment.Managed || !deployment.Ready || deployment.Process == "" || deployment.Family != wantFamily {
			return nil, fmt.Errorf("%s is %+v, want a ready independent managed %s worker", name, deployment, wantFamily)
		}
		if previous, exists := processOwners[deployment.Process]; exists {
			return nil, fmt.Errorf("independent deployments %s and %s share process %s", previous, name, deployment.Process)
		}
		processOwners[deployment.Process] = name
	}
	if err := checkInventoryLabels(ctx, session, listed[mrDomainDeployment]); err != nil {
		return nil, err
	}
	for name, servedName := range map[string]string{mrAttachedDecisions: "decision-a", mrAttachedFeedback: "feedback-a", mrAttachedVela2: "vela2-a"} {
		deployment := listed[name]
		if deployment.Managed || !deployment.Ready || deployment.ServedName != servedName ||
			deployment.Endpoint != "http://"+mrAttachedService+"."+modelruntime.RouterNamespace+".svc.cluster.local:8100" {
			return nil, fmt.Errorf("%s is %+v, want ready, attached as %s by scheme and host", name, deployment, servedName)
		}
	}
	if offline := listed[mrOfflineDeployment]; offline.Managed || offline.Ready {
		return nil, fmt.Errorf("%s is %+v, want attached and not ready", mrOfflineDeployment, offline)
	}
	return states, nil
}

// checkInventoryLabels requires the inventory to show a classifier's labels as
// the runtime serving it reports them.
func checkInventoryLabels(ctx context.Context, session *modelRuntimeSession, deployment mrInventoryDeployment) error {
	client, _, err := session.managed(ctx, deployment.Name)
	if err != nil {
		return err
	}
	models, err := client.Models(ctx)
	if err != nil {
		return err
	}
	for _, card := range models.Data {
		if card.ID != deployment.Name {
			continue
		}
		if len(card.Heads) == 0 || len(deployment.Heads) != len(card.Heads) ||
			strings.Join(deployment.Heads[0].Labels, ",") != strings.Join(card.Heads[0].Labels, ",") {
			return fmt.Errorf("%s heads %+v, the runtime serves %+v", deployment.Name, deployment.Heads, card.Heads)
		}
		return nil
	}
	return fmt.Errorf("no managed runtime serves %s", deployment.Name)
}

// mrManagedProcess is what the lifecycle case reports of one managed process.
type mrManagedProcess struct {
	Models  []string `json:"models"`
	Threads int      `json:"threads"`
}

// checkManagedProcesses requires each logical managed deployment to own one
// ready worker with its fixture's family and surfaces. CPU thread budgets stay
// positive without deriving identity from the current consumer set.
func checkManagedProcesses(ctx context.Context, session *modelRuntimeSession) (map[string]mrManagedProcess, error) {
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
	processes := map[string]mrManagedProcess{}
	owner := map[string]string{}
	for socket, models := range runtimes {
		sort.Strings(models)
		if len(models) != 1 {
			return nil, fmt.Errorf("managed process %s must own one logical deployment, got %v", socket, models)
		}
		for _, model := range models {
			if previous, twice := owner[model]; twice {
				return nil, fmt.Errorf("%s runs in two processes, %s and %s", model, previous, socket)
			}
			if !slices.Contains(mrManagedDeployments, model) {
				return nil, fmt.Errorf("process %s serves undeclared deployment %s", socket, model)
			}
			owner[model] = socket
		}
		threads, err := threadShare(ctx, session, socket)
		if err != nil {
			return nil, err
		}
		processes[filepath.Base(socket)] = mrManagedProcess{Models: models, Threads: threads}
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

// threadShare returns the --threads the Router started the process serving socket with.
func threadShare(ctx context.Context, session *modelRuntimeSession, socket string) (int, error) {
	argv, err := session.pod.RuntimeArgs(ctx, socket)
	if err != nil {
		return 0, err
	}
	index := slices.Index(argv, "--threads")
	if index < 0 || index+1 == len(argv) {
		return 0, fmt.Errorf("process %s runs without a share of the Router's cores: %q", socket, argv)
	}
	threads, err := strconv.Atoi(argv[index+1])
	if err != nil || threads < 1 {
		return 0, fmt.Errorf("process %s runs with --threads %q", socket, argv[index+1])
	}
	return threads, nil
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
