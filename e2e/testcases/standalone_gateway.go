package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/yaml"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// The standalone profile's Router, its config and the models it serves.
const (
	standaloneNamespace     = "vllm-semantic-router-system"
	standaloneRouter        = "semantic-router"
	standaloneConfigMap     = "semantic-router-config"
	standalonePrimaryModel  = "primary-model"
	standaloneFallbackModel = "fallback-model"
	standaloneDefaultRoute  = "standalone_default"
	standaloneFailoverRoute = "standalone_failover"
	standaloneFailoverQuery = "standalonefailover: which model answers?"
)

func init() {
	pkgtestcases.Register("standalone-chat-completions", pkgtestcases.TestCase{
		Description: "A chat completion through the standalone Router's Service is routed and served with no Envoy",
		Tags:        []string{"standalone", "gateway", "kubernetes"},
		Fn:          testStandaloneChatCompletions,
	})
	pkgtestcases.Register("standalone-models", pkgtestcases.TestCase{
		Description: "GET /v1/models through the standalone Router's Service lists the configured models",
		Tags:        []string{"standalone", "gateway", "kubernetes"},
		Fn:          testStandaloneModels,
	})
	pkgtestcases.Register("standalone-fallback", pkgtestcases.TestCase{
		Description: "A candidate whose backend refuses connections falls back to the next model in the standalone Router",
		Tags:        []string{"standalone", "gateway", "fallback", "kubernetes"},
		Fn:          testStandaloneFallback,
	})
	pkgtestcases.Register("standalone-config-rollout", pkgtestcases.TestCase{
		Description: "A ConfigMap change rolled out to the standalone Router routes new requests through the same Service",
		Tags:        []string{"standalone", "gateway", "config", "kubernetes"},
		Fn:          testStandaloneConfigRollout,
	})
}

type standaloneReply struct {
	status   int
	headers  http.Header
	model    string
	decision string
	echo     map[string]any
}

// standaloneChat sends one chat completion through a fresh port-forward to the
// profile's Service, so a case can call it again after the Pod is replaced.
func standaloneChat(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions, content string) (standaloneReply, error) {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return standaloneReply{}, err
	}
	defer session.Close()
	response, err := fixtures.DoPOSTRequest(ctx, session.HTTPClient(60*time.Second), session.URL("/v1/chat/completions"),
		fixtures.ChatCompletionsRequest{
			Model:    "auto",
			Messages: []fixtures.ChatMessage{{Role: "user", Content: content}},
		})
	if err != nil {
		return standaloneReply{}, err
	}
	reply := standaloneReply{
		status:   response.StatusCode,
		headers:  response.Headers,
		model:    response.Headers.Get("x-vsr-selected-model"),
		decision: response.Headers.Get("x-vsr-selected-decision"),
	}
	if response.StatusCode != http.StatusOK {
		return reply, fmt.Errorf("chat completion status %d: %s", response.StatusCode, truncateString(string(response.Body), 400))
	}
	var payload struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := response.DecodeJSON(&payload); err != nil || len(payload.Choices) == 0 {
		return reply, fmt.Errorf("chat completion body has no choices: %s", truncateString(string(response.Body), 400))
	}
	if err := json.Unmarshal([]byte(payload.Choices[0].Message.Content), &reply.echo); err != nil {
		return reply, fmt.Errorf("the answer is not provider-mocker's echo: %s", truncateString(payload.Choices[0].Message.Content, 400))
	}
	return reply, nil
}

func (r standaloneReply) servedBy(model, decision string) error {
	if r.decision != decision || r.model != model {
		return fmt.Errorf("decision %q model %q, want %q and %q", r.decision, r.model, decision, model)
	}
	if r.echo["mock"] != "provider-mocker" || r.echo["model"] != model {
		return fmt.Errorf("the backend received model %v from %v, want %q from provider-mocker", r.echo["model"], r.echo["mock"], model)
	}
	return nil
}

func testStandaloneChatCompletions(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	reply, err := standaloneChat(ctx, client, opts, "Say hello from the standalone Router.")
	if err != nil {
		return err
	}
	if err := reply.servedBy(standalonePrimaryModel, standaloneDefaultRoute); err != nil {
		return err
	}
	// Standalone mode has no Envoy on the path.
	if server := reply.headers.Get("server"); strings.EqualFold(server, "envoy") {
		return fmt.Errorf("the response came through Envoy (server: %s)", server)
	}
	if reply.headers.Get("x-envoy-upstream-service-time") != "" {
		return fmt.Errorf("the response carries Envoy's x-envoy-upstream-service-time")
	}
	return nil
}

func testStandaloneModels(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	response, err := fixtures.DoGETRequest(ctx, session.HTTPClient(30*time.Second), session.URL("/v1/models"))
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("GET /v1/models status %d: %s", response.StatusCode, truncateString(string(response.Body), 400))
	}
	var list struct {
		Object string `json:"object"`
		Data   []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := response.DecodeJSON(&list); err != nil {
		return fmt.Errorf("decode /v1/models: %w", err)
	}
	ids := make([]string, 0, len(list.Data))
	for _, model := range list.Data {
		ids = append(ids, model.ID)
	}
	for _, want := range []string{standalonePrimaryModel, standaloneFallbackModel} {
		if !slices.Contains(ids, want) {
			return fmt.Errorf("/v1/models lists %v, want %q among them", ids, want)
		}
	}
	return nil
}

func testStandaloneFallback(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	reply, err := standaloneChat(ctx, client, opts, standaloneFailoverQuery)
	if err != nil {
		return err
	}
	if reply.decision != standaloneFailoverRoute {
		return fmt.Errorf("decision %q, want %q", reply.decision, standaloneFailoverRoute)
	}
	if reply.echo["mock"] != "provider-mocker" || reply.echo["model"] != standaloneFallbackModel {
		return fmt.Errorf("the answer came from model %v, want the fallback %q", reply.echo["model"], standaloneFallbackModel)
	}
	if reply.headers.Get("x-vsr-fallback-attempts") == "" {
		return fmt.Errorf("the response does not report the fallback attempts")
	}
	return nil
}

// testStandaloneConfigRollout follows the chart's documented config update: a
// changed ConfigMap reaches the Router with the rollout it triggers, and the
// replacement Pod serves the Service once its /ready probe passes.
func testStandaloneConfigRollout(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	before, err := standaloneChat(ctx, client, opts, "Which model answers before the change?")
	if err != nil {
		return err
	}
	if err = before.servedBy(standalonePrimaryModel, standaloneDefaultRoute); err != nil {
		return fmt.Errorf("before the change: %w", err)
	}

	configMaps := client.CoreV1().ConfigMaps(standaloneNamespace)
	cm, err := configMaps.Get(ctx, standaloneConfigMap, metav1.GetOptions{})
	if err != nil {
		return err
	}
	var document map[string]any
	if err = yaml.Unmarshal([]byte(cm.Data["config.yaml"]), &document); err != nil {
		return fmt.Errorf("decode the Router config: %w", err)
	}
	if err = setDecisionModel(document, standaloneDefaultRoute, standaloneFallbackModel); err != nil {
		return err
	}
	changed, err := yaml.Marshal(document)
	if err != nil {
		return err
	}
	cm.Data["config.yaml"] = string(changed)
	if _, err = configMaps.Update(ctx, cm, metav1.UpdateOptions{}); err != nil {
		return fmt.Errorf("update the Router ConfigMap: %w", err)
	}

	deployments := client.AppsV1().Deployments(standaloneNamespace)
	restart := fmt.Sprintf(`{"spec":{"template":{"metadata":{"annotations":{"kubectl.kubernetes.io/restartedAt":%q}}}}}`,
		time.Now().UTC().Format(time.RFC3339))
	if _, err = deployments.Patch(ctx, standaloneRouter, types.StrategicMergePatchType, []byte(restart), metav1.PatchOptions{}); err != nil {
		return fmt.Errorf("roll out the Router: %w", err)
	}
	if err = wait.PollUntilContextTimeout(ctx, 5*time.Second, 15*time.Minute, true, func(ctx context.Context) (bool, error) {
		deployment, getErr := deployments.Get(ctx, standaloneRouter, metav1.GetOptions{})
		if getErr != nil {
			return false, nil
		}
		return rolloutComplete(deployment), nil
	}); err != nil {
		return fmt.Errorf("wait for the Router rollout: %w", err)
	}

	after, err := standaloneChat(ctx, client, opts, "Which model answers after the change?")
	if err != nil {
		return fmt.Errorf("after the rollout: %w", err)
	}
	if err := after.servedBy(standaloneFallbackModel, standaloneDefaultRoute); err != nil {
		return fmt.Errorf("after the rollout: %w", err)
	}
	return nil
}

func setDecisionModel(document map[string]any, decision, model string) error {
	routing, _ := document["routing"].(map[string]any)
	decisions, _ := routing["decisions"].([]any)
	for _, item := range decisions {
		entry, _ := item.(map[string]any)
		if entry["name"] != decision {
			continue
		}
		entry["modelRefs"] = []any{map[string]any{"model": model, "use_reasoning": false}}
		return nil
	}
	return fmt.Errorf("the Router config has no decision %q", decision)
}

func rolloutComplete(deployment *appsv1.Deployment) bool {
	replicas := int32(1)
	if deployment.Spec.Replicas != nil {
		replicas = *deployment.Spec.Replicas
	}
	status := deployment.Status
	return status.ObservedGeneration >= deployment.Generation &&
		status.UpdatedReplicas == replicas &&
		status.Replicas == replicas &&
		status.AvailableReplicas == replicas
}
