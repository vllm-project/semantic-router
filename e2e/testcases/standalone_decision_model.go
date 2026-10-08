package testcases

import (
	"context"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/yaml"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// standaloneDecisionModel is the profile's Helm decisionModel value, in the
// casing a user may type: the Router matches decision model names in any case.
const standaloneDecisionModel = "vela-1.0"

func init() {
	pkgtestcases.Register("standalone-decision-model", pkgtestcases.TestCase{
		Description: "The Helm decisionModel value reaches global.model_catalog.system.decision_model and the Router serves with it",
		Tags:        []string{"standalone", "config", "kubernetes"},
		Fn:          testStandaloneDecisionModel,
	})
}

func testStandaloneDecisionModel(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	cm, err := client.CoreV1().ConfigMaps(standaloneNamespace).Get(ctx, standaloneConfigMap, metav1.GetOptions{})
	if err != nil {
		return err
	}
	if got := cm.Annotations["semantic-router.vllm.ai/decision-model"]; got != standaloneDecisionModel {
		return fmt.Errorf("the ConfigMap records decision model %q, want the Helm value %q", got, standaloneDecisionModel)
	}
	var document struct {
		Global struct {
			ModelCatalog struct {
				System map[string]any `json:"system"`
			} `json:"model_catalog"`
		} `json:"global"`
	}
	if err = yaml.Unmarshal([]byte(cm.Data["config.yaml"]), &document); err != nil {
		return fmt.Errorf("decode the Router config: %w", err)
	}
	if got := document.Global.ModelCatalog.System["decision_model"]; got != standaloneDecisionModel {
		return fmt.Errorf("global.model_catalog.system.decision_model = %v, want %q", got, standaloneDecisionModel)
	}
	// The Router refuses a decision model it does not know, so a routed reply
	// shows it loaded the config that names this one.
	reply, err := standaloneChat(ctx, client, opts, "Which model answers with the decision model set?")
	if err != nil {
		return err
	}
	return reply.servedBy(standalonePrimaryModel, standaloneDefaultRoute)
}
