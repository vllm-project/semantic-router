package testcases

import (
	"context"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/yaml"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// standaloneDecisionModel names the deployment selected by the Helm value.
const standaloneDecisionModel = "primary"

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
				System struct {
					DecisionModel struct {
						Deployment string `json:"deployment"`
					} `json:"decision_model"`
				} `json:"system"`
				Deployments map[string]struct {
					Provider string `json:"provider"`
					Artifact string `json:"artifact"`
				} `json:"deployments"`
			} `json:"model_catalog"`
		} `json:"global"`
	}
	if err = yaml.Unmarshal([]byte(cm.Data["config.yaml"]), &document); err != nil {
		return fmt.Errorf("decode the Router config: %w", err)
	}
	if got := document.Global.ModelCatalog.System.DecisionModel.Deployment; got != standaloneDecisionModel {
		return fmt.Errorf("global.model_catalog.system.decision_model = %v, want %q", got, standaloneDecisionModel)
	}
	deployment, exists := document.Global.ModelCatalog.Deployments[standaloneDecisionModel]
	if !exists || deployment.Provider != "model_runtime" || deployment.Artifact != "vllm-sr/Vela-2.0-0.3B" {
		return fmt.Errorf("the selected judgment deployment was not preserved: %+v", deployment)
	}
	// The Router refuses a decision model it does not know, so a routed reply
	// shows it loaded the config that names this one.
	reply, err := standaloneChat(ctx, client, opts, "Which model answers with the decision model set?")
	if err != nil {
		return err
	}
	return reply.servedBy(standalonePrimaryModel, standaloneDefaultRoute)
}
