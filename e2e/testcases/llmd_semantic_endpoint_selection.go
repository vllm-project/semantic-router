package testcases

import (
	"context"
	"fmt"
	"net/http"
	"sort"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const llmdEndpointSelectionAttempts = 3

type llmdEndpointSelectionCase struct {
	name         string
	prompt       string
	wantDecision string
	wantModel    string
	backendLabel string
}

func init() {
	pkgtestcases.Register("llm-d-semantic-endpoint-selection", pkgtestcases.TestCase{
		Description: "Verify Semantic Router selects a logical model pool and llm-d selects a ready replica inside that pool",
		Tags:        []string{"llm-d", "gateway", "inference", "routing"},
		Fn:          testLLMDSemanticEndpointSelection,
	})
}

func testLLMDSemanticEndpointSelection(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	cases := []llmdEndpointSelectionCase{
		{
			name:         "math",
			prompt:       "Solve the algebra equation 3x + 7 = 22 and explain each mathematical step.",
			wantDecision: "math_decision",
			wantModel:    "phi4-mini",
			backendLabel: "phi4-mini",
		},
		{
			name:         "computer-science",
			prompt:       "Explain how a hash table resolves collisions in a computer program.",
			wantDecision: "computer_science_decision",
			wantModel:    "llama3-8b",
			backendLabel: "vllm-llama3-8b-instruct",
		},
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	details := make([]map[string]interface{}, 0, len(cases))
	for _, testCase := range cases {
		caseDetails, err := runLLMDEndpointSelectionCase(ctx, client, localPort, testCase)
		if err != nil {
			return fmt.Errorf("%s route: %w", testCase.name, err)
		}
		details = append(details, caseDetails)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"attempts_per_pool": llmdEndpointSelectionAttempts,
			"pools":             details,
		})
	}
	return nil
}

func runLLMDEndpointSelectionCase(
	ctx context.Context,
	client *kubernetes.Clientset,
	localPort string,
	testCase llmdEndpointSelectionCase,
) (map[string]interface{}, error) {
	readyPods, err := readyPodNames(ctx, client, "default", "app="+testCase.backendLabel)
	if err != nil {
		return nil, fmt.Errorf("list ready backend pods: %w", err)
	}
	if len(readyPods) < 2 {
		return nil, fmt.Errorf("need at least two ready pods for endpoint selection, found %d", len(readyPods))
	}

	selectedPods := make([]string, 0, llmdEndpointSelectionAttempts)
	for attempt := 0; attempt < llmdEndpointSelectionAttempts; attempt++ {
		response, err := sendLocalChatCompletion(ctx, localPort, "vllm-sr/auto", testCase.prompt, 30*time.Second)
		if err != nil {
			return nil, fmt.Errorf("attempt %d request: %w", attempt+1, err)
		}
		if response.StatusCode != http.StatusOK {
			return nil, fmt.Errorf(
				"attempt %d returned status %d: %s",
				attempt+1,
				response.StatusCode,
				string(response.Body),
			)
		}

		decision := response.Headers.Get("x-vsr-selected-decision")
		if decision != testCase.wantDecision {
			return nil, fmt.Errorf("attempt %d selected decision %q, want %q", attempt+1, decision, testCase.wantDecision)
		}
		model := response.Headers.Get("x-vsr-selected-model")
		if model != testCase.wantModel {
			return nil, fmt.Errorf("attempt %d selected model %q, want %q", attempt+1, model, testCase.wantModel)
		}

		pod := response.Headers.Get("x-inference-pod")
		if pod == "" {
			return nil, fmt.Errorf("attempt %d response omitted x-inference-pod", attempt+1)
		}
		if _, ok := readyPods[pod]; !ok {
			return nil, fmt.Errorf("attempt %d selected pod %q outside ready %s pool", attempt+1, pod, testCase.backendLabel)
		}
		selectedPods = append(selectedPods, pod)
	}

	readyPodList := make([]string, 0, len(readyPods))
	for pod := range readyPods {
		readyPodList = append(readyPodList, pod)
	}
	sort.Strings(readyPodList)

	return map[string]interface{}{
		"decision":         testCase.wantDecision,
		"model_pool":       testCase.wantModel,
		"ready_pods":       readyPodList,
		"selected_pods":    selectedPods,
		"request_attempts": llmdEndpointSelectionAttempts,
	}, nil
}

func readyPodNames(
	ctx context.Context,
	client *kubernetes.Clientset,
	namespace string,
	labelSelector string,
) (map[string]struct{}, error) {
	pods, err := client.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{LabelSelector: labelSelector})
	if err != nil {
		return nil, err
	}

	ready := make(map[string]struct{}, len(pods.Items))
	for _, pod := range pods.Items {
		for _, condition := range pod.Status.Conditions {
			if condition.Type == "Ready" && condition.Status == "True" {
				ready[pod.Name] = struct{}{}
				break
			}
		}
	}
	return ready, nil
}
