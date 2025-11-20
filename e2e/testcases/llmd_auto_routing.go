package testcases

import (
    "context"
    "fmt"
    "strings"
    "time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("llmd-auto-routing", pkgtestcases.TestCase{
		Description: "Auto model selection routes math and cs",
		Tags:        []string{"llmd", "routing"},
		Fn:          llmdAutoRouting,
	})
}

func llmdAutoRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	cases := []struct {
		prompt string
		model  string
	}{
		{prompt: "What is 2+2?", model: "phi4-mini"},
		{prompt: "Explain TCP three-way handshake", model: "llama3-8b"},
	}

    for _, c := range cases {
        res, err := doLLMDChat(ctx, localPort, "auto", c.prompt, 45*time.Second)
        if err != nil {
            return err
        }
        selected := getSelectedModel(res.headers)
        pod := getInferencePod(res.headers)
        if selected == "" && pod != "" {
            if strings.HasPrefix(pod, "phi4-mini-") {
                selected = "phi4-mini"
            } else if strings.HasPrefix(pod, "vllm-llama3-8b-instruct-") {
                selected = "llama3-8b"
            }
        }
        if selected != c.model {
            return fmt.Errorf("prompt '%s' expected model %s got %s", c.prompt, c.model, selected)
        }
        if pod == "" {
            return fmt.Errorf("missing x-inference-pod for prompt '%s'", c.prompt)
        }
    }
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"cases": len(cases)})
	}
	return nil
}
