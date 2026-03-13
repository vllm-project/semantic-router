package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("apiserver-runtime-config-endpoints", pkgtestcases.TestCase{
		Description: "Verify runtime-config-backed apiserver endpoints expose the active model list and classifier config",
		Tags:        []string{"apiserver", "config", "api"},
		Fn:          testAPIServerRuntimeConfigEndpoints,
	})
}

type openAIModelsResponse struct {
	Object string `json:"object"`
	Data   []struct {
		ID string `json:"id"`
	} `json:"data"`
}

type classifierInfoResponse struct {
	Status string `json:"status"`
	Config struct {
		Decisions []struct {
			Name string `json:"Name"`
		} `json:"Decisions"`
	} `json:"config"`
}

type modelsInfoResponse struct {
	Models []struct {
		Name      string `json:"name"`
		Type      string `json:"type"`
		Loaded    bool   `json:"loaded"`
		State     string `json:"state"`
		ModelPath string `json:"model_path"`
		Registry  struct {
			LocalPath string `json:"local_path"`
			RepoID    string `json:"repo_id"`
		} `json:"registry"`
	} `json:"models"`
	Summary struct {
		Ready        bool   `json:"ready"`
		Phase        string `json:"phase"`
		LoadedModels int    `json:"loaded_models"`
		TotalModels  int    `json:"total_models"`
	} `json:"summary"`
}

func testAPIServerRuntimeConfigEndpoints(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	httpClient := session.HTTPClient(30 * time.Second)
	modelIDs, err := fetchModelIDs(ctx, httpClient, session.URL("/v1/models"))
	if err != nil {
		return err
	}
	decisionNames, err := fetchClassifierDecisions(ctx, httpClient, session.URL("/info/classifier"))
	if err != nil {
		return err
	}
	modelNames, loadedModels, totalModels, err := fetchModelsInfo(ctx, httpClient, session.URL("/info/models"))
	if err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"model_ids":       modelIDs,
			"decision_count":  len(decisionNames),
			"decision_sample": decisionNames,
			"router_models":   modelNames,
			"loaded_models":   loadedModels,
			"total_models":    totalModels,
		})
	}

	return nil
}

func fetchModelIDs(ctx context.Context, httpClient *http.Client, url string) ([]string, error) {
	modelsResp, err := getJSON(ctx, httpClient, url)
	if err != nil {
		return nil, err
	}
	if modelsResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("expected /v1/models status 200, got %d: %s", modelsResp.StatusCode, string(modelsResp.Body))
	}

	var models openAIModelsResponse
	if err := json.Unmarshal(modelsResp.Body, &models); err != nil {
		return nil, fmt.Errorf("decode /v1/models response: %w", err)
	}
	if models.Object != "list" {
		return nil, fmt.Errorf("expected /v1/models object=list, got %q", models.Object)
	}

	modelIDs := make([]string, 0, len(models.Data))
	for _, model := range models.Data {
		modelIDs = append(modelIDs, model.ID)
	}
	if !containsString(modelIDs, "MoM") {
		return nil, fmt.Errorf("expected /v1/models to include MoM, got %v", modelIDs)
	}
	return modelIDs, nil
}

func fetchClassifierDecisions(
	ctx context.Context,
	httpClient *http.Client,
	url string,
) ([]string, error) {
	classifierResp, err := getJSON(ctx, httpClient, url)
	if err != nil {
		return nil, err
	}
	if classifierResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("expected /info/classifier status 200, got %d: %s", classifierResp.StatusCode, string(classifierResp.Body))
	}

	var classifierInfo classifierInfoResponse
	if err := json.Unmarshal(classifierResp.Body, &classifierInfo); err != nil {
		return nil, fmt.Errorf("decode /info/classifier response: %w", err)
	}
	if classifierInfo.Status != "config_loaded" {
		return nil, fmt.Errorf("expected classifier config to be loaded, got %q", classifierInfo.Status)
	}

	decisionNames := make([]string, 0, len(classifierInfo.Config.Decisions))
	for _, decision := range classifierInfo.Config.Decisions {
		decisionNames = append(decisionNames, decision.Name)
	}
	if !containsString(decisionNames, "health_decision") {
		return nil, fmt.Errorf("expected classifier config to include health_decision, got %v", decisionNames)
	}
	return decisionNames, nil
}

func fetchModelsInfo(
	ctx context.Context,
	httpClient *http.Client,
	url string,
) ([]string, int, int, error) {
	modelsResp, err := getJSON(ctx, httpClient, url)
	if err != nil {
		return nil, 0, 0, err
	}
	if modelsResp.StatusCode != http.StatusOK {
		return nil, 0, 0, fmt.Errorf("expected /info/models status 200, got %d: %s", modelsResp.StatusCode, string(modelsResp.Body))
	}

	var modelsInfo modelsInfoResponse
	if err := json.Unmarshal(modelsResp.Body, &modelsInfo); err != nil {
		return nil, 0, 0, fmt.Errorf("decode /info/models response: %w", err)
	}
	if len(modelsInfo.Models) == 0 {
		return nil, 0, 0, fmt.Errorf("expected /info/models to include at least one router model")
	}
	if modelsInfo.Summary.TotalModels < len(modelsInfo.Models) {
		return nil, 0, 0, fmt.Errorf("expected summary total_models >= model entries, got %+v", modelsInfo.Summary)
	}

	modelNames := make([]string, 0, len(modelsInfo.Models))
	for _, model := range modelsInfo.Models {
		if model.Name == "" || model.Type == "" || model.State == "" {
			return nil, 0, 0, fmt.Errorf("expected router model entries to include name/type/state, got %+v", model)
		}
		modelNames = append(modelNames, model.Name)
	}

	return modelNames, modelsInfo.Summary.LoadedModels, modelsInfo.Summary.TotalModels, nil
}

type httpResponse struct {
	StatusCode int
	Body       []byte
}

func getJSON(ctx context.Context, httpClient *http.Client, url string) (*httpResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("create GET request %s: %w", url, err)
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send GET request %s: %w", url, err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read GET response %s: %w", url, err)
	}

	return &httpResponse{
		StatusCode: resp.StatusCode,
		Body:       body,
	}, nil
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
