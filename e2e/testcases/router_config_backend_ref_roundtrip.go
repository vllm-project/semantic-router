package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("router-config-backend-ref-roundtrip", pkgtestcases.TestCase{
		Description: "Verify backend reference names stabilize after canonicalization across repeated router config replacements",
		Tags:        []string{"kubernetes", "apiserver", "config", "regression"},
		Fn:          testRouterConfigBackendRefRoundTrip,
	})
}

type backendRefNameDocument struct {
	Providers struct {
		Models []struct {
			Name        string `json:"name"`
			BackendRefs []struct {
				Name string `json:"name"`
			} `json:"backend_refs"`
		} `json:"models"`
	} `json:"providers"`
}

func testRouterConfigBackendRefRoundTrip(
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
	configURL := session.URL("/config/router")
	body, sourceNames, err := fetchBackendRefNames(ctx, httpClient, configURL)
	if err != nil {
		return err
	}

	var canonicalNames map[string][]string
	for cycle := 1; cycle <= 3; cycle++ {
		updateBody, err := json.Marshal(struct {
			YAML string `json:"yaml"`
		}{YAML: string(body)})
		if err != nil {
			return fmt.Errorf("cycle %d: encode router config update: %w", cycle, err)
		}

		resp, err := postJSON(ctx, httpClient, http.MethodPut, configURL, updateBody)
		if err != nil {
			return err
		}
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("cycle %d: expected PUT /config/router status 200, got %d: %s", cycle, resp.StatusCode, string(resp.Body))
		}

		var persistedNames map[string][]string
		body, persistedNames, err = fetchBackendRefNames(ctx, httpClient, configURL)
		if err != nil {
			return fmt.Errorf("cycle %d: %w", cycle, err)
		}
		if cycle == 1 {
			canonicalNames = persistedNames
			continue
		}
		if !reflect.DeepEqual(persistedNames, canonicalNames) {
			return fmt.Errorf("cycle %d: canonical backend reference names changed: canonical=%v persisted=%v", cycle, canonicalNames, persistedNames)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"replacements":        3,
			"source_names":        sourceNames,
			"canonicalized_names": canonicalNames,
		})
	}

	return nil
}

func fetchBackendRefNames(
	ctx context.Context,
	httpClient *http.Client,
	url string,
) ([]byte, map[string][]string, error) {
	resp, err := getJSON(ctx, httpClient, url)
	if err != nil {
		return nil, nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, nil, fmt.Errorf("expected GET /config/router status 200, got %d: %s", resp.StatusCode, string(resp.Body))
	}

	names, err := decodeBackendRefNames(resp.Body)
	if err != nil {
		return nil, nil, fmt.Errorf("decode GET /config/router response: %w", err)
	}
	return resp.Body, names, nil
}

func decodeBackendRefNames(body []byte) (map[string][]string, error) {
	var doc backendRefNameDocument
	if err := json.Unmarshal(body, &doc); err != nil {
		return nil, err
	}

	names := make(map[string][]string)
	for _, model := range doc.Providers.Models {
		if model.Name == "" || len(model.BackendRefs) == 0 {
			continue
		}
		for _, backendRef := range model.BackendRefs {
			if backendRef.Name == "" {
				return nil, fmt.Errorf("model %q has an empty backend reference name", model.Name)
			}
			names[model.Name] = append(names[model.Name], backendRef.Name)
		}
	}
	if len(names) == 0 {
		return nil, fmt.Errorf("expected /config/router to include named backend references")
	}
	return names, nil
}
