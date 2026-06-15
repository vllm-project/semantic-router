package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("vectorstore-registry-restart-recovery", pkgtestcases.TestCase{
		Description: "Vector store and file metadata stored in Postgres survive a semantic-router pod restart",
		Tags:        []string{"vectorstore", "registry", "functional", "postgres", "restart"},
		Fn:          testVectorStoreRegistryRestartRecovery,
	})
}

func testVectorStoreRegistryRestartRecovery(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing VectorStore Registry: restart recovery (Postgres persistence)")
	}

	storeID, fileID, err := createRegistryEntriesBeforeRestart(ctx, client, opts)
	if err != nil {
		return err
	}

	if err := deleteSemanticRouterPod(ctx, client, opts); err != nil {
		return err
	}

	if err := waitForSemanticRouterReady(ctx, client, opts); err != nil {
		return err
	}

	return verifyRegistryEntriesAfterRestart(ctx, client, opts, storeID, fileID)
}

// createRegistryEntriesBeforeRestart creates a vector store and uploads a file,
// then verifies both are accessible and persisted in Postgres.
func createRegistryEntriesBeforeRestart(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) (string, string, error) {
	session, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return "", "", fmt.Errorf("open session for pre-restart setup: %w", err)
	}
	defer session.Close()

	httpClient := session.HTTPClient(30 * time.Second)
	baseURL := session.BaseURL()

	storeID, err := registryCreateVectorStore(ctx, httpClient, baseURL, opts.Verbose)
	if err != nil {
		return "", "", err
	}

	fileID, err := registryUploadFile(ctx, httpClient, baseURL, opts.Verbose)
	if err != nil {
		return "", "", err
	}

	if err := assertRegistryPostgresRows(ctx, client, storeID, fileID, opts); err != nil {
		return "", "", fmt.Errorf("registry entries not confirmed in Postgres before restart: %w", err)
	}

	return storeID, fileID, nil
}

func registryCreateVectorStore(ctx context.Context, httpClient *http.Client, baseURL string, verbose bool) (string, error) {
	payload := map[string]interface{}{
		"name":     "e2e-registry-durability-test",
		"metadata": map[string]interface{}{"env": "e2e", "purpose": "restart-recovery"},
	}

	resp, err := fixtures.DoPOSTRequest(ctx, httpClient, baseURL+"/v1/vector_stores", payload)
	if err != nil {
		return "", fmt.Errorf("POST /v1/vector_stores failed: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("POST /v1/vector_stores returned status %d: %s", resp.StatusCode, string(resp.Body))
	}

	var result struct {
		ID   string `json:"id"`
		Name string `json:"name"`
	}
	if err := resp.DecodeJSON(&result); err != nil {
		return "", fmt.Errorf("decode create vector store response: %w", err)
	}
	if result.ID == "" {
		return "", fmt.Errorf("vector store response has empty ID")
	}

	if verbose {
		fmt.Printf("[Test] Created vector store: %s (name=%s)\n", result.ID, result.Name)
	}
	return result.ID, nil
}

func registryUploadFile(ctx context.Context, httpClient *http.Client, baseURL string, verbose bool) (string, error) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	fw, err := w.CreateFormFile("file", "registry-test-document.txt")
	if err != nil {
		return "", fmt.Errorf("create form file: %w", err)
	}
	if _, err := fw.Write([]byte("Registry durability test document.\nThis content exists to verify file metadata survives restart.")); err != nil {
		return "", fmt.Errorf("write form file: %w", err)
	}
	if err := w.WriteField("purpose", "assistants"); err != nil {
		return "", fmt.Errorf("write purpose field: %w", err)
	}
	if err := w.Close(); err != nil {
		return "", fmt.Errorf("close multipart writer: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", baseURL+"/v1/files", &buf)
	if err != nil {
		return "", fmt.Errorf("create upload request: %w", err)
	}
	req.Header.Set("Content-Type", w.FormDataContentType())

	resp, err := httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("POST /v1/files failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read upload response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("POST /v1/files returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var result struct {
		ID       string `json:"id"`
		Filename string `json:"filename"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("decode upload response: %w", err)
	}
	if result.ID == "" {
		return "", fmt.Errorf("file response has empty ID")
	}

	if verbose {
		fmt.Printf("[Test] Uploaded file: %s (filename=%s)\n", result.ID, result.Filename)
	}
	return result.ID, nil
}

func assertRegistryPostgresRows(ctx context.Context, client *kubernetes.Clientset, storeID, fileID string, opts pkgtestcases.TestCaseOptions) error {
	podName, found, err := getPostgresPod(ctx, client)
	if err != nil {
		return err
	}
	if !found {
		if opts.Verbose {
			fmt.Println("[Test] No postgres pod found — skipping direct DB verification")
		}
		return nil
	}

	for _, check := range []struct {
		table string
		id    string
		label string
	}{
		{"vector_store_registry", storeID, "vector store"},
		{"file_registry", fileID, "file"},
	} {
		query := fmt.Sprintf("SELECT COUNT(*) FROM %s WHERE id = '%s'", check.table, check.id)
		output, err := execPsql(ctx, podName, opts.Verbose, query)
		if err != nil {
			return fmt.Errorf("psql query for %s failed: %w", check.label, err)
		}
		if strings.TrimSpace(output) == "0" {
			return fmt.Errorf("%s %s not found in Postgres table %s", check.label, check.id, check.table)
		}
		if opts.Verbose {
			fmt.Printf("[Test] %s %s confirmed in Postgres (%s)\n", check.label, check.id, check.table)
		}
	}
	return nil
}

// verifyRegistryEntriesAfterRestart polls the vector store and file APIs until
// both entries are accessible after the pod restart.
func verifyRegistryEntriesAfterRestart(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions, storeID, fileID string) error {
	const verifyTimeout = 90 * time.Second
	deadline := time.Now().Add(verifyTimeout)
	var lastErr error

	for time.Now().Before(deadline) {
		err := verifyRegistryOnce(ctx, client, opts, storeID, fileID)
		if err == nil {
			if opts.SetDetails != nil {
				opts.SetDetails(map[string]interface{}{
					"store_id": storeID,
					"file_id":  fileID,
					"survived": true,
				})
			}
			return nil
		}
		lastErr = err
		time.Sleep(3 * time.Second)
	}

	return fmt.Errorf("registry entries not retrievable after %s: %w", verifyTimeout, lastErr)
}

func verifyRegistryOnce(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions, storeID, fileID string) error {
	session, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	httpClient := session.HTTPClient(30 * time.Second)
	baseURL := session.BaseURL()

	if err := verifyStoreExists(ctx, httpClient, baseURL, storeID, opts.Verbose); err != nil {
		return err
	}
	return verifyFileExists(ctx, httpClient, baseURL, fileID, opts.Verbose)
}

func verifyStoreExists(ctx context.Context, httpClient *http.Client, baseURL, storeID string, verbose bool) error {
	resp, err := fixtures.DoGETRequest(ctx, httpClient, baseURL+"/v1/vector_stores/"+storeID)
	if err != nil {
		if verbose {
			fmt.Printf("[Test] GET vector store %s not ready yet: %v — retrying\n", storeID, err)
		}
		return err
	}
	if resp.StatusCode != http.StatusOK {
		retryErr := fmt.Errorf("GET /v1/vector_stores/%s returned %d: %s", storeID, resp.StatusCode, string(resp.Body))
		if verbose {
			fmt.Printf("[Test] %v — retrying\n", retryErr)
		}
		return retryErr
	}

	var result struct {
		ID   string `json:"id"`
		Name string `json:"name"`
	}
	if err := resp.DecodeJSON(&result); err != nil {
		return fmt.Errorf("decode vector store response: %w", err)
	}
	if result.ID != storeID {
		return fmt.Errorf("vector store ID mismatch: got %s, expected %s", result.ID, storeID)
	}

	if verbose {
		fmt.Printf("[Test] Vector store %s survived restart (name=%s)\n", storeID, result.Name)
	}
	return nil
}

func verifyFileExists(ctx context.Context, httpClient *http.Client, baseURL, fileID string, verbose bool) error {
	resp, err := fixtures.DoGETRequest(ctx, httpClient, baseURL+"/v1/files/"+fileID)
	if err != nil {
		if verbose {
			fmt.Printf("[Test] GET file %s not ready yet: %v — retrying\n", fileID, err)
		}
		return err
	}
	if resp.StatusCode != http.StatusOK {
		retryErr := fmt.Errorf("GET /v1/files/%s returned %d: %s", fileID, resp.StatusCode, string(resp.Body))
		if verbose {
			fmt.Printf("[Test] %v — retrying\n", retryErr)
		}
		return retryErr
	}

	var result struct {
		ID       string `json:"id"`
		Filename string `json:"filename"`
	}
	if err := resp.DecodeJSON(&result); err != nil {
		return fmt.Errorf("decode file response: %w", err)
	}
	if result.ID != fileID {
		return fmt.Errorf("file ID mismatch: got %s, expected %s", result.ID, fileID)
	}

	if verbose {
		fmt.Printf("[Test] File %s survived restart (filename=%s)\n", fileID, result.Filename)
	}
	return nil
}
