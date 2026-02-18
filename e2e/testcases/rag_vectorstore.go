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

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("rag-vectorstore", pkgtestcases.TestCase{
		Description: "Test local Vector Store RAG: create store, upload file, attach, ingest, search",
		Tags:        []string{"rag", "vectorstore", "ingestion", "search"},
		Fn:          RAGVectorStoreTestCase,
	})
}

// RAGVectorStoreTestCase exercises the full vector store ingestion pipeline:
// 1. Create vector store
// 2. Upload file
// 3. Attach file to vector store (triggers async ingestion)
// 4. Poll until ingestion completes
// 5. Search and verify results
func RAGVectorStoreTestCase(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup service connection: %w", err)
	}
	defer stopPortForward()

	baseURL := fmt.Sprintf("http://localhost:%s", localPort)
	httpClient := &http.Client{Timeout: 30 * time.Second}

	if opts.Verbose {
		fmt.Println("[RAG VectorStore] Starting local vector store E2E test")
	}

	// Step 1: Create vector store
	vectorStoreID, err := createVectorStore(ctx, httpClient, baseURL, opts)
	if err != nil {
		return fmt.Errorf("step 1 (create vector store) failed: %w", err)
	}

	// Step 2: Upload file
	fileID, err := uploadTestFile(ctx, httpClient, baseURL, opts)
	if err != nil {
		return fmt.Errorf("step 2 (upload file) failed: %w", err)
	}

	// Step 3: Attach file to vector store
	err = attachFile(ctx, httpClient, baseURL, vectorStoreID, fileID, opts)
	if err != nil {
		return fmt.Errorf("step 3 (attach file) failed: %w", err)
	}

	// Step 4: Poll until ingestion completes
	err = waitForIngestion(ctx, httpClient, baseURL, vectorStoreID, opts)
	if err != nil {
		return fmt.Errorf("step 4 (wait for ingestion) failed: %w", err)
	}

	// Step 5: Search and verify
	err = searchAndVerify(ctx, httpClient, baseURL, vectorStoreID, opts)
	if err != nil {
		return fmt.Errorf("step 5 (search) failed: %w", err)
	}

	// Cleanup: delete vector store
	deleteVectorStore(ctx, httpClient, baseURL, vectorStoreID)

	if opts.Verbose {
		fmt.Println("[RAG VectorStore] All steps passed")
	}

	return nil
}

func createVectorStore(ctx context.Context, client *http.Client, baseURL string, opts pkgtestcases.TestCaseOptions) (string, error) {
	body, _ := json.Marshal(map[string]interface{}{
		"name": "e2e-test-store",
	})

	req, err := http.NewRequestWithContext(ctx, "POST", baseURL+"/v1/vector_stores", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("unexpected status %d: %s", resp.StatusCode, string(respBody))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}

	id, ok := result["id"].(string)
	if !ok || id == "" {
		return "", fmt.Errorf("no id in response: %v", result)
	}

	if opts.Verbose {
		fmt.Printf("[RAG VectorStore] Created vector store: %s\n", id)
	}
	return id, nil
}

func uploadTestFile(ctx context.Context, client *http.Client, baseURL string, opts pkgtestcases.TestCaseOptions) (string, error) {
	// Create multipart form with a test document.
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	fw, err := w.CreateFormFile("file", "test-document.txt")
	if err != nil {
		return "", err
	}

	testContent := `Company PTO Policy

All full-time employees receive 20 days of paid time off per year.
PTO accrues at a rate of 1.67 days per month.

Unused PTO can be carried over up to a maximum of 5 days.
PTO requests must be submitted at least 2 weeks in advance.

For questions about PTO, contact HR at hr@company.example.com.`

	if _, err := fw.Write([]byte(testContent)); err != nil {
		return "", err
	}

	if err := w.WriteField("purpose", "assistants"); err != nil {
		return "", err
	}
	w.Close()

	req, err := http.NewRequestWithContext(ctx, "POST", baseURL+"/v1/files", &buf)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", w.FormDataContentType())

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("unexpected status %d: %s", resp.StatusCode, string(respBody))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}

	id, ok := result["id"].(string)
	if !ok || id == "" {
		return "", fmt.Errorf("no id in response: %v", result)
	}

	if opts.Verbose {
		fmt.Printf("[RAG VectorStore] Uploaded file: %s\n", id)
	}
	return id, nil
}

func attachFile(ctx context.Context, client *http.Client, baseURL, vectorStoreID, fileID string, opts pkgtestcases.TestCaseOptions) error {
	body, _ := json.Marshal(map[string]interface{}{
		"file_id": fileID,
	})

	url := fmt.Sprintf("%s/v1/vector_stores/%s/files", baseURL, vectorStoreID)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("unexpected status %d: %s", resp.StatusCode, string(respBody))
	}

	if opts.Verbose {
		fmt.Printf("[RAG VectorStore] Attached file %s to store %s\n", fileID, vectorStoreID)
	}
	return nil
}

func waitForIngestion(ctx context.Context, client *http.Client, baseURL, vectorStoreID string, opts pkgtestcases.TestCaseOptions) error {
	url := fmt.Sprintf("%s/v1/vector_stores/%s/files", baseURL, vectorStoreID)
	deadline := time.Now().Add(60 * time.Second)

	for time.Now().Before(deadline) {
		req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
		if err != nil {
			return err
		}

		resp, err := client.Do(req)
		if err != nil {
			return err
		}

		var result map[string]interface{}
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			resp.Body.Close()
			return err
		}
		resp.Body.Close()

		data, ok := result["data"].([]interface{})
		if !ok || len(data) == 0 {
			time.Sleep(500 * time.Millisecond)
			continue
		}

		allDone := true
		for _, item := range data {
			file, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			status, _ := file["status"].(string)
			if status == "failed" {
				lastErr, _ := file["last_error"].(map[string]interface{})
				return fmt.Errorf("ingestion failed: %v", lastErr)
			}
			if status != "completed" {
				allDone = false
			}
		}

		if allDone {
			if opts.Verbose {
				fmt.Println("[RAG VectorStore] Ingestion completed")
			}
			return nil
		}

		time.Sleep(500 * time.Millisecond)
	}

	return fmt.Errorf("ingestion did not complete within timeout")
}

func searchAndVerify(ctx context.Context, client *http.Client, baseURL, vectorStoreID string, opts pkgtestcases.TestCaseOptions) error {
	body, _ := json.Marshal(map[string]interface{}{
		"query":           "What is the PTO policy?",
		"max_num_results": 3,
	})

	url := fmt.Sprintf("%s/v1/vector_stores/%s/search", baseURL, vectorStoreID)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("unexpected status %d: %s", resp.StatusCode, string(respBody))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return err
	}

	data, ok := result["data"].([]interface{})
	if !ok || len(data) == 0 {
		return fmt.Errorf("search returned no results")
	}

	// Verify at least one result contains PTO-related content.
	foundRelevant := false
	for _, item := range data {
		sr, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		content, _ := sr["content"].(string)
		if strings.Contains(strings.ToLower(content), "pto") || strings.Contains(strings.ToLower(content), "paid time off") {
			foundRelevant = true
			break
		}
	}

	if !foundRelevant {
		return fmt.Errorf("search results do not contain expected PTO content")
	}

	if opts.Verbose {
		fmt.Printf("[RAG VectorStore] Search returned %d results with relevant content\n", len(data))
	}
	return nil
}

func deleteVectorStore(ctx context.Context, client *http.Client, baseURL, vectorStoreID string) {
	url := fmt.Sprintf("%s/v1/vector_stores/%s", baseURL, vectorStoreID)
	req, _ := http.NewRequestWithContext(ctx, "DELETE", url, nil)
	resp, err := client.Do(req)
	if err == nil {
		resp.Body.Close()
	}
}
