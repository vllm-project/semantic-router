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
	pkgtestcases.Register("rag-vectorstore-hybrid", pkgtestcases.TestCase{
		Description: "Test multiple vector stores with hybrid search (BM25 + n-gram + vector)",
		Tags:        []string{"rag", "vectorstore", "hybrid", "multi-store", "search"},
		Fn:          RAGVectorStoreHybridTestCase,
	})
}

// RAGVectorStoreHybridTestCase exercises multiple vector stores and hybrid search:
//  1. Create two vector stores (policy-store, tech-store)
//  2. Upload and ingest different documents into each store
//  3. Vector-only search on store 1 (baseline)
//  4. Hybrid search (weighted mode) on store 1
//  5. Hybrid search (RRF mode) on store 2
//  6. Verify stores are isolated (cross-store search returns no results)
//  7. Cleanup
func RAGVectorStoreHybridTestCase(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to setup service connection: %w", err)
	}
	defer stopPortForward()

	baseURL := fmt.Sprintf("http://localhost:%s", localPort)
	httpClient := &http.Client{Timeout: 180 * time.Second}

	logVerbose := func(format string, args ...interface{}) {
		if opts.Verbose {
			fmt.Printf("[RAG Hybrid] "+format+"\n", args...)
		}
	}

	logVerbose("Starting multi-store hybrid search E2E test")

	// --- Step 1: Create two stores ---

	logVerbose("Step 1/7: Creating two vector stores")
	policyStoreID, err := hybridCreateStore(ctx, httpClient, baseURL, "e2e-policy-store")
	if err != nil {
		return fmt.Errorf("step 1 (create policy store) failed: %w", err)
	}
	logVerbose("  Created policy store: %s", policyStoreID)

	techStoreID, err := hybridCreateStore(ctx, httpClient, baseURL, "e2e-tech-store")
	if err != nil {
		return fmt.Errorf("step 1 (create tech store) failed: %w", err)
	}
	logVerbose("  Created tech store: %s", techStoreID)

	defer func() {
		logVerbose("Cleanup: deleting stores")
		hybridDeleteStore(ctx, httpClient, baseURL, policyStoreID)
		hybridDeleteStore(ctx, httpClient, baseURL, techStoreID)
	}()

	// --- Step 2: Upload and ingest documents ---

	logVerbose("Step 2/7: Uploading and ingesting documents")

	policyFileID, err := hybridUploadFile(ctx, httpClient, baseURL, "company-policy.txt", policyDocument)
	if err != nil {
		return fmt.Errorf("step 2 (upload policy doc) failed: %w", err)
	}
	logVerbose("  Uploaded policy doc: %s", policyFileID)

	techFileID, err := hybridUploadFile(ctx, httpClient, baseURL, "tech-guide.txt", techDocument)
	if err != nil {
		return fmt.Errorf("step 2 (upload tech doc) failed: %w", err)
	}
	logVerbose("  Uploaded tech doc: %s", techFileID)

	if err := hybridAttachFile(ctx, httpClient, baseURL, policyStoreID, policyFileID); err != nil {
		return fmt.Errorf("step 2 (attach policy file) failed: %w", err)
	}
	if err := hybridAttachFile(ctx, httpClient, baseURL, techStoreID, techFileID); err != nil {
		return fmt.Errorf("step 2 (attach tech file) failed: %w", err)
	}

	logVerbose("  Waiting for ingestion to complete...")
	if err := hybridWaitForIngestion(ctx, httpClient, baseURL, policyStoreID); err != nil {
		return fmt.Errorf("step 2 (policy ingestion) failed: %w", err)
	}
	if err := hybridWaitForIngestion(ctx, httpClient, baseURL, techStoreID); err != nil {
		return fmt.Errorf("step 2 (tech ingestion) failed: %w", err)
	}
	logVerbose("  Both stores ingested successfully")

	// --- Step 3: Vector-only search (baseline) ---

	logVerbose("Step 3/7: Vector-only search on policy store")
	if err := hybridSearchVerify(ctx, httpClient, baseURL, policyStoreID, hybridSearchOpts{
		query:       "What is the PTO policy?",
		maxResults:  3,
		expectTerms: []string{"pto", "paid time off"},
		logPrefix:   "  [vector-only]",
		verbose:     opts.Verbose,
	}); err != nil {
		return fmt.Errorf("step 3 (vector-only search) failed: %w", err)
	}

	// --- Step 4: Hybrid search (weighted mode) on policy store ---

	logVerbose("Step 4/7: Hybrid search (weighted) on policy store")
	if err := hybridSearchVerify(ctx, httpClient, baseURL, policyStoreID, hybridSearchOpts{
		query:      "PTO accrual rate and carry over",
		maxResults: 3,
		hybrid: map[string]interface{}{
			"mode":          "weighted",
			"vector_weight": 0.5,
			"bm25_weight":   0.3,
			"ngram_weight":  0.2,
		},
		expectTerms:       []string{"pto", "accrues", "carried over", "paid time off"},
		expectHybridScore: true,
		logPrefix:         "  [weighted]",
		verbose:           opts.Verbose,
	}); err != nil {
		return fmt.Errorf("step 4 (weighted hybrid search) failed: %w", err)
	}

	// --- Step 5: Hybrid search (RRF mode) on tech store ---

	logVerbose("Step 5/7: Hybrid search (RRF) on tech store")
	if err := hybridSearchVerify(ctx, httpClient, baseURL, techStoreID, hybridSearchOpts{
		query:      "Kubernetes pod deployment configuration",
		maxResults: 3,
		hybrid: map[string]interface{}{
			"mode": "rrf",
		},
		expectTerms:       []string{"kubernetes", "pod", "deployment", "container"},
		expectHybridScore: true,
		logPrefix:         "  [rrf]",
		verbose:           opts.Verbose,
	}); err != nil {
		return fmt.Errorf("step 5 (RRF hybrid search) failed: %w", err)
	}

	// --- Step 6: Cross-store isolation ---

	logVerbose("Step 6/7: Verifying store isolation")
	if err := hybridSearchExpectEmpty(ctx, httpClient, baseURL, policyStoreID, "Kubernetes deployment"); err != nil {
		return fmt.Errorf("step 6 (store isolation) failed: %w", err)
	}
	logVerbose("  Cross-store isolation verified")

	// --- Step 7: Hybrid search with custom BM25 parameters ---

	logVerbose("Step 7/7: Hybrid search with custom BM25/n-gram parameters")
	if err := hybridSearchVerify(ctx, httpClient, baseURL, techStoreID, hybridSearchOpts{
		query:      "docker container resource limits",
		maxResults: 5,
		hybrid: map[string]interface{}{
			"mode":          "weighted",
			"vector_weight": 0.4,
			"bm25_weight":   0.4,
			"ngram_weight":  0.2,
			"bm25_k1":       1.5,
			"bm25_b":        0.8,
			"ngram_size":    4,
		},
		expectTerms:       []string{"container", "resource", "limits"},
		expectHybridScore: true,
		logPrefix:         "  [custom-params]",
		verbose:           opts.Verbose,
	}); err != nil {
		return fmt.Errorf("step 7 (custom params hybrid) failed: %w", err)
	}

	logVerbose("All hybrid search steps passed")
	return nil
}

// ---------------------------------------------------------------------------
// Test documents
// ---------------------------------------------------------------------------

const policyDocument = `Company PTO Policy

All full-time employees receive 20 days of paid time off per year.
PTO accrues at a rate of 1.67 days per month.

Unused PTO can be carried over up to a maximum of 5 days into the next calendar year.
PTO requests must be submitted at least 2 weeks in advance through the HR portal.

For questions about PTO, contact HR at hr@company.example.com.

Sick Leave Policy

Employees receive 10 days of sick leave annually. Sick leave does not accrue
and resets on January 1st each year. A doctor's note is required for absences
exceeding 3 consecutive days.`

const techDocument = `Kubernetes Deployment Guide

This guide covers deploying applications on Kubernetes clusters.

Pod Configuration

A Kubernetes pod is the smallest deployable unit. Each pod contains one or more
containers that share networking and storage. Pod specifications include resource
limits for CPU and memory.

Example pod configuration:
- Container image: myapp:latest
- CPU request: 100m, CPU limit: 500m
- Memory request: 128Mi, Memory limit: 512Mi

Deployment Strategy

A Kubernetes deployment manages pod replicas and rolling updates. Set the
deployment strategy to RollingUpdate for zero-downtime deployments. Configure
maxSurge and maxUnavailable to control the rollout speed.

Docker Container Resource Limits

When running containers with Docker, set resource limits using the --memory
and --cpus flags. For example: docker run --memory=512m --cpus=1.0 myapp.
Resource limits prevent containers from consuming excessive host resources.`

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type hybridSearchOpts struct {
	query             string
	maxResults        int
	hybrid            map[string]interface{}
	expectTerms       []string
	expectHybridScore bool
	logPrefix         string
	verbose           bool
}

func hybridCreateStore(ctx context.Context, client *http.Client, baseURL, name string) (string, error) {
	body, _ := json.Marshal(map[string]interface{}{"name": name})
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

	id, _ := result["id"].(string)
	if id == "" {
		return "", fmt.Errorf("no id in response: %v", result)
	}
	return id, nil
}

func hybridUploadFile(ctx context.Context, client *http.Client, baseURL, filename, content string) (string, error) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	fw, err := w.CreateFormFile("file", filename)
	if err != nil {
		return "", err
	}
	if _, err := fw.Write([]byte(content)); err != nil {
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

	id, _ := result["id"].(string)
	if id == "" {
		return "", fmt.Errorf("no id in response: %v", result)
	}
	return id, nil
}

func hybridAttachFile(ctx context.Context, client *http.Client, baseURL, storeID, fileID string) error {
	body, _ := json.Marshal(map[string]interface{}{"file_id": fileID})
	url := fmt.Sprintf("%s/v1/vector_stores/%s/files", baseURL, storeID)
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
	return nil
}

func hybridWaitForIngestion(ctx context.Context, client *http.Client, baseURL, storeID string) error {
	url := fmt.Sprintf("%s/v1/vector_stores/%s/files", baseURL, storeID)
	deadline := time.Now().Add(90 * time.Second)

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
			return nil
		}

		time.Sleep(500 * time.Millisecond)
	}

	return fmt.Errorf("ingestion did not complete within timeout")
}

func hybridSearchVerify(ctx context.Context, client *http.Client, baseURL, storeID string, searchOpts hybridSearchOpts) error {
	reqBody := map[string]interface{}{
		"query":           searchOpts.query,
		"max_num_results": searchOpts.maxResults,
	}
	if searchOpts.hybrid != nil {
		reqBody["hybrid"] = searchOpts.hybrid
	}

	body, _ := json.Marshal(reqBody)
	url := fmt.Sprintf("%s/v1/vector_stores/%s/search", baseURL, storeID)
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

	if searchOpts.verbose {
		prettyJSON, _ := json.MarshalIndent(result, "  ", "  ")
		fmt.Printf("%s Search response:\n  %s\n", searchOpts.logPrefix, string(prettyJSON))
	}

	data, ok := result["data"].([]interface{})
	if !ok || len(data) == 0 {
		return fmt.Errorf("search returned no results for query %q", searchOpts.query)
	}

	// Check that at least one result contains one of the expected terms.
	foundRelevant := false
	for _, item := range data {
		sr, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		content := strings.ToLower(fmt.Sprintf("%v", sr["content"]))
		for _, term := range searchOpts.expectTerms {
			if strings.Contains(content, strings.ToLower(term)) {
				foundRelevant = true
				break
			}
		}

		// When hybrid is used, verify component scores are present.
		if searchOpts.expectHybridScore {
			if _, hasVec := sr["vector_score"]; !hasVec {
				return fmt.Errorf("hybrid result missing vector_score field")
			}
			if _, hasBM25 := sr["bm25_score"]; !hasBM25 {
				return fmt.Errorf("hybrid result missing bm25_score field")
			}
			if _, hasNgram := sr["ngram_score"]; !hasNgram {
				return fmt.Errorf("hybrid result missing ngram_score field")
			}
		}

		if foundRelevant {
			break
		}
	}

	if !foundRelevant {
		return fmt.Errorf("search results for %q do not contain any expected terms %v",
			searchOpts.query, searchOpts.expectTerms)
	}

	if searchOpts.verbose {
		fmt.Printf("%s %d results, relevant content found\n", searchOpts.logPrefix, len(data))
	}
	return nil
}

func hybridSearchExpectEmpty(ctx context.Context, client *http.Client, baseURL, storeID, query string) error {
	reqBody := map[string]interface{}{
		"query":           query,
		"max_num_results": 3,
		"ranking_options": map[string]interface{}{
			"score_threshold": 0.8,
		},
	}

	body, _ := json.Marshal(reqBody)
	url := fmt.Sprintf("%s/v1/vector_stores/%s/search", baseURL, storeID)
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

	data, _ := result["data"].([]interface{})
	for _, item := range data {
		sr, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		content := strings.ToLower(fmt.Sprintf("%v", sr["content"]))
		if strings.Contains(content, "kubernetes") || strings.Contains(content, "docker") {
			return fmt.Errorf("policy store returned cross-domain tech content: %s", content)
		}
	}

	return nil
}

func hybridDeleteStore(ctx context.Context, client *http.Client, baseURL, storeID string) {
	url := fmt.Sprintf("%s/v1/vector_stores/%s", baseURL, storeID)
	req, _ := http.NewRequestWithContext(ctx, "DELETE", url, nil)
	resp, err := client.Do(req)
	if err == nil {
		resp.Body.Close()
	}
}
