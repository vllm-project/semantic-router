package mlpipeline

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"path/filepath"
	"strings"
)

// sseEvent represents a single SSE event from the ML service.
type sseEvent struct {
	Percent     int      `json:"percent"`
	Step        string   `json:"step"`
	Message     string   `json:"message"`
	Done        bool     `json:"done,omitempty"`
	Success     bool     `json:"success,omitempty"`
	OutputFiles []string `json:"output_files,omitempty"`
}

// readSSEStream reads SSE events from the ML service response and relays
// progress updates. It blocks until the stream is closed or a "done" event
// is received. Returns (outputFiles, error).
func (r *Runner) readSSEStream(jobID string, body io.ReadCloser) ([]string, error) {
	defer body.Close()

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 256*1024), 256*1024)

	var finalFiles []string
	var finalErr error

	for scanner.Scan() {
		line := scanner.Text()

		// SSE format: "data: {...json...}"
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		var evt sseEvent
		if err := json.Unmarshal([]byte(data), &evt); err != nil {
			log.Printf("[ml-service/%s] Failed to parse SSE event: %v (data: %s)", jobID, err, data)
			continue
		}

		// Relay progress to our own SSE channel
		r.sendProgress(jobID, evt.Percent, evt.Step, evt.Message)
		if evt.Done {
			if evt.Success {
				finalFiles = evt.OutputFiles
			} else {
				finalErr = fmt.Errorf("ML service error: %s", evt.Message)
			}
			break
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("SSE stream read error: %w", err)
	}

	return finalFiles, finalErr
}

// runBenchmarkHTTP delegates benchmark to the Python ML sidecar via HTTP.
func (r *Runner) runBenchmarkHTTP(ctx context.Context, modelsYAMLPath, queryJSONLPath string, req BenchmarkRequest) (string, error) {
	job := r.createJob("benchmark")
	jobDir := r.JobDir(job.ID)
	if err := ensureDir(jobDir); err != nil {
		return "", fmt.Errorf("failed to create job dir: %w", err)
	}

	r.mu.Lock()
	job.Status = StatusRunning
	r.mu.Unlock()

	go func() {
		r.sendProgress(job.ID, 5, "Starting benchmark", "Sending request to ML service")

		payload := map[string]interface{}{
			"queries_path":     queryJSONLPath,
			"models_yaml_path": modelsYAMLPath,
			"output_dir":       jobDir,
			"concurrency":      req.Concurrency,
			"max_tokens":       req.MaxTokens,
			"temperature":      req.Temperature,
			"concise":          req.Concise,
			"limit":            req.Limit,
		}

		body, err := json.Marshal(payload)
		if err != nil {
			r.failJob(job.ID, fmt.Sprintf("failed to marshal request: %v", err))
			r.sendProgress(job.ID, 100, "Failed", err.Error())
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST", r.mlServiceURL+"/api/benchmark", bytes.NewReader(body))
		if err != nil {
			r.failJob(job.ID, fmt.Sprintf("failed to create request: %v", err))
			r.sendProgress(job.ID, 100, "Failed", err.Error())
			return
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "text/event-stream")

		log.Printf("[benchmark/%s] Calling ML service: POST %s/api/benchmark", job.ID, r.mlServiceURL)
		resp, err := (&http.Client{Timeout: 0}).Do(httpReq)
		if err != nil {
			r.failJob(job.ID, fmt.Sprintf("ML service request failed: %v", err))
			r.sendProgress(job.ID, 100, "Failed", err.Error())
			return
		}
		if resp.StatusCode != http.StatusOK {
			respBody, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			errMsg := fmt.Sprintf("ML service returned %d: %s", resp.StatusCode, string(respBody))
			r.failJob(job.ID, errMsg)
			r.sendProgress(job.ID, 100, "Failed", errMsg)
			return
		}

		// Read SSE stream
		outputFiles, streamErr := r.readSSEStream(job.ID, resp.Body)
		if streamErr != nil {
			r.failJob(job.ID, streamErr.Error())
			r.sendProgress(job.ID, 100, "Failed", streamErr.Error())
			return
		}

		r.completeJob(job.ID, outputFiles)
		r.sendProgress(job.ID, 100, "Completed", "Benchmark finished successfully")
	}()

	return job.ID, nil
}

// runTrainHTTP delegates training to the Python ML sidecar via HTTP.
func (r *Runner) runTrainHTTP(ctx context.Context, benchmarkDataPath string, req TrainRequest) (string, error) {
	job := r.createJob("train")
	jobDir := r.TrainDir()
	if err := ensureDir(jobDir); err != nil {
		return "", fmt.Errorf("failed to create job dir: %w", err)
	}

	r.mu.Lock()
	job.Status = StatusRunning
	r.mu.Unlock()

	go func() {
		r.sendProgress(job.ID, 5, "Starting training", "Sending request to ML service")

		algorithms := req.Algorithms
		if len(algorithms) == 0 {
			algorithms = []string{"knn", "kmeans", "svm", "mlp"}
		}
		device := req.Device
		if device == "" {
			device = "cpu"
		}

		payload := map[string]interface{}{
			"data_file":         benchmarkDataPath,
			"output_dir":        jobDir,
			"algorithms":        algorithms,
			"device":            device,
			"embedding_model":   req.EmbeddingModel,
			"cache_dir":         filepath.Join(jobDir, ".cache"),
			"quality_weight":    req.QualityWeight,
			"batch_size":        req.BatchSize,
			"knn_k":             req.KnnK,
			"kmeans_clusters":   req.KmeansClusters,
			"svm_kernel":        req.SvmKernel,
			"svm_gamma":         req.SvmGamma,
			"mlp_hidden_sizes":  req.MlpHiddenSizes,
			"mlp_epochs":        req.MlpEpochs,
			"mlp_learning_rate": req.MlpLearningRate,
			"mlp_dropout":       req.MlpDropout,
		}

		body, err := json.Marshal(payload)
		if err != nil {
			r.failJob(job.ID, fmt.Sprintf("failed to marshal request: %v", err))
			r.sendProgress(job.ID, 100, "Failed", err.Error())
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST", r.mlServiceURL+"/api/train", bytes.NewReader(body))
		if err != nil {
			r.failJob(job.ID, fmt.Sprintf("failed to create request: %v", err))
			r.sendProgress(job.ID, 100, "Failed", err.Error())
			return
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "text/event-stream")

		algNames := strings.Join(algorithms, ", ")
		log.Printf("[train/%s] Calling ML service: POST %s/api/train (algorithms=%s, device=%s)", job.ID, r.mlServiceURL, algNames, device)
		resp, err := (&http.Client{Timeout: 0}).Do(httpReq) // Training can take many minutes
		if err != nil {
			r.failJob(job.ID, fmt.Sprintf("ML service request failed: %v", err))
			r.sendProgress(job.ID, 100, "Failed", err.Error())
			return
		}
		if resp.StatusCode != http.StatusOK {
			respBody, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			errMsg := fmt.Sprintf("ML service returned %d: %s", resp.StatusCode, string(respBody))
			r.failJob(job.ID, errMsg)
			r.sendProgress(job.ID, 100, "Failed", errMsg)
			return
		}
		// Read SSE stream
		outputFiles, streamErr := r.readSSEStream(job.ID, resp.Body)
		if streamErr != nil {
			r.failJob(job.ID, streamErr.Error())
			r.sendProgress(job.ID, 100, "Failed", streamErr.Error())
			return
		}

		r.completeJob(job.ID, outputFiles)
		r.sendProgress(job.ID, 100, "Completed", fmt.Sprintf("Training finished: %d model(s) generated", len(outputFiles)))
	}()

	return job.ID, nil
}
