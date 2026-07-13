package mlpipeline

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net"
	"net/http"
	"path/filepath"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

const (
	mlBenchmarkJobTimeout    = 2 * time.Hour
	mlTrainingJobTimeout     = 12 * time.Hour
	mlResponseHeaderTimeout  = 30 * time.Second
	mlSSEMaximumLineBytes    = 64 << 10
	mlSSEMaximumBodyBytes    = 16 << 20
	mlSSEMaximumLines        = 50_000
	mlSSEMaximumEvents       = 25_000
	mlSSEMaximumStepRunes    = 256
	mlSSEMaximumMessageRunes = 2048
	mlErrorBodyDrainBytes    = 8 << 10
)

var errMLServiceJobFailed = errors.New("ML service job failed")

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

	scanner := bufio.NewScanner(io.LimitReader(body, mlSSEMaximumBodyBytes+1))
	scanner.Buffer(make([]byte, 4096), mlSSEMaximumLineBytes)

	totalBytes := 0
	lineCount := 0
	eventCount := 0
	lastRelayedPercent := -1

	for scanner.Scan() {
		line := scanner.Text()
		lineCount++
		totalBytes += len(line) + 1
		if lineCount > mlSSEMaximumLines || totalBytes > mlSSEMaximumBodyBytes {
			return nil, errors.New("ML service progress stream exceeded its limit")
		}

		// SSE format: "data: {...json...}"
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		eventCount++
		if eventCount > mlSSEMaximumEvents {
			return nil, errors.New("ML service sent too many progress events")
		}

		var evt sseEvent
		decoder := json.NewDecoder(strings.NewReader(data))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&evt); err != nil {
			return nil, errors.New("ML service sent an invalid progress event")
		}
		var trailing any
		if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
			return nil, errors.New("ML service sent an invalid progress event")
		}
		if evt.Percent < 0 || evt.Percent > 100 ||
			!safeSSEText(evt.Step, mlSSEMaximumStepRunes) ||
			!safeSSEText(evt.Message, mlSSEMaximumMessageRunes) ||
			len(evt.OutputFiles) > maximumManagedOutputFiles ||
			(evt.Done && evt.Percent != 100) || (!evt.Done && evt.Percent >= 100) {
			return nil, errors.New("ML service sent an invalid progress event")
		}
		if evt.Percent < lastRelayedPercent {
			return nil, errors.New("ML service progress percent moved backwards")
		}
		for _, outputPath := range evt.OutputFiles {
			if outputPath == "" || len(outputPath) > maxManagedPathBytes || !utf8.ValidString(outputPath) {
				return nil, errors.New("ML service sent an invalid output path")
			}
		}

		if evt.Done {
			if evt.Success {
				if len(evt.OutputFiles) == 0 {
					return nil, errors.New("ML service did not provide output files")
				}
				r.sendProgress(jobID, evt.Percent, "ML service", "ML service job completed")
				return evt.OutputFiles, nil
			}
			r.sendProgress(jobID, evt.Percent, "ML service", "ML service job failed")
			return nil, errMLServiceJobFailed
		}

		// Treat the sidecar as a trust boundary: its raw step/message content is
		// validated for framing but is never persisted or relayed to clients.
		if evt.Percent > lastRelayedPercent {
			r.sendProgress(jobID, evt.Percent, "ML service", "ML service progress update")
			lastRelayedPercent = evt.Percent
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, errors.New("ML service progress stream could not be read")
	}
	return nil, errors.New("ML service progress stream ended without a terminal event")
}

func safeSSEText(value string, maxRunes int) bool {
	if !utf8.ValidString(value) || utf8.RuneCountInString(value) > maxRunes {
		return false
	}
	for _, r := range value {
		if r == '\r' || r == '\n' || r == 0 || unicode.IsControl(r) {
			return false
		}
	}
	return true
}

func newMLServiceHTTPClient() *http.Client {
	transport := &http.Transport{
		Proxy:                  nil,
		DialContext:            (&net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		ForceAttemptHTTP2:      true,
		MaxIdleConns:           8,
		MaxIdleConnsPerHost:    2,
		MaxConnsPerHost:        4,
		IdleConnTimeout:        90 * time.Second,
		TLSHandshakeTimeout:    10 * time.Second,
		ResponseHeaderTimeout:  mlResponseHeaderTimeout,
		MaxResponseHeaderBytes: 64 << 10,
		ExpectContinueTimeout:  time.Second,
		TLSClientConfig:        &tls.Config{MinVersion: tls.VersionTLS12},
	}
	return &http.Client{
		Transport: transport,
		Timeout:   mlTrainingJobTimeout,
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
}

func validateMLServiceResponse(resp *http.Response) error {
	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, mlErrorBodyDrainBytes))
		_ = resp.Body.Close()
		return fmt.Errorf("ML service returned HTTP %d", resp.StatusCode)
	}
	mediaType, _, err := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	if err != nil || mediaType != "text/event-stream" {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, mlErrorBodyDrainBytes))
		_ = resp.Body.Close()
		return errors.New("ML service returned an unexpected content type")
	}
	return nil
}

// runBenchmarkHTTP delegates benchmark to the Python ML sidecar via HTTP.
func (r *Runner) runBenchmarkHTTP(ctx context.Context, modelsYAMLPath, queryJSONLPath string, req BenchmarkRequest, release func()) (string, error) {
	job, err := r.createJob("benchmark")
	if err != nil {
		release()
		return "", err
	}
	jobDir := r.JobDir(job.ID)
	if err := ensureDir(jobDir); err != nil {
		r.failJob(job.ID, "benchmark output directory could not be created")
		release()
		return "", fmt.Errorf("failed to create job dir: %w", err)
	}

	if err := r.setJobRunning(job); err != nil {
		release()
		return "", err
	}

	r.startAsyncJob(job.ID, release, func() {
		defer r.cleanupManagedUploads(modelsYAMLPath, queryJSONLPath)
		jobCtx, cancel := context.WithTimeout(ctx, mlBenchmarkJobTimeout)
		defer cancel()
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
			r.failJob(job.ID, "ML service request could not be encoded")
			r.sendProgress(job.ID, 100, "Failed", "ML service request could not be encoded")
			return
		}

		httpReq, err := http.NewRequestWithContext(jobCtx, "POST", r.mlServiceURL+"/api/benchmark", bytes.NewReader(body))
		if err != nil {
			r.failJob(job.ID, "ML service request could not be created")
			r.sendProgress(job.ID, 100, "Failed", "ML service request could not be created")
			return
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "text/event-stream")

		resp, err := r.mlHTTPClient.Do(httpReq)
		if err != nil {
			r.failJob(job.ID, "ML service request failed")
			r.sendProgress(job.ID, 100, "Failed", "ML service request failed")
			return
		}
		if validationErr := validateMLServiceResponse(resp); validationErr != nil {
			r.failJob(job.ID, validationErr.Error())
			r.sendProgress(job.ID, 100, "Failed", "ML service response was rejected")
			return
		}

		// Read SSE stream
		outputFiles, streamErr := r.readSSEStream(job.ID, resp.Body)
		if streamErr != nil {
			r.failJob(job.ID, "ML service progress stream was rejected")
			r.sendProgress(job.ID, 100, "Failed", "ML service progress stream was rejected")
			return
		}

		if err := r.completeJob(job.ID, outputFiles); err != nil {
			r.failJob(job.ID, "ML service output was rejected")
			r.sendProgress(job.ID, 100, "Failed", "ML service output was rejected")
			return
		}
		r.sendProgress(job.ID, 100, "Completed", "Benchmark finished successfully")
	})

	return job.ID, nil
}

// runTrainHTTP delegates training to the Python ML sidecar via HTTP.
func (r *Runner) runTrainHTTP(ctx context.Context, benchmarkDataPath string, req TrainRequest, release func()) (string, error) {
	job, err := r.createJob("train")
	if err != nil {
		release()
		return "", err
	}
	jobDir := r.TrainDir()
	if err := ensureDir(jobDir); err != nil {
		r.failJob(job.ID, "training output directory could not be created")
		release()
		return "", fmt.Errorf("failed to create job dir: %w", err)
	}

	if err := r.setJobRunning(job); err != nil {
		release()
		return "", err
	}

	r.startAsyncJob(job.ID, release, func() {
		r.executeTrainHTTPJob(ctx, job.ID, jobDir, benchmarkDataPath, req)
	})

	return job.ID, nil
}

func (r *Runner) executeTrainHTTPJob(
	ctx context.Context,
	jobID string,
	jobDir string,
	benchmarkDataPath string,
	req TrainRequest,
) {
	defer r.cleanupManagedUploads(benchmarkDataPath)
	r.trainMu.Lock()
	defer r.trainMu.Unlock()
	if prepareErr := prepareTrainOutputDir(jobDir); prepareErr != nil {
		r.failJob(jobID, "training output directory could not be prepared")
		r.sendProgress(jobID, 100, "Failed", "Training output directory could not be prepared")
		return
	}
	jobCtx, cancel := context.WithTimeout(ctx, mlTrainingJobTimeout)
	defer cancel()
	r.sendProgress(jobID, 5, "Starting training", "Sending request to ML service")

	body, marshalErr := json.Marshal(buildTrainHTTPPayload(benchmarkDataPath, jobDir, req))
	if marshalErr != nil {
		r.failJob(jobID, "ML service request could not be encoded")
		r.sendProgress(jobID, 100, "Failed", "ML service request could not be encoded")
		return
	}

	httpReq, requestErr := http.NewRequestWithContext(jobCtx, "POST", r.mlServiceURL+"/api/train", bytes.NewReader(body))
	if requestErr != nil {
		r.failJob(jobID, "ML service request could not be created")
		r.sendProgress(jobID, 100, "Failed", "ML service request could not be created")
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, requestErr := r.mlHTTPClient.Do(httpReq)
	if requestErr != nil {
		r.failJob(jobID, "ML service request failed")
		r.sendProgress(jobID, 100, "Failed", "ML service request failed")
		return
	}
	if validationErr := validateMLServiceResponse(resp); validationErr != nil {
		r.failJob(jobID, validationErr.Error())
		r.sendProgress(jobID, 100, "Failed", "ML service response was rejected")
		return
	}
	outputFiles, streamErr := r.readSSEStream(jobID, resp.Body)
	if streamErr != nil {
		r.failJob(jobID, "ML service progress stream was rejected")
		r.sendProgress(jobID, 100, "Failed", "ML service progress stream was rejected")
		return
	}

	snapshots, snapshotErr := r.snapshotTrainOutputFiles(jobID, outputFiles)
	if snapshotErr != nil {
		r.failJob(jobID, "training output could not be snapshotted")
		r.sendProgress(jobID, 100, "Failed", "Training output could not be snapshotted")
		return
	}
	if completeErr := r.completeJob(jobID, snapshots); completeErr != nil {
		r.failJob(jobID, "ML service output was rejected")
		r.sendProgress(jobID, 100, "Failed", "ML service output was rejected")
		return
	}
	r.sendProgress(jobID, 100, "Completed", fmt.Sprintf("Training finished: %d model(s) generated", len(snapshots)))
}

func buildTrainHTTPPayload(benchmarkDataPath, jobDir string, req TrainRequest) map[string]interface{} {
	algorithms := req.Algorithms
	if len(algorithms) == 0 {
		algorithms = []string{"knn", "kmeans", "svm", "mlp"}
	}
	device := req.Device
	if device == "" {
		device = "cpu"
	}

	return map[string]interface{}{
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
}
