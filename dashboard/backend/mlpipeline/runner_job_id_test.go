package mlpipeline

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

func TestGenerateConfigConcurrentJobsRemainDistinct(t *testing.T) {
	dir := t.TempDir()
	store, err := workflowstore.Open(filepath.Join(dir, "workflow.sqlite"), workflowstore.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	runner, err := NewRunner(RunnerConfig{DataDir: filepath.Join(dir, "data"), Workflow: store})
	if err != nil {
		t.Fatal(err)
	}

	const jobCount = 32
	results := make([]struct {
		id  string
		err error
	}, jobCount)
	start := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(jobCount)
	for i := range jobCount {
		go func() {
			defer wg.Done()
			<-start
			results[i].id, results[i].err = runner.GenerateConfig(ConfigRequest{
				ModelsPath: fmt.Sprintf("models-%d", i),
				Device:     "cpu",
				Decisions: []DecisionEntry{{
					Name:       "general",
					Domains:    []string{"general"},
					Algorithm:  "knn",
					Priority:   100,
					ModelNames: []string{"local-model"},
				}},
			})
		}()
	}
	close(start)
	wg.Wait()

	ids := make(map[string]struct{}, jobCount)
	for i, result := range results {
		if result.err != nil {
			t.Fatalf("GenerateConfig request %d: %v", i, result.err)
		}
		if !strings.HasPrefix(result.id, "ml-config-") {
			t.Fatalf("unexpected job ID: %q", result.id)
		}
		ids[result.id] = struct{}{}
	}
	jobs, err := store.ListMLJobs()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("successful_requests=%d distinct_ids=%d persisted_jobs=%d", jobCount, len(ids), len(jobs))
	if len(ids) != jobCount || len(jobs) != jobCount {
		t.Fatalf("concurrent jobs lost identity: got %d distinct IDs and %d persisted jobs, want %d each",
			len(ids), len(jobs), jobCount)
	}

	for i, result := range results {
		job := runner.GetJob(result.id)
		if job == nil || job.Status != StatusCompleted || job.Progress != 100 || len(job.OutputFiles) != 1 {
			t.Fatalf("unexpected completed job %q: %+v", result.id, job)
		}
		wantOutput := filepath.Join(runner.JobDir(result.id), "ml-model-selection-values.yaml")
		if job.OutputFiles[0] != wantOutput {
			t.Fatalf("job %q output = %q, want %q", result.id, job.OutputFiles[0], wantOutput)
		}
		data, err := os.ReadFile(job.OutputFiles[0])
		if err != nil {
			t.Fatal(err)
		}
		var config yamlConfig
		if err := yaml.Unmarshal(data, &config); err != nil {
			t.Fatal(err)
		}
		if want := fmt.Sprintf("models-%d", i); config.Config.ModelSelection.ML.ModelsPath != want {
			t.Fatalf("job %q models path = %q, want %q",
				result.id, config.Config.ModelSelection.ML.ModelsPath, want)
		}
		events, err := runner.ListProgressEvents(result.id, 10)
		if err != nil {
			t.Fatal(err)
		}
		if len(events) != 2 {
			t.Fatalf("job %q has %d progress events, want 2", result.id, len(events))
		}
	}
	t.Logf("matching_artifacts=%d jobs_with_separate_progress=%d", jobCount, jobCount)
}
