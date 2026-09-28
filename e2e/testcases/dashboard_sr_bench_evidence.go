package testcases

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"net/http"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

const srBenchDeployment = "semantic-router-sr-bench"

type dashboardBenchReport struct {
	Version    string `json:"version"`
	RunID      string `json:"run_id"`
	Status     string `json:"status"`
	Provenance struct {
		PlanSHA string `json:"plan_sha256"`
	} `json:"provenance"`
	Summary struct {
		Targets []struct {
			ID           string         `json:"id"`
			Total        int            `json:"total"`
			Completed    int            `json:"completed"`
			Correct      int            `json:"correct"`
			RequestCount int            `json:"request_count"`
			Accuracy     float64        `json:"accuracy"`
			Cost         *float64       `json:"cost_usd"`
			CostComplete bool           `json:"cost_complete"`
			Tokens       map[string]int `json:"tokens"`
		} `json:"targets"`
	} `json:"summary"`
}

func verifySrBenchReport(raw json.RawMessage, run dashboardBenchRun) error {
	var report dashboardBenchReport
	if err := json.Unmarshal(raw, &report); err != nil {
		return err
	}
	if report.Version != "sr-bench-1.0" || report.Status != "completed" || report.RunID != run.ID || report.Provenance.PlanSHA != run.Manifest.PlanSHA || len(report.Summary.Targets) != 1 {
		return fmt.Errorf("sr-bench report identity/status differs from completed frozen run")
	}
	score := report.Summary.Targets[0]
	if score.ID != "fixture" || score.Total != 2 || score.Completed != 2 || score.Correct != 2 || score.Accuracy != 1 || score.RequestCount != 2 {
		return fmt.Errorf("synthetic final-channel score or denominator differs: %+v", score)
	}
	if !score.CostComplete || score.Cost == nil || math.Abs(*score.Cost-36.4/1_000_000) > 1e-12 {
		return fmt.Errorf("synthetic four-bucket cost is missing or incorrect")
	}
	want := map[string]int{"input_tokens": 14, "cached_input_tokens": 4, "cache_write_tokens": 2, "output_tokens": 6}
	if len(score.Tokens) != len(want) {
		return fmt.Errorf("expected all four token buckets")
	}
	for bucket, value := range want {
		if score.Tokens[bucket] != value {
			return fmt.Errorf("%s=%d, expected %d", bucket, score.Tokens[bucket], value)
		}
	}
	return nil
}

// Digest the persisted public evidence, not SQLite/WAL bytes that legitimately
// change as other runs are journaled. A restart must preserve every saved field.
func srBenchEvidenceDigest(ctx context.Context, client *http.Client, base, token string, run dashboardBenchRun) (string, error) {
	snapshot := map[string]json.RawMessage{}
	for _, action := range []string{"report", "results", "calls", "events"} {
		var raw json.RawMessage
		if err := srBenchJSON(ctx, client, http.MethodGet, base+srBenchAPI+"/runs/"+run.ID+"/"+action, token, nil, &raw, http.StatusOK); err != nil {
			return "", err
		}
		snapshot[action] = raw
	}
	if err := verifySrBenchReport(snapshot["report"], run); err != nil {
		return "", err
	}
	var results struct {
		Total   int `json:"total"`
		Results []struct {
			Status  string `json:"status"`
			Answer  string `json:"answer"`
			Correct bool   `json:"correct"`
		} `json:"results"`
	}
	if err := json.Unmarshal(snapshot["results"], &results); err != nil {
		return "", err
	}
	if results.Total != 2 || len(results.Results) != 2 {
		return "", fmt.Errorf("missing completed case evidence")
	}
	for _, result := range results.Results {
		if result.Status != "completed" || result.Answer != "A" || !result.Correct {
			return "", fmt.Errorf("grader did not use the final A independently of reasoning B")
		}
	}
	var calls struct {
		Total int `json:"total"`
		Calls []struct {
			ID     string `json:"id"`
			Status string `json:"status"`
		} `json:"calls"`
	}
	if err := json.Unmarshal(snapshot["calls"], &calls); err != nil {
		return "", err
	}
	if calls.Total != 2 || len(calls.Calls) != 2 || calls.Calls[0].ID == calls.Calls[1].ID {
		return "", fmt.Errorf("expected exactly two unique inference receipts")
	}
	for _, call := range calls.Calls {
		if call.Status != "completed" {
			return "", fmt.Errorf("inference receipt %s is not complete", call.ID)
		}
		var raw json.RawMessage
		if err := srBenchJSON(ctx, client, http.MethodGet, base+srBenchAPI+"/runs/"+run.ID+"/calls/"+call.ID, token, nil, &raw, http.StatusOK); err != nil {
			return "", err
		}
		var detail struct {
			Final     string `json:"final"`
			Reasoning string `json:"reasoning"`
		}
		if err := json.Unmarshal(raw, &detail); err != nil {
			return "", err
		}
		if detail.Final != "A" || detail.Reasoning != "B" {
			return "", fmt.Errorf("full call detail did not preserve separate final/reasoning channels")
		}
		snapshot[call.ID] = raw
	}
	serialized, err := json.Marshal(snapshot)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", sha256.Sum256(serialized)), nil
}

func srBenchWorkerIdentity(ctx context.Context, client *kubernetes.Clientset) (string, error) {
	pods, err := client.CoreV1().Pods(dashboardRestartNamespace).List(ctx, metav1.ListOptions{LabelSelector: "app=" + srBenchDeployment})
	if err != nil {
		return "", err
	}
	if len(pods.Items) != 1 || pods.Items[0].DeletionTimestamp != nil {
		return "", fmt.Errorf("expected exactly one independent benchmark worker pod")
	}
	for _, status := range pods.Items[0].Status.ContainerStatuses {
		if status.Name == "worker" && status.Ready && status.State.Running != nil && status.ContainerID != "" {
			return fmt.Sprintf("%s/%s/%d", pods.Items[0].UID, status.ContainerID, status.RestartCount), nil
		}
	}
	return "", fmt.Errorf("independent benchmark worker is not ready")
}
