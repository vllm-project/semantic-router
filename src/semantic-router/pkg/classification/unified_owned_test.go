package classification

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func TestOwnedUnifiedLegacyPlaceholderFailsExplicitly(t *testing.T) {
	err := (&UnifiedClassifier{}).Initialize("base", "intent", "pii", "security", []string{"a"}, []string{"O"}, []string{"safe"}, true)
	if !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("placeholder initialized without real task bindings: %v", err)
	}
}

func TestOwnedUnifiedCloseWaitsForInference(t *testing.T) {
	started, finish, closed := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var calls atomic.Int32
	classifier := &UnifiedClassifier{initialized: true}
	classifier.testClassifyBatchLegacy = func(texts []string) (*UnifiedBatchResults, error) {
		calls.Add(1)
		close(started)
		<-finish
		return &UnifiedBatchResults{BatchSize: len(texts), IntentResults: make([]IntentResult, len(texts)), PIIResults: make([]PIIResult, len(texts)), SecurityResults: make([]SecurityResult, len(texts))}, nil
	}
	returned := make(chan error, 1)
	go func() {
		_, err := classifier.ClassifyBatchContext(context.Background(), []string{"one"})
		returned <- err
	}()
	<-started
	go func() { _ = classifier.Close(); close(closed) }()
	select {
	case <-closed:
		t.Fatal("closed during a live model call")
	case <-time.After(20 * time.Millisecond):
	}
	close(finish)
	if err := <-returned; err != nil {
		t.Fatal(err)
	}
	<-closed
	if _, err := classifier.ClassifyBatch([]string{"two"}); !errors.Is(err, binding.ErrClosed) {
		t.Fatalf("closed owner was usable: %v", err)
	}
	if calls.Load() != 1 {
		t.Fatal("closed owner reached native inference")
	}
}

func TestOwnedUnifiedUnavailableScoresAreNull(t *testing.T) {
	available := false
	for _, result := range []any{PIIResult{Confidence: .9, ScoresAvailable: &available}, SecurityResult{Confidence: .9, ScoresAvailable: &available}} {
		data, err := json.Marshal(result)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(data), `"confidence":null`) || !strings.Contains(string(data), `"scores_available":false`) {
			t.Fatalf("invented unavailable confidence: %s", data)
		}
	}
}

func TestOwnedUnifiedSecurityLabelPolarity(t *testing.T) {
	labels := []string{"safe", "unsafe", "benign", "no_threat", "harmful"}
	for _, sample := range []struct {
		label  string
		threat bool
	}{{"safe", false}, {"unsafe", true}, {"benign", false}, {"no_threat", false}, {"harmful", true}} {
		threat, err := legacyLoRAThreatLabel(sample.label, labels)
		if err != nil || threat != sample.threat {
			t.Fatalf("%s: threat=%v err=%v", sample.label, threat, err)
		}
	}
	if _, err := legacyLoRAThreatLabel("unknown", labels); err == nil {
		t.Fatal("unknown label treated as a verdict")
	}
}

func TestOwnedModalityUsesDeclaredOrder(t *testing.T) {
	labels, err := modalityLabels([]string{"both", "AR", "diffusion"})
	if err != nil || !reflect.DeepEqual(labels, []string{"BOTH", "AR", "DIFFUSION"}) {
		t.Fatalf("mapping reordered: %v %v", labels, err)
	}
	if _, err = modalityLabels([]string{"AR", "AR", "BOTH"}); err == nil {
		t.Fatal("ambiguous modality mapping accepted")
	}
	if _, err = modalityLabels([]string{"LABEL_0", "LABEL_1", "LABEL_2"}); err == nil {
		t.Fatal("unknown modality semantics accepted")
	}
}
