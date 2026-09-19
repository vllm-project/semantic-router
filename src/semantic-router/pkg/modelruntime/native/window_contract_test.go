package native

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func TestWindowContractsDoNotMixCategoricalAndIndependentScores(t *testing.T) {
	input := &tasks.InputUsage{OriginalTokens: 8, ProcessedTokens: 8}
	windows := tasks.WindowedLabelScores{ContentTokens: 6, Input: input, Windows: []tasks.LabelScoresWindow{{Start: 0, End: 4, Scores: []float32{.8, .9}}, {Start: 2, End: 6, Scores: []float32{.1, .2}}}}
	if err := validateWindowScores(tasks.TextWindowsRequest{}, windows); err != nil {
		t.Fatal(err)
	}
	categorical := tasks.WindowedLabelDistribution{ContentTokens: 6, Input: input, Windows: []tasks.LabelDistributionWindow{{Start: 0, End: 4, Probabilities: []float32{.8, .9}}, {Start: 2, End: 6, Probabilities: []float32{.1, .9}}}}
	if err := validateWindowDistribution(tasks.TextWindowsRequest{}, categorical); err == nil {
		t.Fatal("categorical path accepted independent probabilities")
	}
	categorical.Windows[0].Probabilities = []float32{.8, .2}
	if err := validateWindowDistribution(tasks.TextWindowsRequest{}, categorical); err != nil {
		t.Fatal(err)
	}
	windows.Windows[1].Scores = []float32{.1}
	if err := validateWindowScores(tasks.TextWindowsRequest{}, windows); err == nil {
		t.Fatal("accepted a partial final label vector")
	}
}
