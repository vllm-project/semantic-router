package classification

import (
	"reflect"
	"sort"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// localEntitiesFor renders a fixture the way the native Candle backend reports
// it: byte offsets into the request string, the mapping's label spelling, the
// entity text and score.
func localEntitiesFor(tc tokenSpansFixtureCase) []candle_binding.TokenEntity {
	entities := make([]candle_binding.TokenEntity, 0, len(tc.Spans))
	for _, sp := range tc.Spans {
		entities = append(entities, candle_binding.TokenEntity{
			EntityType: sp.Label,
			Start:      sp.ByteStart,
			End:        sp.ByteEnd,
			Text:       sp.Text,
			Confidence: sp.Score,
		})
	}
	return entities
}

// The parity gate from #2922: for every accepted fixture, the local backend and
// the remote token_spans.v1 backend must produce the same PIIDetected,
// PIIEntities and MatchedPIIRules for the routing signal, and the same
// PIIDetection list (type, text, byte offsets, score) that the detection API
// hands to masking. Masked text is a pure function of that list, covered on the
// services side by TestBuildMaskedPIITextLocalAndRemoteOffsetsAgree.
func TestLocalAndRemotePIIBackendsAgreeOnFixtures(t *testing.T) {
	for _, tc := range loadTokenSpansFixtures(t) {
		if tc.Expect != "accept" {
			continue
		}
		t.Run(tc.Name, func(t *testing.T) {
			remote := newRemotePIIClassifier(t, config.OnErrorAllow, func(string) any { return wireSpans(tc.Spans) })
			localModel := &MockPIIInference{responseMap: make(map[string]MockPIIInferenceResponse)}
			localModel.setMockResponse(tc.Text, localEntitiesFor(tc), nil)
			local := newPIIRuleClassifier(t, config.OnErrorAllow, localModel)
			assertSignalParity(t, remote, local, tc.Text)
			assertDetectionParity(t, remote, local, tc)
		})
	}
}

func assertSignalParity(t *testing.T, remote, local *Classifier, text string) {
	t.Helper()
	r, l := runPIISignal(remote, text), runPIISignal(local, text)
	if r.PIIDetected != l.PIIDetected {
		t.Fatalf("PIIDetected remote=%v local=%v", r.PIIDetected, l.PIIDetected)
	}
	if !reflect.DeepEqual(r.PIIEntities, l.PIIEntities) {
		t.Fatalf("PIIEntities remote=%v local=%v", r.PIIEntities, l.PIIEntities)
	}
	if !reflect.DeepEqual(r.MatchedPIIRules, l.MatchedPIIRules) {
		t.Fatalf("MatchedPIIRules remote=%v local=%v", r.MatchedPIIRules, l.MatchedPIIRules)
	}
}

func assertDetectionParity(t *testing.T, remote, local *Classifier, tc tokenSpansFixtureCase) {
	t.Helper()
	remoteDetections, err := remote.ClassifyPIIWithDetails(tc.Text)
	if err != nil {
		t.Fatalf("remote detections: %v", err)
	}
	localDetections, err := local.ClassifyPIIWithDetails(tc.Text)
	if err != nil {
		t.Fatalf("local detections: %v", err)
	}
	sortDetections(remoteDetections)
	sortDetections(localDetections)
	if !reflect.DeepEqual(remoteDetections, localDetections) {
		t.Fatalf("detections differ\nremote=%+v\nlocal =%+v", remoteDetections, localDetections)
	}
	if len(remoteDetections) != len(tc.Spans) {
		t.Fatalf("got %d detections, fixture has %d spans", len(remoteDetections), len(tc.Spans))
	}
}

func sortDetections(d []PIIDetection) {
	sort.Slice(d, func(i, j int) bool {
		if d[i].Start != d[j].Start {
			return d[i].Start < d[j].Start
		}
		if d[i].End != d[j].End {
			return d[i].End < d[j].End
		}
		return d[i].EntityType < d[j].EntityType
	})
}
