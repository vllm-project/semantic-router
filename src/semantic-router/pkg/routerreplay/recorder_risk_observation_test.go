package routerreplay

import "testing"

func TestGuardRiskObservationReplayLog(t *testing.T) {
	for _, risk := range []float32{0, .2} {
		for _, enabled := range []bool{false, true} {
			fields := map[string]interface{}{}
			appendGuardrailLogFields(fields, RoutingRecord{JailbreakEnabled: enabled, JailbreakScoreAvailable: true, JailbreakConfidence: risk})
			if fields["jailbreak_detected"] != false || fields["jailbreak_score_available"] != true || fields["jailbreak_confidence"] != risk {
				t.Fatalf("non-match log risk dropped: %+v", fields)
			}
		}
	}
	fields := map[string]interface{}{}
	appendGuardrailLogFields(fields, RoutingRecord{JailbreakEnabled: true, JailbreakDetected: true})
	if _, ok := fields["jailbreak_confidence"]; ok {
		t.Fatal("invented unavailable probability")
	}
}
