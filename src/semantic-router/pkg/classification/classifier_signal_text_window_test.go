package classification

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestTextForRoutingSignalBoundsSemanticInferenceButPreservesExactSignals(t *testing.T) {
	longText := strings.Repeat("a ", semanticSignalUnitLimit) + "middle-marker" + strings.Repeat("z ", semanticSignalUnitLimit)

	semantic := textForRoutingSignal(config.SignalTypeLanguage, longText)
	if units := signalUnits(semantic); units > semanticSignalUnitLimit+2*signalUnits(signalWindowOmissionMarker) {
		t.Fatalf("semantic signal text was not bounded: %d units", units)
	}
	if !strings.HasPrefix(semantic, strings.Repeat("a ", 32)) || !strings.HasSuffix(semantic, strings.Repeat("z ", 32)) {
		t.Fatal("semantic signal text must preserve both request boundaries")
	}
	if !strings.Contains(semantic, "middle-marker") {
		t.Fatal("semantic signal text must sample the request middle")
	}
	if got := textForRoutingSignal(config.SignalTypeKeyword, longText); got != longText {
		t.Fatal("keyword signal must retain the full exact-match text")
	}
}

func TestTextForSignalFuncBoundsUncompressedSemanticView(t *testing.T) {
	compressed := "short"
	uncompressed := strings.Repeat("x ", semanticSignalUnitLimit)
	resolve := textForSignalFunc(
		compressed,
		uncompressed,
		map[string]bool{config.SignalTypeLanguage: true},
	)
	got := resolve(config.SignalTypeLanguage)
	if !strings.Contains(got, signalWindowOmissionMarker) {
		t.Fatal("uncompressed semantic view was not bounded")
	}
	// Two omission markers are added on top of the bounded content itself.
	if units := signalUnits(got); units > semanticSignalUnitLimit+2*signalUnits(signalWindowOmissionMarker) {
		t.Fatalf("unexpected bounded view size: %d units", units)
	}
}

func TestRepresentativeSignalTextBoundsCJKByTokens(t *testing.T) {
	head := strings.Repeat("請說明台灣民法中關於租賃契約終止的規定，房東與房客各自的權利義務為何？", 40)
	tail := strings.Repeat("請計算函數的極值，並證明其在區間上的積分值。", 40)
	got := representativeSignalText(head+tail, semanticSignalUnitLimit)
	if units := signalUnits(got); units/4 > 512 {
		t.Fatalf("CJK window estimates %d tokens, over the 512 cap", units/4)
	}
	if !strings.HasSuffix(got, "積分值。") || !strings.HasPrefix(got, "請說明") {
		t.Fatal("CJK window must keep head and tail")
	}
	if strings.Count(got, signalWindowOmissionMarker) != 2 {
		t.Fatal("CJK window must sample head, middle, and tail")
	}
}

// signalUnits is the unit count of text.
func signalUnits(text string) int {
	runes := []rune(text)
	return signalUnitPrefix(runes)[len(runes)]
}
