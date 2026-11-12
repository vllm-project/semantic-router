package classification

import (
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// longChapter is a benign made-up text of about tokens estimated tokens: an
// opening line, numbered paragraphs, and an ending.
func longChapter(tokens int, ending string) string {
	return longChapterWith(tokens, nil, ending)
}

// longChapterWith is longChapter with inserts placed after the paragraphs
// their keys number.
func longChapterWith(tokens int, inserts map[int]string, ending string) string {
	var text strings.Builder
	opening := "Here is a chapter of the novel I am reading.\n\n"
	text.WriteString(opening)
	units := signalTextUnits(opening)
	for paragraph := 1; units < tokens*4; paragraph++ {
		next := fmt.Sprintf("Paragraph %d. The river town kept its ledgers in a stone house by the bridge, and every spring the clerks counted the barges that came down from the hills with timber, wool and salt. "+
			"Nobody remembered who had built the house, but the children said a ferryman had carried each stone across on his back.\n\n", paragraph)
		next += inserts[paragraph]
		text.WriteString(next)
		units += signalTextUnits(next)
	}
	text.WriteString(ending)
	return text.String()
}

// denseText is a made-up turn of about size bytes that takes far fewer bytes
// a token than prose: a JSON export of barge records, or a base64
// attachment, then ending.
func denseText(kind string, size int, ending string) string {
	var text strings.Builder
	switch kind {
	case "json":
		text.WriteString(`{"barges":[`)
		for index := 0; text.Len() < size; index++ {
			fmt.Fprintf(&text, `{"id":%d,"tag":"rt-%05x","cargo":["timber","wool","salt"],"tons":%d.%02d,"checked":%t},`, index, index*7919, index%97, index%100, index%3 == 0)
		}
		text.WriteString(`{}]}`)
	case "base64":
		raw := make([]byte, 0, size*3/4+sha256.Size)
		for block := sha256.Sum256([]byte("barge ledger")); len(raw) < size*3/4; block = sha256.Sum256(block[:]) {
			raw = append(raw, block[:]...)
		}
		text.WriteString(base64.StdEncoding.EncodeToString(raw))
	}
	text.WriteString("\n\n" + ending)
	return text.String()
}

func TestRoutingViewSendsAStateWithinItsBudgetAsItIs(t *testing.T) {
	short := "Summarize this paragraph in one sentence: the town counted barges every spring."
	if view := routingView(short, 4096, 0.25); view != short {
		t.Fatalf("a short state is bounded: %q", view)
	}
	exact := longChapter(900, "Summarize the chapter above.")
	if view := routingView(exact, len(exact), 0.25); view != exact {
		t.Fatal("a state of exactly the budget is bounded")
	}
	request := modelservice.Request{State: short, Images: []string{"data:image/png;base64,AAAA"}}
	if got := boundedState(request, 1024, 0.25); got.State != short || len(got.Images) != 1 {
		t.Fatalf("a short state changed: %+v", got)
	}
	parts := modelservice.Request{Parts: map[string]string{"request": short, "history": "an earlier turn"}}
	if got := boundedState(parts, 1024, 0.25); got.Parts["request"] != short || got.Parts["history"] != "an earlier turn" {
		t.Fatalf("short parts changed: %+v", got.Parts)
	}
}

func TestRoutingViewKeepsAQuarterHeadAndThreeQuartersTail(t *testing.T) {
	text := longChapter(64_000, "Summarize the chapter above in five bullet points.")
	const budget = 32_768
	view := routingView(text, budget, 0.25)
	if len(view) > budget || len(view) < budget*9/10 {
		t.Fatalf("the view has %d bytes for a budget of %d", len(view), budget)
	}
	if !strings.HasPrefix(view, "Here is a chapter of the novel I am reading.") || !strings.HasSuffix(view, "Summarize the chapter above in five bullet points.") {
		t.Fatal("the view lost the turn's head or its closing instruction")
	}
	const closing = " tokens omitted ...]\n\n"
	marker, tailAt := strings.Index(view, "\n\n[... "), strings.Index(view, closing)
	if marker < 0 || tailAt < 0 || strings.Count(view, closing) != 1 {
		t.Fatal("the view has no single marker")
	}
	var omitted int
	if _, err := fmt.Sscanf(view[marker:], "\n\n[... %d tokens omitted ...]", &omitted); err != nil {
		t.Fatal(err)
	}
	head, tail := view[:marker], view[tailAt+len(closing):]
	if !strings.HasPrefix(text, head) || !strings.HasSuffix(text, tail) {
		t.Fatal("the head and tail are not the state's own")
	}
	if missing := signalTextUnits(text[len(head):len(text)-len(tail)]) / 4; omitted != missing {
		t.Fatalf("the marker says %d tokens omitted, %d are", omitted, missing)
	}
	if share := float64(len(head)) / float64(len(head)+len(tail)); share < 0.23 || share > 0.27 {
		t.Fatalf("the head takes %.2f of the view, want about a quarter", share)
	}
	if next := text[len(head)]; next != ' ' && next != '\n' {
		t.Fatalf("the head ends inside a word: %q", head[len(head)-20:])
	}
	if before := text[len(text)-len(tail)-1]; before != ' ' && before != '\n' {
		t.Fatalf("the tail starts inside a word: %q", tail[:20])
	}
	if again := routingView(text, budget, 0.25); again != view {
		t.Fatal("the view of one state differs between calls")
	}
}

func TestRoutingViewBoundsTextWithoutSpacesAndInvalidUTF8(t *testing.T) {
	for name, text := range map[string]string{
		"chinese":  strings.Repeat("河边小镇每年春天都清点从山上运下来的木材、羊毛和盐。", 4000),
		"emoji":    strings.Repeat("🚢🌊🪵🧂", 20_000),
		"base64":   denseText("base64", 120_000, "Decode it."),
		"invalid":  strings.Repeat("ledger \xff\xfe stone \x80\x80\x80\x80\x80 ", 9000),
		"one word": strings.Repeat("a", 200_000),
	} {
		const limit = 8192
		view := routingView(text, limit, 0.25)
		if len(view) > limit || len(view) < limit*9/10 {
			t.Fatalf("%s: %d bytes for a budget of %d", name, len(view), limit)
		}
		if name != "invalid" && !utf8.ValidString(view) {
			t.Fatalf("%s: a cut split a character", name)
		}
		windows := scanWindowsOf(text, limit, 1024)
		for index, window := range windows {
			if size := window.end - window.start; size > limit || (window.end < len(text) && size < limit*9/10) {
				t.Fatalf("%s: window %d has %d bytes", name, index, size)
			}
			if name != "invalid" && !utf8.ValidString(text[window.start:window.end]) {
				t.Fatalf("%s: window %d splits a character", name, index)
			}
		}
	}
}

func TestBoundedStateSharesTheBudgetAmongParts(t *testing.T) {
	long := longChapter(40_000, "What does the ferryman story say about the town?")
	request := modelservice.Request{Parts: map[string]string{
		"request":        long,
		"previous_turns": "user: hello\nassistant: Hello! How can I help?",
		"history":        longChapter(20_000, "Thanks."),
	}}
	const budget = 32_768
	got := boundedState(request, budget, 0.25)
	if got.Parts["previous_turns"] != request.Parts["previous_turns"] {
		t.Fatal("a part within its share was cut")
	}
	if total := stateBytes(got); total > budget {
		t.Fatalf("the parts have %d bytes, over the budget of %d", total, budget)
	}
	if !strings.HasSuffix(got.Parts["request"], "What does the ferryman story say about the town?") || !strings.HasSuffix(got.Parts["history"], "Thanks.") {
		t.Fatal("a bounded part lost its tail")
	}
	if request.Parts["request"] != long {
		t.Fatal("bounding changed the caller's parts")
	}
	if shares := fairShares([]int{10, 500, 2000}, 1000); shares[0] != 10 || shares[1] != 495 || shares[2] != 495 {
		t.Fatalf("fair shares %v", shares)
	}
}

func TestScanWindowsCoverTheWholeTextWithOverlap(t *testing.T) {
	text := longChapter(100_000, "The end.")
	const size, overlap = 32_768, 2048
	windows := scanWindowsOf(text, size, overlap)
	if len(windows) < 12 || windows[0].start != 0 || windows[len(windows)-1].end != len(text) {
		t.Fatalf("%d windows from %d to %d of %d bytes", len(windows), windows[0].start, windows[len(windows)-1].end, len(text))
	}
	for index, window := range windows {
		if bytes := window.end - window.start; bytes > size {
			t.Fatalf("window %d has %d bytes, over %d", index, bytes, size)
		}
		if window.runeStart != utf8.RuneCountInString(text[:window.start]) {
			t.Fatalf("window %d starts at code point %d, not %d", index, window.runeStart, utf8.RuneCountInString(text[:window.start]))
		}
		if index == 0 {
			continue
		}
		previous := windows[index-1]
		if window.start > previous.end || window.start <= previous.start {
			t.Fatalf("window %d [%d,%d) leaves a gap after [%d,%d)", index, window.start, window.end, previous.start, previous.end)
		}
		if shared := previous.end - window.start; shared < overlap-routingViewSnap || shared > overlap {
			t.Fatalf("windows %d and %d share %d bytes, want about %d", index-1, index, shared, overlap)
		}
	}
	if order := scanOrder(5); fmt.Sprint(order) != "[4 0 1 2 3]" {
		t.Fatalf("reading order %v, want the tail, the head, then the middle", order)
	}
	if order := scanOrder(maxScanWindows + 36); len(order) != maxScanWindows || order[0] != maxScanWindows+35 || order[1] != 0 {
		t.Fatalf("reading order of %d windows %v", maxScanWindows+36, order)
	}
}

func TestScanAnswersTakeTheMostAlarmingWindowAndJoinSpans(t *testing.T) {
	jailbreak := config.SystemOneRule{Name: "attack", Labels: []string{"jailbreak"}, Question: config.SystemOneQuestion{Type: config.SystemOneQuestionChoice, Choices: []config.SystemOneOption{{Key: "benign"}, {Key: "jailbreak"}}}}
	read := []windowAnswer{
		{answer: modelservice.Answer{Type: "choice", Choice: "benign", Probabilities: map[string]float64{"benign": 0.97, "jailbreak": 0.03}}},
		{answer: modelservice.Answer{Type: "choice", Choice: "jailbreak", Probabilities: map[string]float64{"benign": 0.2, "jailbreak": 0.8}}, window: 1},
		{answer: modelservice.Answer{Type: "choice", Choice: "benign", Probabilities: map[string]float64{"benign": 0.6, "jailbreak": 0.4}}, window: 2},
	}
	if got := scanAnswer(jailbreak, read, nil); got.Choice != "jailbreak" || got.Probabilities["jailbreak"] != 0.8 {
		t.Fatalf("the most alarming window decides: %+v", got)
	}
	categories := config.SystemOneRule{Name: "harm", Question: config.SystemOneQuestion{Type: config.SystemOneQuestionSet, Labels: []config.SystemOneOption{{Key: "violence"}, {Key: "fraud"}}}}
	set := scanAnswer(categories, []windowAnswer{
		{answer: modelservice.Answer{Type: "set", Probabilities: map[string]float64{"violence": 0.1, "fraud": 0.7}, Selected: []string{"fraud"}, InputCoverage: "complete"}},
		{answer: modelservice.Answer{Type: "set", Probabilities: map[string]float64{"violence": 0.9, "fraud": 0.2}, Selected: []string{"violence"}, InputCoverage: "complete"}, window: 1},
	}, nil)
	if set.Probabilities["violence"] != 0.9 || set.Probabilities["fraud"] != 0.7 || fmt.Sprint(set.Selected) != "[violence fraud]" || set.InputCoverage != "complete" {
		t.Fatalf("a set takes each label's highest probability: %+v", set)
	}

	// Two windows overlap on "Ada Lovelace"; the second also finds an email.
	text := "Notes: Ada Lovelace met the clerks. Write to ada@example.com today."
	scan := &guardScan{text: text, windows: []scanWindow{{start: 0, end: 40, runeStart: 0}, {start: 7, end: len(text), runeStart: 7}}}
	pii := config.SystemOneRule{Name: "contact", Question: config.SystemOneQuestion{Type: config.SystemOneQuestionSpan}}
	spans := scanAnswer(pii, []windowAnswer{
		{answer: modelservice.Answer{Type: "span", Noul: 0.9, Spans: []modelservice.Span{{Label: "PERSON", Start: 7, End: 15, Text: "Ada Love", Probability: 0.7}}}},
		{answer: modelservice.Answer{Type: "span", Noul: 0.95, Spans: []modelservice.Span{
			{Label: "PERSON", Start: 0, End: 12, Text: "Ada Lovelace", Probability: 0.9},
			{Label: "EMAIL_ADDRESS", Start: 38, End: 53, Text: "ada@example.com", Probability: 0.99},
		}}, window: 1},
	}, scan)
	if len(spans.Spans) != 2 || spans.Noul != 0.95 {
		t.Fatalf("spans %+v", spans.Spans)
	}
	for _, span := range spans.Spans {
		if got := string([]rune(text)[span.Start:span.End]); got != span.Text {
			t.Fatalf("span %+v points at %q", span, got)
		}
	}
	if person := spans.Spans[0]; person.Text != "Ada Lovelace" || person.Probability != 0.9 {
		t.Fatalf("the overlapping person spans join: %+v", person)
	}
}

func TestViewAndWindowBudgetsComeFromTheDeploymentAndTheCard(t *testing.T) {
	half := 0.5
	window := 2048
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{
		"local":     {Provider: config.ModelRuntimeProvider, Endpoint: "http://127.0.0.1:1"},
		"bounded":   {Provider: config.ModelRuntimeProvider, Endpoint: "http://127.0.0.1:1", RoutingView: &config.DeploymentRoutingView{MaxTokens: 2048, HeadShare: &half}, ScanWindow: &config.DeploymentScanWindow{Tokens: window}},
		"small":     {Provider: config.ModelRuntimeProvider, Endpoint: "http://127.0.0.1:1"},
		"hosted-d3": {Provider: config.SystemOneProvider, Endpoint: "https://decisions.example.com/v1/systemone", Model: "d3"},
	}
	classifier := &Classifier{Config: cfg, decisionCards: map[string]modelservice.ModelCard{
		"local":   {MaxInputTokens: 262_144, ValidatedInputTokens: 65_536},
		"bounded": {MaxInputTokens: 262_144, ValidatedInputTokens: 32_768},
		"small":   {MaxInputTokens: 4096},
	}}
	remote := newFakeSystemOne()
	classifier.SetSystemOneDecider(remote)
	// Bytes: four a configured token, at most a byte a token of the card's
	// cap less the reserve.
	for deployment, want := range map[string][4]float64{
		"local":     {32_768, 0.25, 32_768, 2048},
		"bounded":   {8192, 0.5, 8192, 2048},
		"small":     {4096 - routingViewReserveTokens, 0.25, 4096 - routingViewReserveTokens, (4096-routingViewReserveTokens)/2 - 1},
		"hosted-d3": {16_384, 0.25, 16_384, 1024},
		"unknown":   {32_768, 0.25, 32_768, 2048},
	} {
		budget, share := classifier.routingViewBudget(deployment)
		size, overlap, _ := classifier.scanWindowBudget(deployment)
		if got := [4]float64{float64(budget), share, float64(size), float64(overlap)}; got != want {
			t.Fatalf("%s: view %d bytes (head %.2f), windows %d overlapping %d; want %v", deployment, budget, share, size, overlap, want)
		}
	}
	remote.card.MaxInputTokens, remote.card.ValidatedInputTokens = 262_144, 2048
	if budget, _ := classifier.routingViewBudget("hosted-d3"); budget != 2048-routingViewReserveTokens {
		t.Fatalf("a listed validated length bounds a remote view: %d", budget)
	}
}

// An operator's budget at the card's cap still leaves every view and window
// within the model on dense text, where the Router's four bytes a token
// counts far too few: each holds no more bytes, and so no more tokens of a
// byte-level tokenizer, than the cap less the reserve.
func TestViewsAndWindowsOfDenseTextNeverOverrunTheModel(t *testing.T) {
	const validated = 32_768
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{
		"default": {Provider: config.ModelRuntimeProvider, Endpoint: "http://127.0.0.1:1"},
		"at-cap": {
			Provider: config.ModelRuntimeProvider, Endpoint: "http://127.0.0.1:1",
			RoutingView: &config.DeploymentRoutingView{MaxTokens: validated}, ScanWindow: &config.DeploymentScanWindow{Tokens: validated},
		},
	}
	card := modelservice.ModelCard{MaxInputTokens: 262_144, ValidatedInputTokens: validated}
	classifier := &Classifier{Config: cfg, decisionCards: map[string]modelservice.ModelCard{"default": card, "at-cap": card}}
	for _, kind := range []string{"json", "base64"} {
		text := denseText(kind, 200_000, "Summarize the attachment above.")
		for _, deployment := range []string{"default", "at-cap"} {
			budget, share := classifier.routingViewBudget(deployment)
			view := boundedState(modelservice.Request{State: text}, budget, share).State
			if len(view)+routingViewReserveTokens > validated || !strings.HasSuffix(view, "Summarize the attachment above.") {
				t.Fatalf("%s on %s: a %d-byte view, over the cap of %d less the reserve", kind, deployment, len(view), validated)
			}
			size, overlap, parallel := classifier.scanWindowBudget(deployment)
			scan := newGuardScan(deployment, modelservice.Request{State: text}, size, overlap, parallel)
			if scan == nil || len(scan.windows) < 6 || scan.windows[len(scan.windows)-1].end != len(text) {
				t.Fatalf("%s on %s: the scan does not cover the turn", kind, deployment)
			}
			for index, window := range scan.windows {
				if bytes := window.end - window.start; bytes+routingViewReserveTokens > validated {
					t.Fatalf("%s on %s: window %d has %d bytes, over the cap less the reserve", kind, deployment, index, bytes)
				}
			}
		}
	}
}
