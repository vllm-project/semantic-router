package classification

import (
	"unicode"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A model call made for a routing signal takes one of three input treatments,
// chosen by what a missed part of the text would cost.
//
//   - Sample. A signal that labels the request as a whole, such as domain,
//     language, complexity or modality, may read a head, middle and tail view,
//     which carries the request's purpose and bounds the forward pass.
//     textForRoutingSignal does this for boundedSemanticSignalTypes.
//   - Scan. A signal that asks whether something appears anywhere (PII, an
//     attack, an unsupported claim) must read all of the text, because a
//     dropped part is a silent miss. Use securitySignalChunkSpans with a budget
//     below, or a native token window defined by the model's tokenizer.
//   - Whole text. When the configuration declares a budget above 512 tokens
//     for the model (hasLongContextClassifier), pass the text unchanged. The
//     declared budget then governs, and a deployment's overflow policy, reject
//     by default, decides what happens past it.
//
// Never hand a long text to a model that truncates it to its own limit. That
// answers on a prefix while reporting a verdict for the whole input, which is
// how #3204, #3333 and #3364 happened.
//
// A caller that reports positions also owes offsets into the original text.
// Take them from securitySignalChunkSpans by adding StartByte to the offset
// inside the chunk. The de-duplicated chunk helpers merge identical chunks
// found at different offsets, and searching the text for a chunk maps repeated
// text to its first occurrence.

// The request's full text still reaches exact-match, structure, and context
// signals. Budgets are quarter-token units from signalRuneUnits, set to stay at
// or above the mmBERT token count (TestSignalUnitsUpperBoundMeasuredTokenCounts),
// against a 512-token forward.
//
// semanticSignalUnitLimit is 440 estimated tokens for the three samples, which
// leaves room for the two omission markers (39 estimated tokens) and the
// special tokens. PII scans 128-token chunks because local token
// classification loses entity confidence in long windows. That leaves room for
// text about four times denser than the estimate before a forward truncates.
// Jailbreak scans 384-token chunks because detection gains from broader
// context, which leaves room for about a third more.
const (
	semanticSignalUnitLimit     = 440 * 4
	piiSignalChunkBudget        = 128 * 4
	piiSignalChunkOverlapRunes  = 64
	jailbreakSignalChunkBudget  = 384 * 4
	jailbreakSignalOverlapRunes = 64
)

const signalWindowOmissionMarker = "\n... [content sampled for routing signal evaluation] ...\n"

var boundedSemanticSignalTypes = map[string]struct{}{
	config.SignalTypeEmbedding:    {},
	config.SignalTypeDomain:       {},
	config.SignalTypeFactCheck:    {},
	config.SignalTypeUserFeedback: {},
	config.SignalTypeReask:        {},
	config.SignalTypePreference:   {},
	config.SignalTypeLanguage:     {},
	config.SignalTypeComplexity:   {},
	config.SignalTypeModality:     {},
	config.SignalTypeKB:           {},
	config.SignalTypeEvent:        {},
}

func textForRoutingSignal(signalType, text string) string {
	if _, bounded := boundedSemanticSignalTypes[signalType]; !bounded {
		return text
	}
	return representativeSignalText(text, semanticSignalUnitLimit)
}

func representativeSignalText(text string, maxUnits int) string {
	runes := []rune(text)
	prefix := signalUnitPrefix(runes)
	if maxUnits <= 0 || prefix[len(runes)] <= maxUnits {
		return text
	}
	part := maxUnits / 3
	headEnd := signalUnitsForward(prefix, 0, part)
	tailStart := signalUnitsBackward(prefix, len(runes), part)
	center := len(runes) / 2
	middleStart := max(signalUnitsBackward(prefix, center, part/2), headEnd)
	middleEnd := min(signalUnitsForward(prefix, center, part-part/2), tailStart)
	return string(runes[:headEnd]) +
		signalWindowOmissionMarker +
		string(runes[middleStart:middleEnd]) +
		signalWindowOmissionMarker +
		string(runes[tailStart:])
}

func piiSignalChunks(text string) []string {
	return uniqueSignalChunks(
		securitySignalChunks(text, piiSignalChunkBudget, piiSignalChunkOverlapRunes),
	)
}

// piiSignalChunkSpans is piiSignalChunks with offsets, for callers that report
// entity positions. It does not de-duplicate: two identical chunks are at
// different offsets, and both of those positions are real.
func piiSignalChunkSpans(text string) []signalChunkSpan {
	return securitySignalChunkSpans(text, piiSignalChunkBudget, piiSignalChunkOverlapRunes)
}

func jailbreakSignalChunks(text string) []string {
	return uniqueSignalChunks(
		securitySignalChunks(text, jailbreakSignalChunkBudget, jailbreakSignalOverlapRunes),
	)
}

// signalChunkSpan is one chunk together with its byte offset in the text it was
// cut from. Callers that report positions back to a client need the offset to
// map a chunk-relative entity onto the original text.
type signalChunkSpan struct {
	Text      string
	StartByte int
}

// securitySignalChunks scans the entire input in bounded, overlapping pieces.
// Unlike semantic intent signals, security detection cannot safely discard the
// middle of a long prompt.
func securitySignalChunks(text string, budget, overlapRunes int) []string {
	spans := securitySignalChunkSpans(text, budget, overlapRunes)
	if spans == nil {
		return nil
	}
	chunks := make([]string, len(spans))
	for i, span := range spans {
		chunks[i] = span.Text
	}
	return chunks
}

// securitySignalChunkSpans is securitySignalChunks with each chunk's byte
// offset in text. An entity found at offset e in span s sits at
// s.StartByte + e in the original text.
//
// Each chunk starts overlapRunes before the previous one ends, so any span of
// at most overlapRunes+1 runes lies inside one chunk, as long as every chunk is
// at least that long. A longer span can straddle a boundary and reach the model
// only in parts. At the PII budget a long unbroken run of digits or base64 can
// produce chunks shorter than the overlap, and the bound does not hold there.
func securitySignalChunkSpans(text string, budget, overlapRunes int) []signalChunkSpan {
	runes := []rune(text)
	if len(runes) == 0 {
		return nil
	}
	prefix := signalUnitPrefix(runes)
	if prefix[len(runes)] <= budget {
		return []signalChunkSpan{{Text: text, StartByte: 0}}
	}

	spans := make([]signalChunkSpan, 0, (len(runes)/1024)+1)
	// Chunk starts only ever move forward, so one cursor converts them to byte
	// offsets without a per-rune index.
	startByte, counted := 0, 0
	for start := 0; start < len(runes); {
		for ; counted < start; counted++ {
			startByte += utf8.RuneLen(runes[counted])
		}
		end := securitySignalChunkEnd(runes, prefix, start, budget)
		spans = append(spans, signalChunkSpan{
			Text:      string(runes[start:end]),
			StartByte: startByte,
		})
		if end == len(runes) {
			break
		}
		start = max(start+1, end-overlapRunes)
	}
	return spans
}

func securitySignalChunkEnd(runes []rune, prefix []int, start, budget int) int {
	end := max(signalUnitsForward(prefix, start, budget-1), start+1)
	if end >= len(runes) || unicode.IsSpace(runes[end]) {
		return end
	}
	for i := end; i > start+1 && i > end-64; i-- {
		if unicode.IsSpace(runes[i-1]) {
			return i
		}
	}
	return end
}

func securitySignalChunkUnits(runes []rune) int {
	return signalUnitPrefix(runes)[len(runes)]
}

func signalUnitPrefix(runes []rune) []int {
	prefix := make([]int, len(runes)+1)
	inWord, wordHasDigit := false, false
	for i, r := range runes {
		var units int
		units, inWord, wordHasDigit = signalRuneUnits(r, inWord, wordHasDigit)
		prefix[i+1] = prefix[i] + units
	}
	return prefix
}

func signalRuneUnits(r rune, inWord, wordHasDigit bool) (int, bool, bool) {
	switch {
	case unicode.IsSpace(r):
		return 0, false, false
	case isCJK(r):
		return 4, false, false
	case unicode.IsDigit(r):
		return 5, true, true
	case unicode.IsLetter(r):
		if inWord && wordHasDigit {
			return 4, true, true
		}
		units := 1
		if !unicode.Is(unicode.Latin, r) {
			units = 2
		}
		if !inWord {
			units++
		}
		return units, true, false
	case unicode.IsSymbol(r):
		return 5, false, false
	default:
		return 4, false, false
	}
}

func signalUnitsForward(prefix []int, start, budget int) int {
	end := start
	for end < len(prefix)-1 && prefix[end+1]-prefix[start] <= budget {
		end++
	}
	return end
}

func signalUnitsBackward(prefix []int, end, budget int) int {
	start := end
	for start > 0 && prefix[end]-prefix[start-1] <= budget {
		start--
	}
	return start
}

// uniqueSignalChunks removes exact duplicate inference work while preserving
// scan order. Generated logs, repeated quoted content, and padded eval inputs
// can contain hundreds of identical security windows; classifying the same
// bytes again cannot improve recall.
func uniqueSignalChunks(chunks []string) []string {
	if len(chunks) < 2 {
		return chunks
	}
	seen := make(map[string]struct{}, len(chunks))
	unique := make([]string, 0, len(chunks))
	for _, chunk := range chunks {
		if _, duplicate := seen[chunk]; duplicate {
			continue
		}
		seen[chunk] = struct{}{}
		unique = append(unique, chunk)
	}
	return unique
}

// Explicit native budgets above 512 opt the trained task into full-context
// inference. Other signals keep their own established input policies.
func (c *Classifier) hasLongContextClassifier(signalType string) bool {
	if c == nil || c.Config == nil {
		return false
	}
	consumer := classifierInputConsumer(signalType)
	if binding, exists := c.Config.ModelBindings[consumer]; exists {
		deployment, exists := c.Config.ModelDeployments[binding.Deployment]
		return exists && deployment.Provider != "http" && deployment.Input.MaxTokens > 512
	}
	switch signalType {
	case config.SignalTypeEmbedding:
		return c.Config.EmbeddingConfig.FullContext
	case config.SignalTypeComplexity:
		// Local Complexity uses the semantic embedding provider's input policy.
		// An independent remote scorer retains its existing bounded input.
		return c.Config.ComplexityModel.Backend == nil && c.Config.EmbeddingConfig.FullContext
	case config.SignalTypeDomain:
		return c.Config.CategoryModel.Backend == nil && c.Config.CategoryModel.MaxSequenceLength > 512
	case config.SignalTypeFactCheck:
		return c.Config.HallucinationMitigation.FactCheckModel.MaxSequenceLength > 512
	case config.SignalTypeUserFeedback:
		return c.Config.FeedbackDetector.MaxSequenceLength > 512
	case config.SignalTypePII:
		return c.Config.PIIModel.Backend == nil && c.Config.PIIModel.MaxSequenceLength > 512
	case config.SignalTypeJailbreak:
		return c.Config.PromptGuard.Backend == nil && c.Config.PromptGuard.MaxSequenceLength > 512
	case config.SignalTypeModality:
		return c.Config.ModalityDetector.Classifier != nil && c.Config.ModalityDetector.Classifier.MaxSequenceLength > 512
	}
	return false
}

// signalReadsWholeText reports whether a signal's prepared model asks a Vela
// 2.0 model the signal's question. That model reads a whole text up to its own
// input budget, so the signal asks about the request as it is, in the same
// call as the request's other questions about it.
// decisionModelQuestionText is the text a decision question that names no
// deployment reads: the decision model reads the request as it came.
const decisionModelQuestionText = "decision_model"

func (c *Classifier) signalReadsWholeText(signalType string) bool {
	if c == nil {
		return false
	}
	var consumer interface{}
	switch signalType {
	case decisionModelQuestionText:
		return true
	case config.SignalTypeDomain:
		consumer = c.categoryInference
	case config.SignalTypeJailbreak:
		consumer = c.jailbreakInference
	case config.SignalTypeFactCheck:
		if c.factCheckClassifier != nil {
			consumer = c.factCheckClassifier.backend
		}
	case config.SignalTypeUserFeedback:
		if c.feedbackDetector != nil {
			consumer = c.feedbackDetector.backend
		}
	case config.SignalTypeModality:
		consumer = c.modalityInference
	case config.SignalTypePII:
		consumer = c.piiInference
	case config.SignalTypeSafety:
		for _, detector := range c.safetyClassifiers {
			if detector == nil {
				continue
			}
			if reader, ok := detector.binary.(interface{ readsWholeText() bool }); ok && reader.readsWholeText() {
				return true
			}
		}
	}
	reader, ok := consumer.(interface{ readsWholeText() bool })
	return ok && reader.readsWholeText()
}

func (c *Classifier) piiInputSpans(text string) []signalChunkSpan {
	if c.piiReadsWholeText() && text != "" {
		return []signalChunkSpan{{Text: text}}
	}
	return piiSignalChunkSpans(text)
}

func (c *Classifier) piiInputs(text string) []string {
	if c.piiReadsWholeText() {
		return []string{text}
	}
	return piiSignalChunks(text)
}

// piiReadsWholeText reports whether the PII model reads a whole text: through
// token windows, a long-context budget, or a decision model's ready-made PII
// question, which then shares the call of the deployment's other questions.
func (c *Classifier) piiReadsWholeText() bool {
	if c == nil || c.Config == nil {
		return false
	}
	if reader, ok := c.piiInference.(interface{ readsWholeText() bool }); ok && reader.readsWholeText() {
		return true
	}
	return c.Config.PIIModel.Window != nil || c.hasLongContextClassifier(config.SignalTypePII)
}

func (c *Classifier) jailbreakInputs(text string) []string {
	fullContext := c.hasLongContextClassifier(config.SignalTypeJailbreak)
	if c != nil && c.models != nil && c.models.jailbreakContrastiveFullContext != nil {
		fullContext = *c.models.jailbreakContrastiveFullContext
	}
	if fullContext {
		return []string{text}
	}
	return jailbreakSignalChunks(text)
}

// Native token windows own tokenization and the total input limit. Decoded
// text chunks would change both windows and overflow checks. Contrastive rules
// continue using their existing text-window policy.
func (c *Classifier) jailbreakModelInputs(text string) []string {
	if text == "" {
		return nil
	}
	if (c != nil && c.Config != nil && c.Config.PromptGuard.Window != nil) || c.signalReadsWholeText(config.SignalTypeJailbreak) {
		return []string{text}
	}
	return c.jailbreakInputs(text)
}

func classifierInputConsumer(signalType string) string {
	switch signalType {
	case config.SignalTypeDomain:
		return "domain_classifier"
	case config.SignalTypeFactCheck:
		return "fact_check_classifier"
	case config.SignalTypeUserFeedback:
		return "feedback_detector"
	case config.SignalTypePII:
		return "pii_classifier"
	case config.SignalTypeJailbreak:
		return "prompt_guard"
	case config.SignalTypeModality:
		return "modality_detector"
	}
	return ""
}
