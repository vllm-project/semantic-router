package memory

import (
	"slices"
	"sort"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Similarity alone can't tell a correction from a second, still-true fact about
// the same subject (https://github.com/vllm-project/semantic-router/issues/4161),
// so a turn supersedes an older one only when one of its sentences says the user
// changed something ("I moved", "we've switched", "I'm no longer", "I don't ...
// anymore") and the clause reporting the change shares a word pair with the
// older statement. Questions and changes made by someone else, negated,
// hypothetical or still planned don't count.
var (
	changeVerbs         = wordSet("moved relocated changed switched raised lowered increased decreased reduced")
	firstPersonSubjects = wordSet("i we i've we've i'm we're")
	changeModifiers     = wordSet("just recently finally already also actually have am are")
	hypotheticalMarkers = wordSet("if wish unless whether had have has would could should might may")
	negations           = wordSet("not don't doesn't can't never")
	elidedSubjectVerbs  = wordSet("work live drive study teach own rent use run manage lead stay")
	auxiliaries         = wordSet("is are was were be been has have had will would can could should may might must do does did")
	functionWords       = wordSet(`a an the this that these those some any all each every no
		and or but so if then than because while
		of to in on at for from with by as about into onto near over under after before
		since until around through during without within
		i me my mine myself you your yours we us our ours he him his she her hers it its
		they them their theirs who whom whose which what where when why how
		am is are was were be been being do does did done have has had having
		will would shall should can could may might must get got
		not yes just now still also very really too there here more most much many
		other such only own same again ever never already yet longer anymore`)
)

type retrievedTurn struct {
	text        string
	statementID int
	clauses     [][]string
	correction  []wordPair
	// Only a one-sentence, single-fact statement is hidden, since clauses such
	// as "and work as a nurse" can hold facts the correction doesn't mention.
	// A correction qualifies only when all its clauses belong to the change.
	hideable bool
}

type turnRef struct {
	memory int
	turn   int
}

type wordPair [2]string

// supersession records, for each turn of each retrieved memory, the newer
// retrieved turns that correct it.
type supersession struct {
	index      map[*RetrieveResult]int
	createdAt  []time.Time
	turns      [][]retrievedTurn
	correctors [][][]turnRef
}

// newSupersession returns nil when no retrieved turn corrects another.
func newSupersession(memories []*RetrieveResult) *supersession {
	statementIDs := make(map[string]int)
	turns := make([][]retrievedTurn, len(memories))
	correctionsByPair := make(map[wordPair][]turnRef)
	for i, m := range memories {
		turns[i] = parseRetrievedTurns(m.Memory.Content, statementIDs)
		for ti, turn := range turns[i] {
			ref := turnRef{memory: i, turn: ti}
			for _, pair := range turn.correction {
				refs := correctionsByPair[pair]
				if len(refs) == 0 || refs[len(refs)-1] != ref {
					correctionsByPair[pair] = append(refs, ref)
				}
			}
		}
	}
	if len(correctionsByPair) == 0 {
		return nil
	}

	s := &supersession{
		index:      make(map[*RetrieveResult]int, len(memories)),
		createdAt:  make([]time.Time, len(memories)),
		turns:      turns,
		correctors: make([][][]turnRef, len(memories)),
	}
	for i, m := range memories {
		s.index[m] = i
		s.createdAt[i] = m.Memory.CreatedAt
		s.correctors[i] = make([][]turnRef, len(turns[i]))
		for ti := range turns[i] {
			s.correctors[i][ti] = correctorsOf(memories, turns, correctionsByPair, turnRef{memory: i, turn: ti})
		}
	}
	return s
}

// corrects reports whether newer corrects any turn of older.
func (s *supersession) corrects(newer *RetrieveResult, older *RetrieveResult) bool {
	if s == nil {
		return false
	}
	newerIndex, olderIndex := s.index[newer], s.index[older]
	for _, correctors := range s.correctors[olderIndex] {
		if slices.ContainsFunc(correctors, func(c turnRef) bool { return c.memory == newerIndex }) {
			return true
		}
	}
	return false
}

// hideCorrected drops each turn that a surviving injected turn corrects, so a
// turn disappears only when its correction reaches the prompt too. A session
// chunk loses only those turns, and stored records are never modified.
func (s *supersession) hideCorrected(injected []*RetrieveResult) ([]*RetrieveResult, bool) {
	if s == nil {
		return injected, false
	}
	inPrompt := make(map[int]bool, len(injected))
	for _, m := range injected {
		inPrompt[s.index[m]] = true
	}
	var refs []turnRef
	dependents := make(map[turnRef][]turnRef)
	for _, m := range injected {
		i := s.index[m]
		for ti := range s.turns[i] {
			ref := turnRef{memory: i, turn: ti}
			refs = append(refs, ref)
			for _, c := range s.correctors[i][ti] {
				if inPrompt[c.memory] {
					dependents[c] = append(dependents[c], ref)
				}
			}
		}
	}
	// A corrector is always newer than the turn it corrects, so deciding the
	// newest turns first tells each turn which of its correctors survive.
	sort.Slice(refs, func(a, b int) bool {
		ta, tb := s.createdAt[refs[a].memory], s.createdAt[refs[b].memory]
		if !ta.Equal(tb) {
			return ta.After(tb)
		}
		if refs[a].memory != refs[b].memory {
			return refs[a].memory < refs[b].memory
		}
		return refs[a].turn > refs[b].turn
	})
	hiddenTurns := make(map[turnRef]bool)
	for _, ref := range refs {
		if !s.turns[ref.memory][ref.turn].hideable {
			continue
		}
		// Hiding a correction also drops what it said about the turns it
		// corrects, so its own corrector must correct those turns directly.
		hiddenTurns[ref] = slices.ContainsFunc(s.correctors[ref.memory][ref.turn], func(c turnRef) bool {
			return inPrompt[c.memory] && !hiddenTurns[c] && s.correctsAll(c, dependents[ref])
		})
	}

	hidden := false
	kept := make([]*RetrieveResult, 0, len(injected))
	for _, m := range injected {
		i := s.index[m]
		live := make([]string, 0, len(s.turns[i]))
		for ti, turn := range s.turns[i] {
			if hiddenTurns[turnRef{memory: i, turn: ti}] {
				logging.Debugf("ReflectionGate: turn %d of memory id=%s superseded", ti, m.Memory.ID)
				hidden = true
				continue
			}
			live = append(live, turn.text)
		}
		switch {
		case len(live) == len(s.turns[i]):
			kept = append(kept, m)
		case len(live) > 0:
			trimmed := *m.Memory
			trimmed.Content = strings.Join(live, sessionTurnSeparator)
			kept = append(kept, &RetrieveResult{Memory: &trimmed, Score: m.Score})
		}
	}
	return kept, hidden
}

func (s *supersession) correctsAll(corrector turnRef, turns []turnRef) bool {
	for _, t := range turns {
		if !slices.Contains(s.correctors[t.memory][t.turn], corrector) {
			return false
		}
	}
	return true
}

// correctorsOf returns the newer turns that correct the given turn.
func correctorsOf(
	memories []*RetrieveResult,
	turns [][]retrievedTurn,
	correctionsByPair map[wordPair][]turnRef,
	older turnRef,
) []turnRef {
	var found []turnRef
	seenPairs := make(map[wordPair]bool)
	checked := make(map[turnRef]bool)
	for _, clause := range turns[older.memory][older.turn].clauses {
		for _, pair := range anchorPairs(clause) {
			if seenPairs[pair] {
				continue
			}
			seenPairs[pair] = true
			for _, newer := range correctionsByPair[pair] {
				if !checked[newer] && supersedes(memories, turns, newer, older) {
					found = append(found, newer)
				}
				checked[newer] = true
			}
		}
	}
	return found
}

func supersedes(memories []*RetrieveResult, turns [][]retrievedTurn, newer turnRef, older turnRef) bool {
	// A session chunk repeats its turns, so an identical statement is a copy, not a correction.
	if turns[newer.memory][newer.turn].statementID == turns[older.memory][older.turn].statementID {
		return false
	}
	if newer.memory == older.memory {
		return newer.turn > older.turn
	}
	// A session chunk is dated when its last turn is stored, so the earlier
	// turns it repeats have no known time and can't outrank another memory.
	isLastTurn := newer.turn == len(turns[newer.memory])-1
	return isLastTurn && createdAfter(memories[newer.memory].Memory, memories[older.memory].Memory)
}

// Memories without a creation time can't be ordered, so they neither supersede
// nor get superseded by another memory.
func createdAfter(a *Memory, b *Memory) bool {
	return !a.CreatedAt.IsZero() && !b.CreatedAt.IsZero() && a.CreatedAt.After(b.CreatedAt)
}

func parseRetrievedTurns(content string, statementIDs map[string]int) []retrievedTurn {
	segments := splitSessionTurns(content)
	turns := make([]retrievedTurn, 0, len(segments))
	for _, segment := range segments {
		turn := retrievedTurn{text: segment}
		var statement []string
		sentences := statementSentences(userStatement(segment))
		otherFacts := 0
		for _, sentence := range sentences {
			clauses, joined := sentenceClauses(sentence.text)
			inCorrection := make([]bool, len(clauses))
			if !sentence.question {
				var pairs []wordPair
				pairs, inCorrection = correctionPairs(clauses)
				turn.correction = append(turn.correction, pairs...)
			}
			for ci, clause := range clauses {
				statement = append(statement, clause...)
				if (ci == 0 || joined[ci] || firstPersonSubjects[clause[0]]) && !inCorrection[ci] {
					otherFacts++
				}
			}
			turn.clauses = append(turn.clauses, clauses...)
		}
		allowedOtherFacts := 1
		if len(turn.correction) > 0 {
			allowedOtherFacts = 0
		}
		turn.hideable = len(sentences) == 1 && otherFacts <= allowedOtherFacts
		key := strings.Join(statement, " ")
		id, seen := statementIDs[key]
		if !seen {
			id = len(statementIDs)
			statementIDs[key] = id
		}
		turn.statementID = id
		turns = append(turns, turn)
	}
	return turns
}

// correctionPairs returns the word pairs of a clause that reports the user's
// own change and of later clauses where the user says what is true now, as in
// "I moved to Denver, and I live there now", and marks those clauses. Other
// clauses, such as "and I still work as a nurse" or "and my dog is now three",
// can't anchor the change.
func correctionPairs(clauses [][]string) ([]wordPair, []bool) {
	var pairs []wordPair
	inCorrection := make([]bool, len(clauses))
	changed := false
	for ci, clause := range clauses {
		if reportsOwnChange(clause) || (changed && slices.Contains(clause, "now") && describesUser(clause)) {
			changed = true
			inCorrection[ci] = true
			pairs = append(pairs, anchorPairs(clause)...)
		}
	}
	return pairs, inCorrection
}

// describesUser accepts a clause whose subject is the user, either named or
// left out before a verb such as "work" in "and now work as a paramedic".
func describesUser(clause []string) bool {
	if firstPersonSubjects[clause[0]] {
		return true
	}
	if clause[0] != "now" || len(clause) < 2 {
		return false
	}
	if firstPersonSubjects[clause[1]] {
		return true
	}
	// "now work is closer" uses "work" as a noun.
	return elidedSubjectVerbs[clause[1]] && (len(clause) == 2 || !auxiliaries[clause[2]])
}

// A turn boundary includes the next turn's question prefix, so a "---" line
// inside a message, as in multi-document YAML, doesn't split that turn.
func splitSessionTurns(content string) []string {
	segments := strings.Split(content, sessionTurnSeparator+turnQuestionPrefix)
	for i := 1; i < len(segments); i++ {
		segments[i] = turnQuestionPrefix + segments[i]
	}
	return segments
}

// userStatement returns the user's half of a stored turn, or the whole text
// for memories written in another format.
func userStatement(turn string) string {
	if strings.HasPrefix(turn, turnAnswerPrefix) {
		return ""
	}
	question, found := strings.CutPrefix(turn, turnQuestionPrefix)
	if !found {
		return turn
	}
	question, _, _ = strings.Cut(question, "\n"+turnAnswerPrefix)
	return question
}

type statementSentence struct {
	text     string
	question bool
}

func statementSentences(statement string) []statementSentence {
	var sentences []statementSentence
	start := 0
	for i, r := range statement {
		if r != '.' && r != '!' && r != '?' && r != ';' && r != '\n' {
			continue
		}
		if text := statement[start:i]; strings.TrimSpace(text) != "" {
			sentences = append(sentences, statementSentence{text: text, question: r == '?'})
		}
		start = i + utf8.RuneLen(r)
	}
	if text := statement[start:]; strings.TrimSpace(text) != "" {
		sentences = append(sentences, statementSentence{text: text})
	}
	return sentences
}

// sentenceClauses splits a sentence at commas and at "and" or "but", and
// reports which clauses a conjunction starts.
func sentenceClauses(sentence string) ([][]string, []bool) {
	var clauses [][]string
	var joined []bool
	for _, part := range strings.Split(sentence, ",") {
		var clause []string
		startsWithConjunction := false
		for _, word := range statementWords(part) {
			if word != "and" && word != "but" {
				clause = append(clause, word)
				continue
			}
			if len(clause) > 0 {
				clauses = append(clauses, clause)
				joined = append(joined, startsWithConjunction)
			}
			clause = nil
			startsWithConjunction = true
		}
		if len(clause) > 0 {
			clauses = append(clauses, clause)
			joined = append(joined, startsWithConjunction)
		}
	}
	return clauses, joined
}

func statementWords(sentence string) []string {
	sentence = strings.ReplaceAll(strings.ToLower(sentence), "’", "'")
	return strings.FieldsFunc(sentence, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '\''
	})
}

func reportsOwnChange(words []string) bool {
	for i, word := range words {
		switch {
		case changeVerbs[word]:
			if endsWithOwnSubject(words[:i]) {
				return true
			}
		case word == "longer" && i > 0 && words[i-1] == "no":
			if endsWithOwnSubject(words[:i-1]) {
				return true
			}
		case word == "anymore" || (word == "more" && i > 0 && words[i-1] == "any"):
			if negatesOwnState(words[:i]) {
				return true
			}
		}
	}
	return false
}

// endsWithOwnSubject accepts "I", "we've just" or "I am" right before the
// change, so "my partner moved", "I haven't moved" and "if someday I moved"
// don't count.
func endsWithOwnSubject(before []string) bool {
	for i := len(before) - 1; i >= 0 && i >= len(before)-3; i-- {
		if firstPersonSubjects[before[i]] {
			return !slices.ContainsFunc(before[:i], isHypotheticalMarker)
		}
		if !changeModifiers[before[i]] {
			return false
		}
	}
	return false
}

func isHypotheticalMarker(word string) bool {
	return hypotheticalMarkers[word]
}

// negatesOwnState finds "I don't", "I'm not" or "I do not" earlier in the clause.
func negatesOwnState(before []string) bool {
	for i := 0; i+1 < len(before); i++ {
		if !firstPersonSubjects[before[i]] || slices.ContainsFunc(before[:i], isHypotheticalMarker) {
			continue
		}
		next := before[i+1]
		if negations[next] || ((next == "do" || next == "am" || next == "are") && i+2 < len(before) && negations[before[i+2]]) {
			return true
		}
	}
	return false
}

func anchorPairs(words []string) []wordPair {
	pairs := make([]wordPair, 0, len(words))
	previousIsContent := false
	for i, word := range words {
		isContent := isContentWord(word)
		if i > 0 && (previousIsContent || isContent) {
			pairs = append(pairs, wordPair{words[i-1], word})
		}
		previousIsContent = isContent
	}
	return pairs
}

// Contractions, possessives and single letters ("don't", "children's") would
// otherwise link unrelated statements.
func isContentWord(word string) bool {
	return utf8.RuneCountInString(word) > 1 && !functionWords[word] && !changeVerbs[word] &&
		!strings.Contains(word, "'")
}
