package classification

import (
	"cmp"
	"regexp"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ActionClassification is the one action a user message names. Score is the
// share of matched action phrases that name it: 1 when the message names a
// single action (or none, for other), lower when it mixes several.
type ActionClassification struct {
	Action string
	Score  float64
}

type actionPhrases struct {
	action  string
	phrases []string
}

type actionPattern struct {
	action     string
	expression *regexp.Regexp
}

type actionMatch struct {
	order int
	start int
	end   int
}

// Declaration order breaks ties between phrases of equal span.
var actionPatterns = compileActionPatterns([]actionPhrases{
	{config.ActionGenerate, []string{
		`(?:write|create|generate|implement|build|scaffold|draft|produce)\s+(?:me\s+)?(?:a|an|some|new)`,
		`write\s+(?:me\s+)?code`,
		`implement`,
		`code\s+up`,
		`add\s+(?:a|an|new|support\s+for)`,
		`(?:give|show)\s+me\s+(?:a|an)\s+(?:function|script|class|snippet|implementation|program|query|regex|command)`,
		`i\s+(?:need|want)\s+(?:a|an)\s+(?:function|script|class|program|tool|endpoint|component|cli|service|page|api)`,
		`boilerplate`,
		`starter\s+(?:code|template|project)`,
	}},
	{config.ActionExplain, []string{
		`explain(?:s|ing)?`,
		`what(?:[’']s|\s+(?:does|do|is|are|was|were|happens))`,
		`why`,
		`how\s+(?:does|do|is|are|did|can|could|should|would|to)`,
		`should\s+(?:i|we)`,
		`walk\s+me\s+through`,
		`help\s+me\s+understand`,
		`tell\s+me\s+(?:about|what|how|why|more)`,
		`describe`,
		`summari[sz]e`,
		`clarify`,
		`meaning\s+of`,
		`difference\s+between`,
		`i\s+don[’']?t\s+understand`,
		`confused\s+(?:about|by)`,
		`purpose\s+of`,
		`is\s+it\s+(?:possible|safe|better|ok|okay)`,
		`which\s+(?:is|one|should)`,
	}},
	{config.ActionFix, []string{
		`fix(?:es|ed|ing)?`,
		`debug(?:s|ged|ging)?`,
		`repair`,
		`patch\s+(?:this|the|it)`,
		`resolve`,
		`(?:doesn[’']?t|does\s+not|isn[’']?t|is\s+not|won[’']?t|will\s+not|can[’']?t|cannot|not|no\s+longer|stopped)\s+(?:work|working|compile|compiling|run|running|build|building|pass|passing|load|loading)`,
		`(?:is|are|keeps|keep)\s+(?:broken|failing|crashing|erroring|throwing|hanging|timing\s+out)`,
		`(?:getting|got|seeing|hitting)\s+(?:an?\s+|this\s+|the\s+)?(?:error|exception|crash|segfault|panic|bug)`,
		`make\s+(?:it|this|the\s+tests?)\s+(?:work|pass|compile)`,
		`(?:correct|solve)\s+(?:the|this|my|these)`,
	}},
	{config.ActionRefactor, []string{
		`refactor(?:s|ed|ing)?`,
		`renam(?:e|es|ed|ing)`,
		`clean\s+(?:up|this|it)`,
		`cleanup`,
		`simplif(?:y|ies|ied|ying)`,
		`restructur(?:e|es|ed|ing)`,
		`reorganiz(?:e|es|ed|ing)`,
		`rewrit(?:e|es|ing|ten)`,
		`extract\s+(?:a|an|the|this|that|it|into|out)`,
		`split\s+(?:this|the|it|up)`,
		`de-?duplicate`,
		`dedupe`,
		`dry\s+(?:this|it)\s+up`,
		`(?:convert|migrate|port)\s+(?:this|it|the)`,
		`(?:make|making)\s+(?:it|this|the\s+code)\s+(?:more\s+)?(?:readable|cleaner|simpler|idiomatic|pythonic|maintainable|modular)`,
		`improve\s+(?:the\s+)?(?:readability|structure|naming|code\s+quality)`,
		`inline\s+(?:this|the)`,
		`modernize`,
		`tidy\s+up`,
	}},
	{config.ActionTest, []string{
		`(?:write|add|create|generate|need|want)\s+(?:some\s+|more\s+|a\s+|an\s+|the\s+)?(?:unit\s+|integration\s+|e2e\s+|end-to-end\s+|regression\s+|table-driven\s+|snapshot\s+|property-based\s+|pytest\s+|jest\s+|go\s+)?tests?`,
		`test\s+cases?`,
		`test\s+coverage`,
		`cover\s+(?:this|it|the\s+\w+)\s+with\s+tests`,
		`(?:increase|improve)\s+(?:the\s+)?(?:test\s+)?coverage`,
		`test\s+(?:this|it|that|these)`,
		`(?:mock|stub)\s+(?:the|this|out)`,
	}},
})

// Code and logs pasted inside backticks are content, not the request.
var actionCodeSpans = regexp.MustCompile("(?s)```.*?```|`[^`\n]*`")

func compileActionPatterns(families []actionPhrases) []actionPattern {
	patterns := make([]actionPattern, 0, len(families))
	for _, family := range families {
		expression := regexp.MustCompile(`(?i)\b(?:` + strings.Join(family.phrases, "|") + `)\b`)
		expression.Longest()
		patterns = append(patterns, actionPattern{action: family.action, expression: expression})
	}
	return patterns
}

// ClassifyAction returns the action named by the earliest action phrase in
// text; a longer phrase wins over a shorter one starting at the same place.
func ClassifyAction(text string) ActionClassification {
	text = actionCodeSpans.ReplaceAllString(text, " ")
	var matches []actionMatch
	for order, pattern := range actionPatterns {
		for _, span := range pattern.expression.FindAllStringIndex(text, -1) {
			matches = append(matches, actionMatch{order: order, start: span[0], end: span[1]})
		}
	}
	if len(matches) == 0 {
		return ActionClassification{Action: config.ActionOther, Score: 1}
	}
	slices.SortFunc(matches, func(a, b actionMatch) int {
		return cmp.Or(cmp.Compare(a.start, b.start), cmp.Compare(b.end, a.end), cmp.Compare(a.order, b.order))
	})

	counts := make([]int, len(actionPatterns))
	taken, covered := 0, 0
	for _, match := range matches {
		if match.start < covered {
			continue
		}
		counts[match.order]++
		taken++
		covered = match.end
	}
	chosen := matches[0].order
	return ActionClassification{
		Action: actionPatterns[chosen].action,
		Score:  float64(counts[chosen]) / float64(taken),
	}
}
