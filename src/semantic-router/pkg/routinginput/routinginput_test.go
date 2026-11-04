package routinginput

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const runtimeContextPattern = `(?s)^\s*OpenClaw runtime context:.*End OpenClaw runtime context\.\s*$`

func TestNilSelectorSelectsEverything(t *testing.T) {
	var selector *Selector
	if selector.SkipsMessage("OpenClaw runtime context:\nEnd OpenClaw runtime context.") {
		t.Fatal("a nil selector must not skip messages")
	}
	if text, removed := selector.Strip("<m>x</m> question"); text != "<m>x</m> question" || removed != nil {
		t.Fatalf("a nil selector stripped text: %q %q", text, removed)
	}
	if For(config.RoutingInputConfig{}) != nil {
		t.Fatal("an empty configuration must compile to a nil selector")
	}
}

func TestForSharesCompiledSelectors(t *testing.T) {
	cfg := config.RoutingInputConfig{SkipUserMessagesMatching: []string{runtimeContextPattern}}
	first := For(cfg)
	if first == nil || For(cfg) != first {
		t.Fatal("the same settings must share one compiled selector")
	}
	other := For(config.RoutingInputConfig{SkipUserMessagesMatching: []string{`^other$`}})
	if other == nil || other == first {
		t.Fatal("different settings must compile separately")
	}
	if For(config.RoutingInputConfig{SkipUserMessagesMatching: []string{"(unclosed"}}) != nil {
		t.Fatal("settings that do not compile must select nothing")
	}
}

func TestSkipsMessageMatchesRuntimeContextTurns(t *testing.T) {
	selector, err := Compile(config.RoutingInputConfig{SkipUserMessagesMatching: []string{runtimeContextPattern}})
	if err != nil {
		t.Fatal(err)
	}
	for text, want := range map[string]bool{
		"OpenClaw runtime context:\nConversation info: {\"chat_id\": \"chat-example\"}\nEnd OpenClaw runtime context.":    true,
		"  OpenClaw runtime context:\nEnd OpenClaw runtime context.\n":                                                    true,
		"What does \"OpenClaw runtime context:\" mean?":                                                                   false,
		"OpenClaw runtime context:\nEnd OpenClaw runtime context.\nAnd now my question: what is on my calendar tomorrow?": false,
		"": false,
	} {
		if got := selector.SkipsMessage(text); got != want {
			t.Errorf("SkipsMessage(%q) = %v, want %v", text, got, want)
		}
	}
}

func TestWalkAssignsRemovedText(t *testing.T) {
	selector, err := Compile(config.RoutingInputConfig{
		SkipUserMessagesMatching: []string{`^Client context:`},
		StripBlocks:              []config.RoutingInputStripBlock{{Start: "<m>", End: "</m>"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	type message struct {
		index    int
		text     string
		textOnly bool
		routed   string
		skipped  bool
	}
	tests := map[string]struct {
		messages []message
		want     Outcome
	}{
		"routing message between skipped turns": {
			messages: []message{
				{index: 0, text: "first question", textOnly: true, routed: "first question"},
				{index: 1, text: "Client context: a", textOnly: true, skipped: true},
				{index: 3, text: "<m>memory</m>\n\nsecond question", textOnly: true, routed: "second question"},
				{index: 4, text: "Client context: b", textOnly: true, skipped: true},
			},
			want: Outcome{
				SkippedMessages: 2, StrippedBlocks: 1, MessageIndex: 3, Stripped: []string{"<m>memory</m>"},
				ExcludedCurrent: []string{"<m>memory</m>", "Client context: b"},
				ExcludedPrior:   []string{"first question", "Client context: a"},
			},
		},
		"every user message skipped": {
			messages: []message{{index: 0, text: "Client context: a", textOnly: true, skipped: true}},
			want:     Outcome{SkippedMessages: 1, MessageIndex: -1, ExcludedCurrent: []string{"Client context: a"}},
		},
		"a trailing message that is only a block is current text": {
			messages: []message{
				{index: 0, text: "question", textOnly: true, routed: "question"},
				{index: 1, text: "<m>memory</m>", textOnly: true},
			},
			want: Outcome{StrippedBlocks: 1, MessageIndex: 0, ExcludedCurrent: []string{"<m>memory</m>"}},
		},
		"a message with media is never skipped": {
			messages: []message{{index: 0, text: "Client context: a", routed: "Client context: a"}},
			want:     Outcome{MessageIndex: 0},
		},
		"nothing removed": {
			messages: []message{
				{index: 0, text: "question", textOnly: true, routed: "question"},
				{index: 2, text: "follow-up", textOnly: true, routed: "follow-up"},
			},
			want: Outcome{MessageIndex: 2},
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			walk := selector.Walk()
			for _, message := range test.messages {
				routed, skipped := walk.UserMessage(message.index, message.text, message.textOnly)
				if routed != message.routed || skipped != message.skipped {
					t.Fatalf("UserMessage(%q) = %q, %v; want %q, %v", message.text, routed, skipped, message.routed, message.skipped)
				}
			}
			if got := walk.Outcome(); !reflect.DeepEqual(got, test.want) {
				t.Fatalf("Outcome() = %+v, want %+v", got, test.want)
			}
		})
	}
	var nothing *Selector
	walk := nothing.Walk()
	if routed, skipped := walk.UserMessage(0, "Client context: a", true); routed != "Client context: a" || skipped {
		t.Fatal("the walk of a nil selector must change nothing")
	}
	if got := walk.Outcome(); got.Changed() || got.MessageIndex != -1 {
		t.Fatalf("the walk of a nil selector reported %+v", got)
	}
}

func TestStrip(t *testing.T) {
	memories := config.RoutingInputStripBlock{Start: "<relevant-memories>", End: "</relevant-memories>"}
	recalled := config.RoutingInputStripBlock{Start: "<recalled-memories>", End: "</recalled-memories>"}
	tests := map[string]struct {
		blocks  []config.RoutingInputStripBlock
		text    string
		want    string
		removed []string
	}{
		"prepended memories": {
			blocks:  []config.RoutingInputStripBlock{memories},
			text:    "<relevant-memories>\n- prefers short answers\n</relevant-memories>\n\nWhat is on my calendar tomorrow?",
			want:    "What is on my calendar tomorrow?",
			removed: []string{"<relevant-memories>\n- prefers short answers\n</relevant-memories>"},
		},
		"blocks of several types in document order": {
			blocks:  []config.RoutingInputStripBlock{memories, recalled},
			text:    "<recalled-memories>a</recalled-memories> one <relevant-memories>b</relevant-memories> two",
			want:    "one  two",
			removed: []string{"<recalled-memories>a</recalled-memories>", "<relevant-memories>b</relevant-memories>"},
		},
		"nested markers end at the first end marker": {
			blocks:  []config.RoutingInputStripBlock{{Start: "<m>", End: "</m>"}},
			text:    "<m>outer <m>inner</m> tail</m> question",
			want:    "tail</m> question",
			removed: []string{"<m>outer <m>inner</m>"},
		},
		"unterminated block stays": {
			blocks: []config.RoutingInputStripBlock{memories},
			text:   "<relevant-memories>\n- cut off by the client\n\nWhat is on my calendar tomorrow?",
			want:   "<relevant-memories>\n- cut off by the client\n\nWhat is on my calendar tomorrow?",
		},
		"an unterminated block does not stop later blocks": {
			blocks:  []config.RoutingInputStripBlock{memories, recalled},
			text:    "<relevant-memories> open <recalled-memories>x</recalled-memories> question",
			want:    "<relevant-memories> open  question",
			removed: []string{"<recalled-memories>x</recalled-memories>"},
		},
		"a tie at one position goes to the block listed first, then to the next": {
			blocks:  []config.RoutingInputStripBlock{{Start: "<mem", End: "</never>"}, {Start: "<memory>", End: "</memory>"}},
			text:    "<memory>x</memory> question",
			want:    "question",
			removed: []string{"<memory>x</memory>"},
		},
		"identical start and end markers": {
			blocks:  []config.RoutingInputStripBlock{{Start: "---", End: "---"}},
			text:    "---\nclient: example\n---\nquestion",
			want:    "question",
			removed: []string{"---\nclient: example\n---"},
		},
		"an end marker before any start marker stays": {
			blocks: []config.RoutingInputStripBlock{memories},
			text:   "</relevant-memories> question <relevant-memories>",
			want:   "</relevant-memories> question <relevant-memories>",
		},
		"text that is only a block": {
			blocks:  []config.RoutingInputStripBlock{memories},
			text:    "<relevant-memories>x</relevant-memories>\n",
			want:    "",
			removed: []string{"<relevant-memories>x</relevant-memories>"},
		},
		"text without markers is unchanged": {
			blocks: []config.RoutingInputStripBlock{memories},
			text:   "  question with surrounding space  ",
			want:   "  question with surrounding space  ",
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			selector, err := Compile(config.RoutingInputConfig{StripBlocks: test.blocks})
			if err != nil {
				t.Fatal(err)
			}
			got, removed := selector.Strip(test.text)
			if got != test.want || !reflect.DeepEqual(removed, test.removed) {
				t.Fatalf("Strip(%q) = %q, %q; want %q, %q", test.text, got, removed, test.want, test.removed)
			}
		})
	}
}
