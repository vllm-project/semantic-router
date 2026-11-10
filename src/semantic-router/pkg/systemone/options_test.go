package systemone

import (
	"strings"
	"testing"
)

func TestPublicOptionsEncodeOnlyTheOptionsTheAPIDefines(t *testing.T) {
	for raw, want := range map[string]string{
		`{}`:                   `{}`,
		`null`:                 `{}`,
		`{"return_meta":true}`: `{"return_meta":true}`,
		`{ "return_meta" : false , "max_tokens" : 512 , "deadline_ms" : 1500.5 }`: `{"deadline_ms":1500.5,"max_tokens":512,"return_meta":false}`,
		`{"deadline_ms":30000,"deadline_ms":20}`:                                  `{"deadline_ms":20}`,
	} {
		got, problem := publicOptions([]byte(raw))
		if problem != "" || string(got) != want {
			t.Fatalf("publicOptions(%s) = %s, %q; want %s", raw, got, problem, want)
		}
	}
}

func TestPublicOptionsRefuseWhatTheAPIDoesNotDefine(t *testing.T) {
	removed := "options.profile was removed: batching and numerics are the deployment's engine arguments " +
		"(max_num_seqs, max_num_batched_tokens, batch_window_ms, enable_shared_context, dtype)"
	for raw, want := range map[string]string{
		`[]`:                           "options must be an object",
		`"exact"`:                      "options must be an object",
		`true`:                         "options must be an object",
		`{"profile":"exact"}`:          removed,
		`{"profile":null}`:             removed,
		`{"verbose":true}`:             "options.verbose is not a System One option; the options are deadline_ms, long_context, max_tokens and return_meta",
		`{"zeta":1,"alpha":1}`:         "options.alpha is not a System One option; the options are deadline_ms, long_context, max_tokens and return_meta",
		`{"deadline_ms":0}`:            "options.deadline_ms must be a number greater than 0",
		`{"deadline_ms":-5}`:           "options.deadline_ms must be a number greater than 0",
		`{"deadline_ms":"30000"}`:      "options.deadline_ms must be a number greater than 0",
		`{"deadline_ms":null}`:         "options.deadline_ms must be a number greater than 0",
		`{"deadline_ms":1e400}`:        "options.deadline_ms must be a number greater than 0",
		`{"max_tokens":0}`:             "options.max_tokens must be an integer of at least 1",
		`{"max_tokens":1.5}`:           "options.max_tokens must be an integer of at least 1",
		`{"max_tokens":"512"}`:         "options.max_tokens must be an integer of at least 1",
		`{"max_tokens":true}`:          "options.max_tokens must be an integer of at least 1",
		`{"return_meta":"true"}`:       "options.return_meta must be true or false",
		`{"return_meta":null}`:         "options.return_meta must be true or false",
		`{"return_meta":1}`:            "options.return_meta must be true or false",
		`{"return_meta":true,"x":[1]}`: "options.x is not a System One option; the options are deadline_ms, long_context, max_tokens and return_meta",
	} {
		got, problem := publicOptions([]byte(raw))
		if got != nil || problem != want {
			t.Fatalf("publicOptions(%s) = %s, %q; want %q", raw, got, problem, want)
		}
		if strings.Contains(problem, "/") {
			t.Fatalf("an option problem names no path: %q", problem)
		}
	}
}
