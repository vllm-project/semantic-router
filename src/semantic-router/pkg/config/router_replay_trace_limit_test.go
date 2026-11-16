package config

import (
	"strings"
	"testing"
)

// traceLimits are the limits a policy gives a record's tool text: each field,
// the steps' raw copies, the steps together, and the prompt.
type traceLimits struct{ field, raw, total, prompt int }

func traceLimitsOf(p *RouterReplayPluginConfig) traceLimits {
	return traceLimits{p.ToolTraceByteLimit(), p.RawToolTraceByteLimit(), p.ToolTraceTotalByteLimit(), p.PromptByteLimit()}
}

// Where no layer sets max_tool_trace_bytes, a record's tool text keeps 64 KiB
// per field, raw copies included, and 256 KiB of steps in all, in either mode
// that keeps text. A max_tool_trace_bytes a layer sets, 0 and negative values
// included, keeps the trace as it was before these defaults: that limit on
// the normalized fields, whole raw copies and no total, unless
// max_tool_trace_total_bytes is set too. The prompt's limit never follows the
// trace's default.
func TestRouterReplayToolTraceLimits(t *testing.T) {
	const raw = "{enabled: true, prompt_capture: raw"
	defaults := traceLimits{64 << 10, 64 << 10, 256 << 10, 64 << 10}
	for _, test := range []struct {
		name, global, recipe, decision string
		inherits, own                  traceLimits
	}{
		{"neither set, raw", raw + "}", "{}", "{}", defaults, defaults},
		{"neither set, verified", "{enabled: true, prompt_capture: verified}", "{}", "{}", defaults, defaults},
		{
			"a trace limit keeps the trace as before", raw + ", max_tool_trace_bytes: 4096}", "{}", "{}",
			traceLimits{4096, 0, 0, 4096},
			traceLimits{4096, 0, 0, 4096},
		},
		{
			"a trace limit of 0 keeps whole traces", raw + ", max_tool_trace_bytes: 0}", "{}", "{}",
			traceLimits{0, 0, 0, 64 << 10},
			traceLimits{0, 0, 0, 64 << 10},
		},
		{
			"a negative trace limit is no limit, as before", raw + ", max_tool_trace_bytes: -1}", "{}", "{}",
			traceLimits{0, 0, 0, 64 << 10},
			traceLimits{0, 0, 0, 64 << 10},
		},
		{
			"a total alone keeps the field default", raw + ", max_tool_trace_total_bytes: 1048576}", "{}", "{}",
			traceLimits{64 << 10, 64 << 10, 1 << 20, 64 << 10},
			traceLimits{64 << 10, 64 << 10, 1 << 20, 64 << 10},
		},
		{
			"a total of 0 keeps every step", raw + ", max_tool_trace_total_bytes: 0}", "{}", "{}",
			traceLimits{64 << 10, 64 << 10, 0, 64 << 10},
			traceLimits{64 << 10, 64 << 10, 0, 64 << 10},
		},
		{
			"both set", raw + ", max_tool_trace_bytes: 8192, max_tool_trace_total_bytes: 524288}", "{}", "{}",
			traceLimits{8192, 0, 512 << 10, 8192},
			traceLimits{8192, 0, 512 << 10, 8192},
		},
		{
			"a decision's trace limit where no other layer sets one", raw + "}", "{}", "{max_tool_trace_bytes: 4096}",
			defaults,
			traceLimits{4096, 0, 0, 4096},
		},
		{
			"a decision's total over a global trace limit", raw + ", max_tool_trace_bytes: 4096}", "{}", "{max_tool_trace_total_bytes: 131072}",
			traceLimits{4096, 0, 0, 4096},
			traceLimits{4096, 0, 128 << 10, 4096},
		},
		{
			"a recipe's 0 reaches its decisions", raw + "}", "{max_tool_trace_bytes: 0}", "{}",
			traceLimits{0, 0, 0, 64 << 10},
			traceLimits{0, 0, 0, 64 << 10},
		},
		{
			"a decision over its recipe", raw + "}", "{max_tool_trace_bytes: 0}", "{max_tool_trace_bytes: 16384}",
			traceLimits{0, 0, 0, 64 << 10},
			traceLimits{16384, 0, 0, 16384},
		},
		{
			"a decision's prompt limit leaves the trace's defaults", raw + "}", "{}", "{max_prompt_bytes: 1048576}",
			defaults,
			traceLimits{64 << 10, 64 << 10, 256 << 10, 1 << 20},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg, err := ParseYAMLBytes([]byte(strings.NewReplacer("GLOBAL", test.global, "RECIPE", test.recipe, "DECISION", test.decision).Replace(promptLimitDocument)))
			if err != nil {
				t.Fatal(err)
			}
			limits := map[string]traceLimits{}
			for i := range cfg.Decisions {
				limits[cfg.Decisions[i].Name] = traceLimitsOf(cfg.EffectiveRouterReplayConfig(&cfg.Decisions[i]))
			}
			if limits["inherits"] != test.inherits || limits["own"] != test.own {
				t.Fatalf("limits %+v and %+v, want %+v and %+v", limits["inherits"], limits["own"], test.inherits, test.own)
			}
		})
	}
	var unset *RouterReplayPluginConfig
	if got := traceLimitsOf(unset); got != defaults {
		t.Fatalf("a nil policy gives %+v, want the defaults %+v", got, defaults)
	}
}

// A negative max_tool_trace_total_bytes is refused wherever it is written.
func TestRouterReplayRefusesANegativeMaxToolTraceTotalBytes(t *testing.T) {
	for _, layer := range []string{"GLOBAL", "RECIPE", "DECISION"} {
		replacements := map[string]string{"GLOBAL": "{enabled: true, prompt_capture: raw}", "RECIPE": "{}", "DECISION": "{}"}
		replacements[layer] = "{enabled: true, max_tool_trace_total_bytes: -1}"
		document := strings.NewReplacer("GLOBAL", replacements["GLOBAL"], "RECIPE", replacements["RECIPE"], "DECISION", replacements["DECISION"]).Replace(promptLimitDocument)
		if _, err := ParseYAMLBytes([]byte(document)); err == nil || !strings.Contains(err.Error(), "max_tool_trace_total_bytes cannot be negative") {
			t.Fatalf("%s: a negative max_tool_trace_total_bytes loaded: %v", layer, err)
		}
	}
}
