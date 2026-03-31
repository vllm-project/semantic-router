package nlgen

import "testing"

func TestSanitizeLLMOutput(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "already clean",
			input: `SIGNAL domain math { description: "math" }`,
			want:  `SIGNAL domain math { description: "math" }`,
		},
		{
			name:  "strip markdown fence",
			input: "Here is the DSL:\n```dsl\nSIGNAL domain math {\n  description: \"math\"\n}\n```\n",
			want:  "SIGNAL domain math {\n  description: \"math\"\n}",
		},
		{
			name:  "strip bare fence",
			input: "```\nSIGNAL domain math {\n  description: \"math\"\n}\n```",
			want:  "SIGNAL domain math {\n  description: \"math\"\n}",
		},
		{
			name:  "strip leading prose",
			input: "Sure! Here is the routing configuration:\n\nSIGNAL domain math {\n  description: \"math\"\n}",
			want:  "SIGNAL domain math {\n  description: \"math\"\n}",
		},
		{
			name:  "strip trailing prose",
			input: "SIGNAL domain math {\n  description: \"math\"\n}\n\nThis routes math queries to the math model.",
			want:  "SIGNAL domain math {\n  description: \"math\"\n}",
		},
		{
			name:  "fence with language tag",
			input: "```text\nROUTE math {\n  PRIORITY 100\n  MODEL \"qwen\"\n}\n```",
			want:  "ROUTE math {\n  PRIORITY 100\n  MODEL \"qwen\"\n}",
		},
		{
			name:  "empty input",
			input: "",
			want:  "",
		},
		{
			name:  "multiple blocks with prose",
			input: "Here:\n```\nSIGNAL domain math {\n  description: \"math\"\n}\n\nROUTE math {\n  PRIORITY 100\n  WHEN domain(\"math\")\n  MODEL \"qwen\"\n}\n```\nDone!",
			want:  "SIGNAL domain math {\n  description: \"math\"\n}\n\nROUTE math {\n  PRIORITY 100\n  WHEN domain(\"math\")\n  MODEL \"qwen\"\n}",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SanitizeLLMOutput(tt.input)
			if got != tt.want {
				t.Errorf("SanitizeLLMOutput() =\n%q\nwant\n%q", got, tt.want)
			}
		})
	}
}
