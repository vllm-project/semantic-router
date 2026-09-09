package mlmodelselection

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
)

func TestMockLLMUsesRunImageAndRequiresSuccessfulRollout(t *testing.T) {
	for _, tc := range []struct {
		name       string
		failure    string
		missing    bool
		wantError  string
		wantCalls  int
		wantDeploy bool
	}{
		{name: "ready", wantCalls: 2, wantDeploy: true},
		{name: "apply fails", failure: "apply", wantError: "failed to apply mock LLM", wantCalls: 1, wantDeploy: true},
		{name: "rollout fails", failure: "rollout", wantError: "mock LLM deployment not ready", wantCalls: 2, wantDeploy: true},
		{name: "missing image mapping", missing: true, wantError: "missing run image", wantCalls: 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := setupMockLLMCommands(t)
			t.Setenv("MOCK_COMMAND_FAILURE", tc.failure)
			original := readMockLLMFile(t, mockLLMManifest)
			image := "ghcr.io/vllm-project/semantic-router/mock-vllm:isolated-run"
			opts := &framework.SetupOptions{
				KubeConfig: filepath.Join(root, "private-kubeconfig"),
				LocalImages: map[string]string{
					mockLLMImage: image,
				},
			}
			if tc.missing {
				opts.LocalImages = nil
			}

			err := NewProfile().deployMockLLM(context.Background(), opts)
			if tc.wantError == "" {
				if err != nil {
					t.Fatalf("deploy mock LLM: %v", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.wantError) {
				t.Fatalf("error = %v, want %q", err, tc.wantError)
			}

			calls := strings.FieldsFunc(string(readMockLLMFile(t, filepath.Join(root, "commands"))), func(r rune) bool { return r == '\n' })
			if len(calls) != tc.wantCalls {
				t.Fatalf("commands = %v, want %d calls", calls, tc.wantCalls)
			}
			for _, call := range calls {
				if !strings.HasPrefix(call, opts.KubeConfig+"|") {
					t.Fatalf("command did not use this run's kubeconfig: %s", call)
				}
			}
			if tc.wantCalls == 2 && !strings.Contains(calls[1], "rollout status deployment/mock-llm --namespace default --timeout=5m") {
				t.Fatalf("mock readiness was not checked: %s", calls[1])
			}
			if tc.wantDeploy {
				applied := string(readMockLLMFile(t, filepath.Join(root, "applied")))
				if !strings.Contains(applied, "image: "+image) || strings.Contains(applied, mockLLMImage) {
					t.Fatalf("applied manifest did not use only the run image: %s", applied)
				}
				var deployment appsv1.Deployment
				if err := utilyaml.NewYAMLOrJSONDecoder(strings.NewReader(applied), 4096).Decode(&deployment); err != nil {
					t.Fatal(err)
				}
				if len(deployment.Spec.Template.Spec.Containers) != 1 {
					t.Fatal("expected one mock container")
				}
				probe := deployment.Spec.Template.Spec.Containers[0].ReadinessProbe
				if probe == nil || probe.HTTPGet == nil || probe.HTTPGet.Path != "/health" || probe.HTTPGet.Port.IntVal != 8000 {
					t.Fatal("rollout must wait for the mock HTTP server to be healthy")
				}
			}
			if !bytes.Equal(original, readMockLLMFile(t, mockLLMManifest)) {
				t.Fatal("deploy changed the checked-in mock manifest")
			}
		})
	}
}

func TestMockLLMTeardownDeletesTheAppliedResources(t *testing.T) {
	root := setupMockLLMCommands(t)
	kubeconfig := filepath.Join(root, "private-kubeconfig")
	if err := NewProfile().Teardown(context.Background(), &framework.TeardownOptions{KubeConfig: kubeconfig}); err != nil {
		t.Fatal(err)
	}
	calls := string(readMockLLMFile(t, filepath.Join(root, "commands")))
	wantDelete := kubeconfig + "|delete -f " + mockLLMManifest + " --ignore-not-found"
	if !strings.Contains(calls, wantDelete) {
		t.Fatalf("mock cleanup did not use the same manifest and private kubeconfig: %s", calls)
	}
	if strings.Contains(calls, "helm|uninstall mock-llm") {
		t.Fatal("attempted to uninstall a nonexistent mock Helm release")
	}
	if !bytes.Equal(readMockLLMFile(t, mockLLMManifest), readMockLLMFile(t, filepath.Join(root, "deleted"))) {
		t.Fatal("cleanup did not delete the mock Deployment and Service")
	}
}

func setupMockLLMCommands(t *testing.T) string {
	t.Helper()
	t.Chdir("../../..")
	root := t.TempDir()
	t.Setenv("PATH", root+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("KUBECONFIG", "unrelated-default-context")
	t.Setenv("MOCK_COMMAND_ROOT", root)
	script := `#!/bin/sh
set -eu
config="$KUBECONFIG"
if [ "$1" = "--kubeconfig" ]; then
  config="$2"
  shift 2
fi
printf '%s|%s\n' "$config" "$*" >> "$MOCK_COMMAND_ROOT/commands"
case "$1" in
  apply) cat "$3" > "$MOCK_COMMAND_ROOT/applied" ;;
  delete)
    case "$3" in */mock-llm.yaml) cat "$3" > "$MOCK_COMMAND_ROOT/deleted" ;; esac
    ;;
esac
if [ "${MOCK_COMMAND_FAILURE:-}" = "$1" ]; then
  echo "mock command failed: $1" >&2
  exit 17
fi
`
	for name, content := range map[string]string{
		"kubectl": script,
		"helm":    "#!/bin/sh\nprintf 'helm|%s\\n' \"$*\" >> \"$MOCK_COMMAND_ROOT/commands\"\n",
	} {
		if err := os.WriteFile(filepath.Join(root, name), []byte(content), 0o700); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, "commands"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	return root
}

func readMockLLMFile(t *testing.T, path string) []byte {
	t.Helper()
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return content
}
