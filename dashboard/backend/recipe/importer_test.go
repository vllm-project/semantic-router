package recipe

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestStoreImportsListsAndIdempotentlyInstallsExactFiveZIP(t *testing.T) {
	directory := copyMaintainedRecipe(t, "accuracy")
	archive := recipeZIPFromDirectory(t, directory, nil)
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/zip")
		_, _ = w.Write(archive)
	}))
	defer server.Close()
	configPath := filepath.Join(t.TempDir(), "runtime-config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	store := NewStore(StoreOptions{Root: filepath.Join(t.TempDir(), "store"), ConfigPath: configPath, HTTPClient: server.Client()})
	request := ImportRequest{URL: server.URL + "/recipe.zip?signature=hidden"}
	installed, created, err := store.Import(context.Background(), request)
	if err != nil {
		t.Fatalf("Import(): %v", err)
	}
	if !created || installed.ID != "accuracy" || installed.RecipeDigest == "" || installed.ArchiveVerified {
		t.Fatalf("installed = %#v, created=%v", installed, created)
	}
	if installed.Counts.Probes == 0 || installed.Counts.Decisions == 0 {
		t.Fatalf("counts = %#v", installed.Counts)
	}
	again, created, err := store.Import(context.Background(), request)
	if err != nil || created || again.RecipeDigest != installed.RecipeDigest {
		t.Fatalf("idempotent Import() = %#v, %v, %v", again, created, err)
	}
	list, err := store.List(PackageListOptions{Page: 1, PageSize: 25, Query: installed.RecipeDigest[7:23]})
	if err != nil || list.Total != 1 || len(list.Items) != 1 || list.Items[0].Active {
		t.Fatalf("List() = %#v, %v", list, err)
	}
	var record packageRecord
	if err := readStrictJSONFile(recordRefPath(store.root, packageRecord{ID: "accuracy", Version: "0.1.0"}), 1<<20, &record); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(record.SourceURL, "signature") || strings.Contains(record.SourceURL, "hidden") {
		t.Fatalf("stored source URL leaked query: %q", record.SourceURL)
	}
}

func TestExtractRecipeZIPRejectsUnsafeEntries(t *testing.T) {
	directory := copyMaintainedRecipe(t, "accuracy")
	tests := map[string]zipEntryOverride{
		"traversal":      {index: 4, header: zip.FileHeader{Name: "../README.md", Method: zip.Store}},
		"backslash":      {index: 4, header: zip.FileHeader{Name: `nested\README.md`, Method: zip.Store}},
		"duplicate case": {index: 4, header: zip.FileHeader{Name: "CONFIG.YAML", Method: zip.Store}},
		"directory":      {index: 4, header: zipHeaderWithMode("README.md", os.ModeDir|0o755)},
		"symlink":        {index: 4, header: zipHeaderWithMode("README.md", os.ModeSymlink|0o777)},
		"encrypted":      {index: 4, header: zip.FileHeader{Name: "README.md", Method: zip.Store, Flags: 0x1}},
	}
	for name, override := range tests {
		t.Run(name, func(t *testing.T) {
			archive := recipeZIPWithOverride(t, directory, override)
			if _, err := extractRecipeZIP(archive); err == nil {
				t.Fatal("unsafe ZIP accepted")
			}
		})
	}
	archive := recipeZIPFromDirectory(t, directory, func(writer *zip.Writer) { _, _ = writer.Create("extra.txt") })
	if _, err := extractRecipeZIP(archive); err == nil {
		t.Fatal("ZIP with extra root entry accepted")
	}
}

func TestExtractRecipeZIPAllowsHighlyCompressibleBoundedContentAndRejectsExpandedFileLimit(t *testing.T) {
	directory := copyMaintainedRecipe(t, "accuracy")
	if err := os.WriteFile(filepath.Join(directory, "README.md"), bytes.Repeat([]byte("A"), 256<<10), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := extractRecipeZIP(recipeZIPFromDirectory(t, directory, nil)); err != nil {
		t.Fatalf("bounded highly-compressible Recipe content rejected: %v", err)
	}
	if err := os.WriteFile(filepath.Join(directory, "README.md"), bytes.Repeat([]byte("B"), int(maxREADMEBytes+1)), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := extractRecipeZIP(recipeZIPFromDirectory(t, directory, nil)); err == nil {
		t.Fatal("oversized expanded Recipe file accepted")
	}
}

func TestExtractRecipeZIPRejectsInvalidUTF8AndDigestMismatch(t *testing.T) {
	directory := copyMaintainedRecipe(t, "accuracy")
	archive := recipeZIPFromDirectory(t, directory, nil)
	if _, err := normalizeExpectedArchiveDigest("not-a-digest"); err == nil {
		t.Fatal("invalid expected digest accepted")
	}
	reader, err := zip.NewReader(bytes.NewReader(archive), int64(len(archive)))
	if err != nil || len(reader.File) != 5 {
		t.Fatal(err)
	}
	var invalid bytes.Buffer
	writer := zip.NewWriter(&invalid)
	for _, entry := range reader.File {
		output, _ := writer.Create(entry.Name)
		input, _ := entry.Open()
		if entry.Name == "README.md" {
			_, _ = output.Write([]byte{0xff, 0xfe})
		} else {
			content, _ := io.ReadAll(io.LimitReader(input, maxExpandedBytes+1))
			_, _ = output.Write(content)
		}
		_ = input.Close()
	}
	_ = writer.Close()
	if _, err := extractRecipeZIP(invalid.Bytes()); err == nil {
		t.Fatal("invalid UTF-8 accepted")
	}
}

func TestURLAndIPSecurityPolicy(t *testing.T) {
	for _, raw := range []string{"http://example.com/a.zip", "https://user:pass@example.com/a.zip", "file:///tmp/a.zip"} {
		if _, err := normalizeRemotePackageURL(raw); err == nil {
			t.Fatalf("unsafe URL accepted: %s", raw)
		}
	}
	for _, raw := range []string{
		"127.0.0.1", "10.0.0.1", "169.254.169.254", "192.0.2.1",
		"::1", "::127.0.0.1", "::ffff:127.0.0.1", "64:ff9b::7f00:1", "64:ff9b:1::7f00:1",
		"2001:0000:4136:e378:8000:63bf:3fff:fdd2", "2002:7f00:1::",
		"fc00::1", "fec0::1",
	} {
		if isPublicIP(netipMustParse(raw)) {
			t.Fatalf("non-public address accepted: %s", raw)
		}
	}
	if !isPublicIP(netipMustParse("8.8.8.8")) {
		t.Fatal("public address rejected")
	}
}

func TestPackageRedirectPolicyAllowsPublicGitHubReleaseAndRejectsUnsafeRedirects(t *testing.T) {
	client := newPackageHTTPClient(staticIPResolver{addresses: []netip.Addr{netip.MustParseAddr("8.8.8.8")}})
	if transport, ok := client.Transport.(*http.Transport); !ok || transport.Proxy != nil {
		t.Fatalf("package transport proxy = %#v", client.Transport)
	}
	allowed, _ := http.NewRequest(http.MethodGet, "https://release-assets.githubusercontent.com/github-production-release-asset/file.zip?token=redacted", nil)
	allowed.Header.Set("Referer", "https://github.com/release.zip?signature=secret")
	if err := client.CheckRedirect(allowed, []*http.Request{{URL: &url.URL{Scheme: "https", Host: "github.com"}}}); err != nil {
		t.Fatalf("public GitHub release redirect rejected: %v", err)
	}
	if allowed.Header.Get("Referer") != "" {
		t.Fatal("signed source URL leaked through redirect Referer")
	}
	for _, target := range []string{
		"http://release-assets.githubusercontent.com/file.zip",
		"https://user:token@release-assets.githubusercontent.com/file.zip",
		"https://release-assets.githubusercontent.com/file.zip#fragment",
	} {
		request, _ := http.NewRequest(http.MethodGet, target, nil)
		if err := client.CheckRedirect(request, nil); err == nil {
			t.Fatalf("unsafe redirect accepted: %s", target)
		}
	}
	via := make([]*http.Request, maxRedirects)
	if err := client.CheckRedirect(allowed, via); err == nil {
		t.Fatal("redirect limit was not enforced")
	}
}

func TestPackageHTTPClientRejectsMixedPublicAndPrivateDNSAnswers(t *testing.T) {
	client := newPackageHTTPClient(staticIPResolver{addresses: []netip.Addr{
		netip.MustParseAddr("8.8.8.8"),
		netip.MustParseAddr("127.0.0.1"),
	}})
	request, _ := http.NewRequest(http.MethodGet, "https://packages.example.invalid/recipe.zip", nil)
	response, err := client.Do(request)
	if response != nil {
		_ = response.Body.Close()
	}
	packageErr, ok := AsPackageError(err)
	if !ok || packageErr.Code != ErrorSourceForbidden {
		t.Fatalf("Do() error = %#v", err)
	}
}

func TestStoreImportVerifiesArchiveDigestAndDownloadLimit(t *testing.T) {
	directory := copyMaintainedRecipe(t, "accuracy")
	archive := recipeZIPFromDirectory(t, directory, nil)
	digest := sha256.Sum256(archive)
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/large.zip" {
			w.Header().Set("Content-Length", fmt.Sprintf("%d", maxArchiveBytes+1))
			return
		}
		_, _ = w.Write(archive)
	}))
	defer server.Close()
	configPath := filepath.Join(t.TempDir(), "runtime-config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	store := NewStore(StoreOptions{Root: filepath.Join(t.TempDir(), "store"), ConfigPath: configPath, HTTPClient: server.Client()})
	verified, created, err := store.Import(context.Background(), ImportRequest{
		URL:                   server.URL + "/recipe.zip",
		ExpectedArchiveSHA256: hex.EncodeToString(digest[:]),
	})
	if err != nil || !created || !verified.ArchiveVerified || verified.ArchiveSHA256 != "sha256:"+hex.EncodeToString(digest[:]) {
		t.Fatalf("verified Import() = %#v, %v, %v", verified, created, err)
	}
	_, _, err = store.Import(context.Background(), ImportRequest{URL: server.URL + "/recipe.zip", ExpectedArchiveSHA256: strings.Repeat("0", 64)})
	if packageErr, ok := AsPackageError(err); !ok || packageErr.Code != ErrorArchiveDigestMismatch {
		t.Fatalf("digest mismatch error = %#v", err)
	}
	_, _, err = store.Import(context.Background(), ImportRequest{URL: server.URL + "/large.zip"})
	if packageErr, ok := AsPackageError(err); !ok || packageErr.Code != ErrorDownloadTooLarge {
		t.Fatalf("large download error = %#v", err)
	}
}

func TestStoreImportRollsSameIDVersionAndPreservesActiveOldDigest(t *testing.T) {
	directory := copyMaintainedRecipe(t, "accuracy")
	first := recipeZIPFromDirectory(t, directory, nil)
	if err := os.WriteFile(filepath.Join(directory, "README.md"), []byte("different package content\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	second := recipeZIPFromDirectory(t, directory, nil)
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/second.zip" {
			_, _ = w.Write(second)
			return
		}
		_, _ = w.Write(first)
	}))
	defer server.Close()
	configPath := filepath.Join(t.TempDir(), "runtime-config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	store := NewStore(StoreOptions{Root: filepath.Join(t.TempDir(), "store"), ConfigPath: configPath, HTTPClient: server.Client()})
	firstSummary, _, err := store.Import(context.Background(), ImportRequest{URL: server.URL + "/first.zip"})
	if err != nil {
		t.Fatal(err)
	}
	_, firstConfig, err := store.ActivationTarget(firstSummary.RecipeDigest, true)
	if err != nil {
		t.Fatal(err)
	}
	previous := []byte("version: v0.3\n")
	transaction, err := store.BeginActivation(firstSummary.RecipeDigest, previous)
	if err != nil {
		t.Fatal(err)
	}
	if writeErr := os.WriteFile(configPath, firstConfig, 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	if commitErr := store.CommitActivation(transaction, ActivePointer{
		RecipeDigest: firstSummary.RecipeDigest, ConfigDigest: firstSummary.ConfigDigest, RealizedConfigDigest: digestBytes(firstConfig),
	}); commitErr != nil {
		t.Fatal(commitErr)
	}
	secondSummary, created, err := store.Import(context.Background(), ImportRequest{URL: server.URL + "/second.zip"})
	if err != nil || !created || secondSummary.RecipeDigest == firstSummary.RecipeDigest {
		t.Fatalf("rolling import = %#v, created=%v, err=%v", secondSummary, created, err)
	}
	for _, digest := range []string{firstSummary.RecipeDigest, secondSummary.RecipeDigest} {
		if _, objectErr := store.ObjectDirectory(digest); objectErr != nil {
			t.Fatalf("ObjectDirectory(%s): %v", digest, objectErr)
		}
		if _, packageLookupErr := store.Package(digest); packageLookupErr != nil {
			t.Fatalf("Package(%s): %v", digest, packageLookupErr)
		}
	}
	list, err := store.List(PackageListOptions{PageSize: 25})
	if err != nil || len(list.Items) != 2 || list.ActiveDigest != firstSummary.RecipeDigest {
		t.Fatalf("List() = %#v, %v", list, err)
	}
	activeCount := 0
	for _, item := range list.Items {
		if item.Active {
			activeCount++
			if item.RecipeDigest != firstSummary.RecipeDigest {
				t.Fatalf("new rolling ref silently became active: %#v", item)
			}
		}
	}
	if activeCount != 1 {
		t.Fatalf("active package count = %d; list=%#v", activeCount, list)
	}
}

func TestCredentialWarningsOnlyExemptPureEnvironmentSelectors(t *testing.T) {
	warnings := detectCredentialWarnings([]byte(`
api_key: ${PURE_TOKEN}
api_key_env: PURE_ACCESS
password: ${PASSWORD:-literal-fallback}
secret: ${SECRET-literal-fallback}
token: literal-token
auth_token: $BARE_TOKEN
client-secret: literal-client
privateKey: literal-private
headers:
  Authorization: Bearer literal
  Cookie: session=literal-cookie
  Proxy-Authorization: Bearer literal-proxy
extra_headers:
  X-API-Key: literal-key
`))
	if len(warnings) != 10 {
		t.Fatalf("warnings = %#v", warnings)
	}
	for _, warning := range warnings {
		if warning.Code != "literal_credential" || !strings.HasPrefix(warning.Path, "/") || strings.Contains(warning.Message, "fallback") {
			t.Fatalf("warning = %#v", warning)
		}
	}
	if err := validateRuntimeConfig([]byte("version: v0.3\nsetup:\n  mode: true\n")); err == nil {
		t.Fatal("setup.mode=true config accepted")
	}
}

func TestPackageValidationRejectsLiteralCredentialBeforeInstall(t *testing.T) {
	directory := copyMaintainedRecipe(t, "accuracy")
	configPath := filepath.Join(directory, "config.yaml")
	config, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	config = append(config, []byte("\nglobal:\n  integrations:\n    external_models:\n      extra_headers:\n        Authorization: Bearer do-not-store\n")...)
	if writeErr := os.WriteFile(configPath, config, 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	// Exercise the policy directly because changing config.yaml makes the DSL
	// intentionally non-canonical before the package reaches store promotion.
	warnings := detectCredentialWarnings(config)
	if len(warnings) != 1 || warnings[0].Path != "/global/integrations/external_models/extra_headers/Authorization" {
		t.Fatalf("credential warnings = %#v", warnings)
	}
	validation, err := validatePackageDirectory(directory)
	if err == nil || len(validation.warnings) != 0 {
		t.Fatalf("validatePackageDirectory() = %#v, %v", validation, err)
	}
	packageErr, ok := AsPackageError(err)
	if !ok || packageErr.Code != ErrorInvalidRecipe || strings.Contains(packageErr.Safe, "do-not-store") {
		t.Fatalf("validation error = %#v", err)
	}
}

func TestEnvironmentReferenceScannerMatchesRouterExpansionForms(t *testing.T) {
	config := []byte(`
plain: $PLAIN_NAME
braced: ${BRACED_NAME}
empty_default: ${EMPTY_DEFAULT:-fallback}
unset_default: ${UNSET_DEFAULT-fallback}
lowercase: ${host_secret}
invalid_default_name: ${:-fallback}
escaped: $$NOT_A_REFERENCE
`)
	_, err := extractEnvironmentBindings(config)
	if err == nil {
		t.Fatal("lowercase environment reference was accepted")
	}
	if (!strings.Contains(err.Error(), "/lowercase") && !strings.Contains(err.Error(), "/invalid_default_name")) || strings.Contains(err.Error(), "host-secret-value") {
		t.Fatalf("unsafe binding error = %v", err)
	}

	valid := bytes.Replace(config, []byte("${host_secret}"), []byte("${HOST_SECRET}"), 1)
	valid = bytes.Replace(valid, []byte("${:-fallback}"), []byte("${VALID_DEFAULT:-fallback}"), 1)
	bindings, err := extractEnvironmentBindings(valid)
	if err != nil {
		t.Fatalf("extractEnvironmentBindings() = %v", err)
	}
	names := make([]string, 0, len(bindings))
	for _, binding := range bindings {
		names = append(names, binding.name)
	}
	for _, expected := range []string{"PLAIN_NAME", "BRACED_NAME", "EMPTY_DEFAULT", "UNSET_DEFAULT", "HOST_SECRET", "VALID_DEFAULT"} {
		if !slices.Contains(names, expected) {
			t.Fatalf("bindings = %#v, missing %s", bindings, expected)
		}
	}
	if slices.Contains(names, "NOT_A_REFERENCE") {
		t.Fatalf("escaped dollar became binding: %#v", bindings)
	}
}

func TestPackageRuntimeValidationRejectsUnsafeKnowledgeBasePaths(t *testing.T) {
	tests := []struct {
		path string
		ok   bool
	}{
		{path: "knowledge_bases/privacy/", ok: true},
		{path: "knowledge_bases/team/privacy/", ok: true},
		{path: "../dashboard-data"},
		{path: "knowledge_bases/../.vllm-sr"},
		{path: "knowledge_bases//privacy"},
		{path: "knowledge_bases\\privacy"},
		{path: "/app/.vllm-sr"},
	}
	for _, test := range tests {
		t.Run(strings.ReplaceAll(test.path, "/", "_"), func(t *testing.T) {
			config := []byte(fmt.Sprintf(`
version: v0.3
global:
  model_catalog:
    kbs:
      - name: privacy_kb
        source:
          path: %q
        threshold: 0.5
`, test.path))
			err := validateRuntimeConfig(config)
			if test.ok && err != nil {
				t.Fatalf("validateRuntimeConfig() = %v", err)
			}
			if !test.ok && err == nil {
				t.Fatal("unsafe knowledge-base path was accepted")
			}
		})
	}
}

func TestPackageRuntimeValidationRejectsYAMLIndirection(t *testing.T) {
	for name, config := range map[string]string{
		"alias": `version: v0.3
literal: &private_value ${HOST_SECRET}
headers:
  Authorization: *private_value
`,
		"merge": `version: v0.3
defaults: &private_headers
  Authorization: ${HOST_SECRET}
headers:
  <<: *private_headers
`,
		"binary value": `version: v0.3
provider:
  base_url: !!binary aHR0cHM6Ly91c2VyOnNlbnNpdGl2ZUBleGFtcGxlLmNvbS92MQ==
`,
		"binary key": `version: v0.3
request:
  headers:
    !!binary QXV0aG9yaXphdGlvbg==: Bearer sensitive
`,
		"explicit string": `version: v0.3
value: !!str sensitive
`,
	} {
		t.Run(name, func(t *testing.T) {
			if err := validateRuntimeConfig([]byte(config)); err == nil || strings.Contains(err.Error(), "HOST_SECRET") {
				t.Fatalf("validateRuntimeConfig() = %v", err)
			}
		})
	}
}

func TestPackageValidationRejectsIndirectionInEveryYAMLDocument(t *testing.T) {
	for _, test := range []struct {
		filename string
		content  string
	}{
		{
			filename: "metadata.yaml",
			content:  "private: &private sensitive\ncopy: *private\n",
		},
		{
			filename: "probes.yaml",
			content:  "private: &private sensitive\ncopy: *private\n",
		},
	} {
		t.Run(test.filename, func(t *testing.T) {
			if err := rejectYAMLIndirection(test.filename, []byte(test.content)); err == nil || !strings.Contains(err.Error(), test.filename) {
				t.Fatalf("rejectYAMLIndirection() = %v", err)
			}
			directory := copyMaintainedRecipe(t, "accuracy")
			path := filepath.Join(directory, test.filename)
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if writeErr := os.WriteFile(path, append(data, test.content...), 0o600); writeErr != nil {
				t.Fatal(writeErr)
			}

			_, err = validatePackageDirectory(directory)
			packageErr, ok := AsPackageError(err)
			if !ok || packageErr.Code != ErrorInvalidRecipe {
				t.Fatalf("validatePackageDirectory() = %#v", err)
			}
			if strings.Contains(packageErr.Safe, "sensitive") {
				t.Fatalf("validation error = %v", err)
			}
		})
	}
}

func TestPackageValidationRejectsAHiddenSecondConfigDocument(t *testing.T) {
	const secretCanary = "second-document-secret-canary"
	directory := copyMaintainedRecipe(t, "accuracy")
	path := filepath.Join(directory, "config.yaml")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	secondDocument := []byte("\n---\nprivate: &private " + secretCanary + "\ncopy: *private\n")
	if writeErr := os.WriteFile(path, append(data, secondDocument...), 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}

	_, err = validatePackageDirectory(directory)
	packageErr, ok := AsPackageError(err)
	if !ok || packageErr.Code != ErrorInvalidRecipe {
		t.Fatalf("validatePackageDirectory() = %#v", err)
	}
	if strings.Contains(packageErr.Safe, secretCanary) || strings.Contains(packageErr.Safe, "private") {
		t.Fatalf("validation error leaked second-document content: %v", err)
	}
}

func TestCredentialDetectorRejectsEmbeddedURLCredentialsWithoutExposingThem(t *testing.T) {
	secret := "url-password-canary"
	for _, rawURL := range []string{
		"https://user:" + secret + "@example.com/v1",
		"https://example.com/v1?api_key=" + secret,
		"https://example.com/v1?access-token=" + secret,
		"https://example.com/v1?api_key=" + secret + ";foo=bar",
	} {
		config := []byte("provider:\n  base_url: " + rawURL + "\n")
		warnings := detectCredentialWarnings(config)
		if len(warnings) != 1 || warnings[0].Path != "/provider/base_url" {
			t.Fatalf("credential warnings = %#v", warnings)
		}
		if strings.Contains(warnings[0].Message, secret) {
			t.Fatal("credential warning exposed an embedded URL credential")
		}
	}
}

func TestPackageRuntimeValidationRejectsLocalProcessMCP(t *testing.T) {
	base := `
version: v0.3
global:
  model_catalog:
    modules:
      classifier:
        mcp:
          enabled: true
%s
          tool_name: classify
`
	tests := []struct {
		name  string
		body  string
		valid bool
	}{
		{name: "explicit stdio", body: "          transport_type: stdio\n          command: canary-do-not-run\n          args: [--canary]\n"},
		{name: "implicit stdio command", body: "          command: canary-do-not-run\n"},
		{name: "stdio env", body: "          transport_type: stdio\n          command: canary-do-not-run\n          env: {INHERITED: value}\n"},
		{name: "http", body: "          transport_type: http\n          url: https://example.com/mcp\n", valid: true},
		{name: "streamable http", body: "          transport_type: streamable-http\n          url: https://example.com/mcp\n", valid: true},
		{name: "implicit http", body: "          url: https://example.com/mcp\n", valid: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateRuntimeConfig([]byte(fmt.Sprintf(base, test.body)))
			if test.valid && err != nil {
				t.Fatalf("HTTP MCP rejected: %v", err)
			}
			if !test.valid && (err == nil || strings.Contains(err.Error(), "canary-do-not-run")) {
				t.Fatalf("local MCP validation error = %v", err)
			}
		})
	}
}

func TestRecipeEnvironmentBindingsRequireExplicitAvailableAllowlist(t *testing.T) {
	config := []byte(`
providers:
  models:
    - backend_refs:
        - api_key_env: OPENROUTER_API_KEY
          endpoint: ${MODEL_ENDPOINT}
          fallback: ${OPTIONAL_ENDPOINT:-https://example.com}
          escaped: $$NOT_A_BINDING
`)
	warnings := detectEnvironmentBindingWarnings(config)
	if len(warnings) != 3 {
		t.Fatalf("warnings = %#v", warnings)
	}
	for _, warning := range warnings {
		if warning.Code != ErrorEnvironmentBinding {
			t.Fatalf("warning = %#v", warning)
		}
	}
	if err := ValidateEnvironmentBindings(config); err == nil {
		t.Fatal("unbound Recipe environment was accepted")
	} else if packageErr, ok := AsPackageError(err); !ok || packageErr.Code != ErrorEnvironmentBinding || strings.Contains(packageErr.Safe, "test-secret") {
		t.Fatalf("binding error = %#v", err)
	}
	t.Setenv(recipeEnvironmentAllowlist, "MODEL_ENDPOINT,OPENROUTER_API_KEY,OPTIONAL_ENDPOINT")
	t.Setenv("MODEL_ENDPOINT", "model.internal:8000")
	t.Setenv("OPENROUTER_API_KEY", "test-secret")
	t.Setenv("OPTIONAL_ENDPOINT", "https://example.com")
	if err := ValidateEnvironmentBindings(config); err != nil {
		t.Fatalf("ValidateEnvironmentBindings() = %v", err)
	}

	reserved := []byte("value: ${PATH}\n")
	if _, err := extractEnvironmentBindings(reserved); err == nil {
		t.Fatal("reserved process environment binding was accepted")
	}
}

func TestActivationTargetRequiresExplicitEnvironmentWarningAcknowledgement(t *testing.T) {
	directory := copyMaintainedRecipe(t, "agent")
	archive := recipeZIPFromDirectory(t, directory, nil)
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(archive) }))
	defer server.Close()
	configPath := filepath.Join(t.TempDir(), "runtime-config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	store := NewStore(StoreOptions{Root: filepath.Join(t.TempDir(), "store"), ConfigPath: configPath, HTTPClient: server.Client()})
	installed, _, err := store.Import(context.Background(), ImportRequest{URL: server.URL + "/agent.zip"})
	if err != nil {
		t.Fatal(err)
	}
	if len(installed.Warnings) == 0 {
		t.Fatal("agent package did not report its environment binding warning")
	}
	for _, warning := range installed.Warnings {
		if warning.Code != ErrorEnvironmentBinding {
			t.Fatalf("unexpected warning = %#v", warning)
		}
	}
	if _, _, err := store.ActivationTarget(installed.RecipeDigest, false); err == nil {
		t.Fatal("activation target accepted without warning acknowledgement")
	} else if packageErr, ok := AsPackageError(err); !ok || packageErr.Code != ErrorWarningsNotAcknowledged {
		t.Fatalf("unacknowledged error = %#v", err)
	}
	if _, config, err := store.ActivationTarget(installed.RecipeDigest, true); err != nil || len(config) == 0 {
		t.Fatalf("acknowledged ActivationTarget() config=%d err=%v", len(config), err)
	}
}

func TestActivationTargetRejectsSymlinkedImmutableObjectDirectory(t *testing.T) {
	directory := copyMaintainedRecipe(t, "accuracy")
	archive := recipeZIPFromDirectory(t, directory, nil)
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(archive) }))
	defer server.Close()
	configPath := filepath.Join(t.TempDir(), "runtime-config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	store := NewStore(StoreOptions{Root: filepath.Join(t.TempDir(), "store"), ConfigPath: configPath, HTTPClient: server.Client()})
	installed, _, err := store.Import(context.Background(), ImportRequest{URL: server.URL + "/accuracy.zip"})
	if err != nil {
		t.Fatal(err)
	}
	objectDirectory, err := store.ObjectDirectory(installed.RecipeDigest)
	if err != nil {
		t.Fatal(err)
	}
	replacement := filepath.Join(t.TempDir(), "replacement-object")
	if err := os.Rename(objectDirectory, replacement); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(replacement, objectDirectory); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if _, _, err := store.ActivationTarget(installed.RecipeDigest, true); err == nil {
		t.Fatal("symlinked Recipe object directory accepted")
	}
}

type staticIPResolver struct {
	addresses []netip.Addr
	err       error
}

func (r staticIPResolver) LookupNetIP(context.Context, string, string) ([]netip.Addr, error) {
	return r.addresses, r.err
}

type zipEntryOverride struct {
	index  int
	header zip.FileHeader
}

func zipHeaderWithMode(name string, mode os.FileMode) zip.FileHeader {
	header := zip.FileHeader{Name: name, Method: zip.Store}
	header.SetMode(mode)
	return header
}

func recipeZIPWithOverride(t *testing.T, directory string, override zipEntryOverride) []byte {
	t.Helper()
	var buffer bytes.Buffer
	writer := zip.NewWriter(&buffer)
	for index, spec := range recipeFiles {
		data, err := os.ReadFile(filepath.Join(directory, spec.name))
		if err != nil {
			t.Fatal(err)
		}
		header := zip.FileHeader{Name: spec.name, Method: zip.Deflate}
		if index == override.index {
			header = override.header
		}
		entry, err := writer.CreateHeader(&header)
		if err != nil {
			t.Fatalf("CreateHeader(%s): %v", header.Name, err)
		}
		if _, err := entry.Write(data); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return buffer.Bytes()
}

func netipMustParse(raw string) netip.Addr {
	return netip.MustParseAddr(raw)
}
