# DSL Direct Deploy to Router — Design Document

> **Status**: Draft  
> **Author**: haowu1234  
> **Date**: 2026-02-28  
> **Branch**: `feat/next`

---

## 1. Motivation

The current Builder/DSL Editor workflow has a broken feedback loop:

```
Builder UI ──WASM──→ DSL Parse → Compile → YAML/CRD
                                                │
                                          ❌ GAP
                                                │
Router   ←── fsnotify ←── config.yaml ←──── Manual copy-paste
```

Users must manually copy the compiled YAML, update `config.yaml`, and wait for fsnotify to trigger a hot-reload. This is error-prone and slow. We need a **one-click Deploy** from the Builder that closes this loop end-to-end.

---

## 2. Current Architecture Summary

### 2.1 DSL Package (`pkg/dsl/`)

| File | Purpose |
|------|---------|
| `parser.go` | Lexer (participle) + Parser → `rawProgram` → `Program` AST |
| `ast.go` | Two-layer AST: raw parse tree + resolved AST |
| `ast_json.go` | AST → JSON serialization for Visual Builder |
| `compiler.go` | AST → `config.RouterConfig` (13 signal types, 12 algo types, 12 plugin types) |
| `decompiler.go` | `config.RouterConfig` → DSL text (with auto plugin template extraction) |
| `emitter_yaml.go` | `RouterConfig` → UserYAML / CRD / Helm output backends |
| `validator.go` | 3-level validation (syntax / reference / constraint) + QuickFix + SymbolTable |
| `cli.go` | CLI wrappers: `CLICompile`, `CLIDecompile`, `CLIValidate`, `CLIFormat` |

### 2.2 WASM Bridge (Browser-side compilation)

5 JS global functions registered from `cmd/wasm/main_wasm.go`:

| Function | Output |
|----------|--------|
| `signalCompile(dsl)` | `{ yaml, crd, ast, diagnostics, error }` |
| `signalValidate(dsl)` | `{ diagnostics, symbols, errorCount }` |
| `signalDecompile(yaml)` | `{ dsl }` |
| `signalFormat(dsl)` | `{ dsl }` |
| `signalParseAST(dsl)` | `{ ast, diagnostics, symbols }` |

### 2.3 Router Hot-Reload

**Path A — File-based (fsnotify):**
```
config.yaml modified → fsnotify event → 250ms debounce + 300ms delay
    → NewOpenAIRouter(configPath)
    → service.Swap(newRouter)     // atomic.Pointer, lock-free
```

**Path B — Kubernetes CRD:**
```
CRD updated → Reconciler (5s poll) → config.Replace()
    → configUpdateCh → watchKubernetesConfigUpdates()
    → NewOpenAIRouter() + Swap()
```

### 2.4 Existing Config API

| Endpoint | Method | Status |
|----------|--------|--------|
| `/api/router/config/all` | GET | ✅ Read config.yaml as JSON |
| `/api/router/config/update` | POST/PUT | ✅ Deep-merge JSON → validate → write config.yaml |
| `/api/router/config/defaults` | GET | ✅ Read router-defaults.yaml |
| `/api/router/config/defaults/update` | POST/PUT | ✅ Update router-defaults.yaml |
| `/config/classification` (core API) | GET/PUT | ❌ 501 Not Implemented |

### 2.5 Missing Pieces

1. **No Deploy button** — Builder only has Compile; output is display-only
2. **No DSL HTTP API** — All compilation happens client-side (WASM)
3. **No `userYaml` in WASM output** — `signalCompile` emits flat `RouterConfig` YAML, not the user-friendly nested format that `config.yaml` uses
4. **No config backup/rollback** — `UpdateConfigHandler` overwrites without backup
5. **No deploy confirmation** — No feedback on whether Router successfully reloaded

---

## 3. Design Overview

### 3.1 End-to-End Data Flow

```
┌───────────────────── Frontend (Browser) ─────────────────────────┐
│                                                                   │
│  Builder UI ──→ DSL Source                                       │
│       │                                                           │
│       ▼                                                           │
│  WASM signalCompile()                                            │
│       │                                                           │
│       ├──→ diagnostics (display in Problems panel)               │
│       ├──→ yaml / crd (display in Output panel)                  │
│       └──→ userYaml (NEW — for deploy)                           │
│                │                                                  │
│       [Deploy Button] ── click ──→ Deploy Confirmation Dialog    │
│                │                    (diff preview + change stats) │
│                ▼                                                  │
│       POST /api/router/config/deploy                             │
│       Body: { yaml: userYaml, dsl: dslSource, message: "..." }  │
│                │                                                  │
│       Deploy Progress Panel                                      │
│       ✅ Compiled → ✅ Validated → ✅ Backed up → ⏳ Reloading  │
│                                                                   │
└──────────────────────────┼────────────────────────────────────────┘
                           │
                           ▼
┌───────────────── Dashboard Backend ──────────────────────────────┐
│                                                                   │
│  DeployHandler (NEW)                                             │
│    ├─ 1. Parse YAML → routerconfig.Parse() server-side validate  │
│    ├─ 2. Backup config.yaml → .vllm-sr/backups/                  │
│    ├─ 3. Archive DSL → .vllm-sr/config.dsl                       │
│    ├─ 4. Atomic write config.yaml (temp + rename)                │
│    ├─ 5. Poll /api/status for reload confirmation (max 5s)       │
│    └─ 6. Return result { status, version, reloadConfirmed }     │
│                           │                                       │
│  fsnotify detects write ──┘                                       │
│                           │                                       │
│                           ▼                                       │
│  ExtProc Server                                                   │
│    NewOpenAIRouter(configPath)                                    │
│    service.Swap(newRouter)                                        │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### 3.2 Why Not Reuse `UpdateConfigHandler`

| Aspect | `UpdateConfigHandler` | `DeployHandler` (new) |
|--------|----------------------|----------------------|
| Input | Partial JSON (deep merge) | Full YAML (complete replace) |
| Source | Manual field edits | DSL compilation |
| Backup | None | Versioned backups |
| DSL archive | N/A | Saves `.dsl` source |
| Deploy confirm | None | Polls router health |
| Rollback | Not supported | Supported |
| Semantics | "Patch a few fields" | "Deploy a complete configuration" |

---

## 4. Detailed Design

### 4.1 WASM Extension — `userYaml` Output

**File**: `src/semantic-router/cmd/wasm/main_wasm.go`

The `signalCompile` function currently returns `yaml` (flat `RouterConfig` format) and `crd`. We add a third output `userYaml` using the existing `EmitUserYAMLOrdered()` function:

```go
// In signalCompile handler, after existing yaml/crd emission:
userYamlBytes, userYamlErr := dsl.EmitUserYAMLOrdered(cfg)
if userYamlErr == nil {
    result["userYaml"] = string(userYamlBytes)
}
```

**`EmitUserYAMLOrdered`** already:
- Converts flat `keyword_rules` → nested `signals.keywords`
- Converts flat `vllm_endpoints` + `model_config` → nested `providers.models[].endpoints[]`
- Prunes zero-value infrastructure fields
- Orders sections: signals → decisions → providers → observability → strategy

This matches the exact format of `config.yaml` that the Router consumes.

**Frontend type update** (`types/dsl.ts`):

```typescript
export interface CompileResult {
  yaml: string
  crd: string
  userYaml: string    // NEW
  ast: ASTProgram
  diagnostics: Diagnostic[]
  error?: string
}
```

### 4.2 Dashboard Backend — Deploy API

#### 4.2.1 New Endpoints

**File**: `dashboard/backend/handlers/deploy.go` (new)

```
POST /api/router/config/deploy
POST /api/router/config/rollback
GET  /api/router/config/versions
GET  /api/router/config/deploy/status
```

#### 4.2.2 Deploy Handler

```go
// POST /api/router/config/deploy
//
// Request:
//   {
//     "yaml":    "<user-yaml-string>",
//     "dsl":     "<original-dsl-source>",       // optional, for audit
//     "message": "Added jailbreak detection"     // optional, deploy note
//   }
//
// Response (success):
//   {
//     "status":           "success",
//     "version":          "20260228-200134",
//     "backup_path":      ".vllm-sr/backups/config.20260228-200134.yaml",
//     "reload_confirmed": true,
//     "reload_time_ms":   1243
//   }
//
// Response (validation failure):
//   {
//     "status":  "validation_failed",
//     "error":   "failed to parse config: ...",
//     "details": ["signal 'foo' referenced but not defined", ...]
//   }
//
// Response (reload timeout):
//   {
//     "status":           "deployed_unconfirmed",
//     "version":          "20260228-200134",
//     "reload_confirmed": false,
//     "warning":          "Config written but router reload not confirmed within 5s"
//   }
```

**Implementation pseudocode:**

```go
func DeployHandler(configPath string, readonlyMode bool) http.HandlerFunc {
    var deployMu sync.Mutex  // serialize deploys

    return func(w http.ResponseWriter, r *http.Request) {
        // 0. Guards
        if readonlyMode { return 403 }
        if !deployMu.TryLock() { return 409 "Deploy already in progress" }
        defer deployMu.Unlock()

        // 1. Parse request
        var req DeployRequest
        json.NewDecoder(r.Body).Decode(&req)

        // 2. Validate YAML via routerconfig.Parse()
        tmpFile := writeTempFile(req.YAML)
        cfg, err := routerconfig.Parse(tmpFile)
        if err != nil { return 400 validation error }

        // 3. Create versioned backup
        version := time.Now().Format("20060102-150405")
        backupDir := filepath.Join(filepath.Dir(configPath), ".vllm-sr", "backups")
        os.MkdirAll(backupDir, 0755)
        copyFile(configPath, filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", version)))

        // 4. Archive DSL source (if provided)
        if req.DSL != "" {
            dslPath := filepath.Join(filepath.Dir(configPath), ".vllm-sr", "config.dsl")
            os.WriteFile(dslPath, []byte(req.DSL), 0644)
        }

        // 5. Atomic write config.yaml
        tmpConfig := configPath + ".tmp"
        os.WriteFile(tmpConfig, []byte(req.YAML), 0644)
        os.Rename(tmpConfig, configPath)  // atomic on same filesystem

        // 6. Wait for reload confirmation
        confirmed, reloadTime := pollRouterHealth(5 * time.Second)

        // 7. Save deploy metadata
        saveDeploy Metadata(version, req.Message, confirmed)

        // 8. Return result
        return DeployResponse{
            Status:          "success",
            Version:         version,
            ReloadConfirmed: confirmed,
            ReloadTimeMs:    reloadTime,
        }
    }
}
```

#### 4.2.3 Rollback Handler

```go
// POST /api/router/config/rollback
//
// Request:
//   { "version": "20260228-200134" }   // specific version
//   { }                                 // latest backup (default)
//
// Behavior:
//   1. Read backup file
//   2. Validate via routerconfig.Parse()
//   3. Backup current config (as pre-rollback snapshot)
//   4. Atomic write → fsnotify → reload
//   5. Return result
```

#### 4.2.4 Versions Handler

```go
// GET /api/router/config/versions
//
// Response:
//   {
//     "versions": [
//       {
//         "version":    "20260228-200134",
//         "timestamp":  "2026-02-28T20:01:34+08:00",
//         "message":    "Added jailbreak detection",
//         "size_bytes": 4523,
//         "source":     "dsl_deploy"    // or "manual_edit" or "rollback"
//       },
//       ...
//     ],
//     "current_version": "20260228-200134",
//     "max_versions":    10
//   }
```

#### 4.2.5 Route Registration

**File**: `dashboard/backend/router/router.go`

```go
// Add after existing config routes:
mux.HandleFunc("/api/router/config/deploy",
    handlers.DeployHandler(configPath, readonlyMode))
mux.HandleFunc("/api/router/config/rollback",
    handlers.RollbackHandler(configPath, readonlyMode))
mux.HandleFunc("/api/router/config/versions",
    handlers.VersionsHandler(configPath))
```

### 4.3 Frontend — Deploy Flow

#### 4.3.1 DSL Store Extension

**File**: `dashboard/frontend/src/stores/dslStore.ts`

New actions:

```typescript
interface DSLStore {
  // ... existing ...

  // Deploy
  deployStatus: 'idle' | 'previewing' | 'deploying' | 'success' | 'error'
  deployResult: DeployResult | null
  deployError: string | null

  // Actions
  deployPreview: () => Promise<DeployPreview>
  deploy: (message?: string) => Promise<DeployResult>
  rollback: (version?: string) => Promise<DeployResult>
  fetchVersions: () => Promise<ConfigVersion[]>
}
```

#### 4.3.2 Deploy Preview (Diff)

Before deploying, show a diff of what will change:

```typescript
deployPreview: async () => {
  const state = get()

  // 1. Compile current DSL to get userYaml
  if (!state.compileResult?.userYaml) {
    await state.compile()
  }
  const newYaml = state.compileResult?.userYaml
  if (!newYaml) throw new Error('Compilation required before deploy')

  // 2. Fetch current running config
  const currentResp = await fetch('/api/router/config/all')
  const currentJson = await currentResp.json()

  // 3. Decompile current config to DSL for text diff
  const currentDsl = await signalDecompile(JSON.stringify(currentJson))

  // 4. Compute diff stats
  return {
    currentDsl: currentDsl.dsl,
    newDsl: state.dslSource,
    stats: computeDiffStats(currentDsl.dsl, state.dslSource),
  }
}
```

#### 4.3.3 Deploy Button & Confirmation Dialog

**In toolbar** (both `BuilderPage` and `DslEditorPage`):

```tsx
<button
  className={styles.deployButton}
  onClick={handleDeploy}
  disabled={
    deployStatus === 'deploying' ||
    !compileResult?.userYaml ||
    (compileResult?.diagnostics?.some(d => d.severity === 'error'))
  }
  title="Deploy compiled configuration to running router"
>
  <DeployIcon />
  Deploy
</button>
```

**Confirmation dialog flow:**

```
┌─────────────────────────────────────────────┐
│  Deploy Configuration                        │
│                                              │
│  You are about to replace the running        │
│  router configuration.                       │
│                                              │
│  ┌─── Changes ────────────────────────────┐ │
│  │  + 2 signals added (jailbreak, pii)    │ │
│  │  ~ 3 decisions modified                │ │
│  │  - 1 backend removed                   │ │
│  │  ~ Global settings changed             │ │
│  └────────────────────────────────────────┘ │
│                                              │
│  Deploy note (optional):                     │
│  ┌────────────────────────────────────────┐ │
│  │ Added security guardrails              │ │
│  └────────────────────────────────────────┘ │
│                                              │
│          [Cancel]    [Deploy Now]             │
└──────────────────────────────────────────────┘
```

**Progress panel (replaces dialog after clicking Deploy Now):**

```
┌─────────────────────────────────────────────┐
│  Deploying...                                │
│                                              │
│  ✅  Compiled successfully                   │
│  ✅  Server-side validation passed           │
│  ✅  Backup created (v20260228-200134)       │
│  ✅  Configuration written                   │
│  ⏳  Waiting for router reload...            │
│                                              │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  60%       │
└──────────────────────────────────────────────┘

         ↓ (after success)

┌─────────────────────────────────────────────┐
│  ✅ Deployed Successfully                    │
│                                              │
│  Version:      v20260228-200134              │
│  Reload time:  1.2s                          │
│  Router:       healthy                       │
│                                              │
│          [Close]    [View Config]             │
└──────────────────────────────────────────────┘
```

### 4.4 Backup & Versioning

#### Directory Structure

```
/app/
├── config.yaml                          # Active config (watched by fsnotify)
└── .vllm-sr/
    ├── config.dsl                       # Latest DSL source (audit trail)
    ├── router-config.yaml               # Generated router config
    ├── router-defaults.yaml             # Default values
    ├── deploy-history.json              # Deploy metadata log
    └── backups/
        ├── config.20260228-193012.yaml  # Backup before deploy v1
        ├── config.20260228-200134.yaml  # Backup before deploy v2
        ├── config.20260228-203045.yaml  # Backup before deploy v3
        └── ...                          # Max 10 versions, oldest pruned
```

#### Deploy History (`deploy-history.json`)

```json
[
  {
    "version": "20260228-200134",
    "timestamp": "2026-02-28T20:01:34+08:00",
    "source": "dsl_deploy",
    "message": "Added jailbreak detection",
    "backup_file": "backups/config.20260228-200134.yaml",
    "reload_confirmed": true,
    "reload_time_ms": 1243,
    "signals_count": 14,
    "decisions_count": 31,
    "models_count": 6
  }
]
```

---

## 5. Safety & High Availability

### 5.1 Validation Pipeline (Two-stage)

```
Stage 1 — Client-side (WASM, instant feedback):
  DSL → Parse → Validate (syntax + reference + constraint)
      → Compile → RouterConfig
  
  Blocks Deploy if: any DiagError exists

Stage 2 — Server-side (authoritative):
  YAML → routerconfig.Parse() → validateConfigStructure()
       → validateEndpointAddress()
  
  Blocks Deploy if: Parse returns error
```

### 5.2 Atomic Write

```go
// Write to temp file first, then rename (atomic on same filesystem)
tmpPath := configPath + ".deploy.tmp"
os.WriteFile(tmpPath, yamlBytes, 0644)
os.Rename(tmpPath, configPath)  // atomic replace
```

This ensures the Router never reads a half-written `config.yaml`.

### 5.3 Deploy Lock (Concurrency Control)

```go
var deployMu sync.Mutex

func DeployHandler(...) http.HandlerFunc {
    return func(w, r) {
        if !deployMu.TryLock() {
            http.Error(w, "Another deploy is in progress", 409)
            return
        }
        defer deployMu.Unlock()
        // ... proceed
    }
}
```

Only one deploy can execute at a time. Concurrent requests get `409 Conflict`.

### 5.4 Auto-Rollback on Reload Failure

```go
// After writing config.yaml, poll router health
confirmed := pollRouterHealth(5 * time.Second)

if !confirmed {
    // Router failed to reload — auto-rollback
    log.Warn("Router reload not confirmed, auto-rolling back")
    copyFile(backupPath, configPath)  // restore backup
    
    return DeployResponse{
        Status:  "rolled_back",
        Warning: "Router failed to reload within 5s. Config auto-rolled back.",
    }
}
```

### 5.5 Readonly Mode

All deploy/rollback endpoints respect the existing `readonlyMode` flag:

```go
if readonlyMode {
    return 403 { error: "readonly_mode", message: "..." }
}
```

The frontend disables the Deploy button when `readonlyMode` is detected (via existing `/api/settings` endpoint).

---

## 6. Kubernetes CRD Deploy Path

When `config_source: kubernetes`, the Deploy flow changes:

```
DSL → Compile → EmitCRD(cfg, name, namespace)
                         │
                         ▼
              POST /api/router/config/deploy-crd
                         │
                         ▼
              Dashboard Backend applies CRD:
                kubectl apply -f <crd.yaml>
                         │
                         ▼
              K8s Reconciler (5s poll) detects change
                         │
                         ▼
              config.Replace() → configUpdateCh
                         │
                         ▼
              watchKubernetesConfigUpdates()
              NewOpenAIRouter() + Swap()
```

The frontend auto-detects `config_source` from `/api/router/config/all` and switches between:
- **File mode**: `POST /api/router/config/deploy` (writes `config.yaml`)
- **K8s mode**: `POST /api/router/config/deploy-crd` (applies CRD via K8s API)

---

## 7. API Reference

### 7.1 Deploy Configuration

```
POST /api/router/config/deploy
Content-Type: application/json

{
  "yaml":    "<user-yaml-string>",       // Required. Full config YAML.
  "dsl":     "<dsl-source-string>",      // Optional. Archived for audit.
  "message": "Deploy note"               // Optional. Human-readable note.
}

Response 200:
{
  "status":           "success",
  "version":          "20260228-200134",
  "backup_path":      ".vllm-sr/backups/config.20260228-200134.yaml",
  "reload_confirmed": true,
  "reload_time_ms":   1243
}

Response 400:
{
  "status":  "validation_failed",
  "error":   "config validation error message"
}

Response 403:
{
  "error":   "readonly_mode",
  "message": "Dashboard is in read-only mode."
}

Response 409:
{
  "error":   "deploy_in_progress",
  "message": "Another deploy is already in progress."
}
```

### 7.2 Rollback Configuration

```
POST /api/router/config/rollback
Content-Type: application/json

{
  "version": "20260228-200134"    // Optional. Defaults to latest backup.
}

Response 200:
{
  "status":            "success",
  "rolled_back_to":    "20260228-200134",
  "reload_confirmed":  true
}

Response 404:
{
  "error": "Backup version not found"
}
```

### 7.3 List Config Versions

```
GET /api/router/config/versions

Response 200:
{
  "versions": [
    {
      "version":    "20260228-200134",
      "timestamp":  "2026-02-28T20:01:34+08:00",
      "source":     "dsl_deploy",
      "message":    "Added jailbreak detection",
      "size_bytes": 4523
    }
  ],
  "current_version": "20260228-200134",
  "max_versions":    10
}
```

---

## 8. Implementation Plan

### Phase 0 — Foundation (P0)

| Task | File(s) | Complexity |
|------|---------|------------|
| Add `userYaml` to WASM `signalCompile` output | `cmd/wasm/main_wasm.go` | Low |
| Update `CompileResult` type | `types/dsl.ts` | Low |
| Implement `DeployHandler` with validation + atomic write + backup | `dashboard/backend/handlers/deploy.go` (new) | Medium |
| Register deploy routes | `dashboard/backend/router/router.go` | Low |
| Add `deploy()` action to `dslStore` | `stores/dslStore.ts` | Low |
| Add Deploy button to Builder toolbar | `BuilderPage.tsx` | Low |
| Add Deploy confirmation dialog | `BuilderPage.tsx` or new component | Medium |

### Phase 1 — UX Polish (P1)

| Task | File(s) | Complexity |
|------|---------|------------|
| Diff preview (decompile current → compare) | `dslStore.ts`, new `DeployDialog` component | Medium |
| Deploy progress panel (step-by-step status) | New `DeployProgress` component | Medium |
| Rollback API + handler | `deploy.go` | Low |
| Versions list API + handler | `deploy.go` | Low |
| Version history panel in Builder | `BuilderPage.tsx` | Medium |

### Phase 2 — Advanced (P2)

| Task | File(s) | Complexity |
|------|---------|------------|
| K8s CRD deploy path | `deploy.go`, new `deploy_crd.go` | High |
| Auto-rollback on reload failure | `deploy.go` | Medium |
| Deploy webhook notifications | `deploy.go` | Low |
| Deploy audit log (persistent) | `deploy.go` | Low |

### Phase 3 — Future (P3)

| Task | Complexity |
|------|------------|
| Incremental config diff-patch (not full replace) | High |
| Multi-instance deploy (fleet management) | High |
| Deploy approval workflow (multi-user) | High |
| Canary deploy (partial traffic on new config) | Very High |

---

## 9. Testing Strategy

### Unit Tests

- `deploy.go`: Validation failure, backup creation, atomic write, version pruning, concurrent deploy rejection
- `dslStore.ts`: Deploy action state transitions, error handling
- WASM: `userYaml` output matches `config.yaml` format

### Integration Tests

```
1. Compile DSL → Deploy → Verify config.yaml written → Verify Router reloaded
2. Deploy invalid config → Verify rejection → Verify original config unchanged
3. Deploy → Rollback → Verify previous config restored
4. Concurrent deploy attempts → Verify only one succeeds
5. Deploy with readonly mode → Verify 403 response
```

### E2E Tests

```
1. Open Builder → Write DSL → Compile → Click Deploy → Verify success
2. Deploy → Check Dashboard stats updated → Playground routes correctly
3. Deploy bad config → Verify error dialog → Config unchanged
4. Deploy → Rollback → Verify previous behavior restored
```

---

## 10. Security Considerations

| Concern | Mitigation |
|---------|-----------|
| Unauthorized deploy | Readonly mode flag; future: RBAC with `authz` signal |
| Config injection | Server-side `routerconfig.Parse()` validates all fields |
| Path traversal in backup | `filepath.Clean()` + restrict to `.vllm-sr/backups/` |
| Denial of service | Deploy mutex + request body size limit (1MB) |
| Sensitive data in DSL | DSL archive stored locally (same security as config.yaml) |

---

## 11. Open Questions

1. **Should Deploy support partial updates (merge) or only full replacement?**
   - Current proposal: full replacement (DSL always describes the complete config)
   - Partial updates can still use the existing `/api/router/config/update` endpoint

2. **How many backup versions to retain?**
   - Proposed: 10 (configurable via environment variable)

3. **Should auto-rollback be opt-in or default?**
   - Proposed: opt-in (default: deploy + warn if reload unconfirmed)

4. **K8s CRD deploy: should it go through kubectl or K8s client-go?**
   - Proposed: client-go (dashboard backend already has K8s dependencies)
