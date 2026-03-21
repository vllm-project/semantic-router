import React, {
  useEffect,
  useCallback,
  useState,
  useMemo,
  useRef,
} from "react";

import { useDSLStore } from "@/stores/dslStore";
import type { DSLFieldObject } from "@/types/dsl";
import type { EditorMode } from "@/types/dsl";
import type { RouteInput } from "@/lib/dslMutations";

import styles from "./BuilderPage.module.css";
import DslEditorPage from "./DslEditorPage";
import { BuilderDeployConfirmModal, BuilderDeployToast, BuilderDragOverlay } from "./builderPageDeployOverlays";
import { VisualMode } from "./builderPageVisualShell";
import { BuilderGuideDrawer } from "./builderPageGuideDrawer";
import { BuilderImportModal } from "./builderPageImportModal";
import { BuilderOutputPanel } from "./builderPageOutputPanel";
import { useResizableWidth } from "./builderPageResizeHooks";
import { BuilderStatusBar } from "./builderPageStatusBar";
import { BuilderToolbar } from "./builderPageToolbar";
import { useReadonly } from "@/contexts/ReadonlyContext";
import type { EntityKind, SectionState, Selection } from "./builderPageTypes";

// ---------- Component ----------

const BuilderPage: React.FC = () => {
  const {
    dslSource,
    diagnostics,
    symbols,
    ast,
    wasmReady,
    wasmError,
    loading,
    mode,
    dirty,
    yamlOutput,
    crdOutput,
    compileError,
    initWasm,
    compile,
    validate,
    parseAST,
    format,
    reset,
    setMode,
    importYaml,
    loadFromRouter,
    mutateModel,
    addModel,
    deleteModel,
    mutateSignal,
    addSignal,
    deleteSignal,
    mutatePlugin,
    addPlugin,
    deletePlugin,
    deleteRoute,
    mutateRoute,
    addRoute,
    requestDeploy,
    executeDeploy,
    dismissDeploy,
    deploying,
    deployStep,
    deployResult,
    showDeployConfirm,
    deployPreviewCurrent,
    deployPreviewMerged,
    deployPreviewLoading,
    deployPreviewError,
  } = useDSLStore();
  const { isReadonly, isLoading: readonlyLoading } = useReadonly();

  const [selection, setSelection] = useState<Selection | null>(null);
  const [sections, setSections] = useState<SectionState>({
    models: true,
    signals: true,
    routes: true,
    plugins: true,
  });
  const [addingEntity, setAddingEntity] = useState<EntityKind | null>(null);
  const [outputPanelOpen, setOutputPanelOpen] = useState(true);
  const [showImportModal, setShowImportModal] = useState(false);
  const [guideOpen, setGuideOpen] = useState(false);

  const contentRef = useRef<HTMLDivElement>(null);
  const { width: guideWidth, isDragging: isGuideDragging, handleDragStart: handleGuideDragStart } =
    useResizableWidth({
      initialWidth: 420,
      minWidth: 300,
      getMaxWidth: () => 800,
      stopPropagation: true,
    });
  const { width: outputWidth, isDragging, handleDragStart } = useResizableWidth({
    initialWidth: 380,
    minWidth: 200,
    getMaxWidth: () =>
      Math.floor((contentRef.current?.offsetWidth ?? window.innerWidth) * 0.6),
  });
  const [importText, setImportText] = useState("");
  const [importError, setImportError] = useState<string | null>(null);
  const [importUrl, setImportUrl] = useState("");
  const [importUrlLoading, setImportUrlLoading] = useState(false);
  const importTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const autoLoadedDefaultConfigRef = useRef(false);
  const autoLoadingDefaultConfigRef = useRef(false);

  // Initialize WASM on mount
  useEffect(() => {
    initWasm();
  }, [initWasm]);

  // Always land on the DSL editor when entering the builder page.
  useEffect(() => {
    setMode("dsl");
  }, [setMode]);

  // Parse AST when entering visual mode or when dslSource changes in visual mode
  useEffect(() => {
    if (mode === "visual" && wasmReady && dslSource.trim()) {
      parseAST();
    }
  }, [mode, wasmReady, dslSource, parseAST]);

  const toggleSection = useCallback((key: keyof SectionState) => {
    setSections((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const handleModeSwitch = useCallback(
    (newMode: EditorMode) => {
      setMode(newMode);
      // When switching to visual, parse AST
      if (newMode === "visual" && wasmReady && dslSource.trim()) {
        parseAST();
      }
    },
    [setMode, wasmReady, dslSource, parseAST],
  );
  const deployDisabled = readonlyLoading || isReadonly;

  // --- Entity CRUD handlers ---

  const handleDeleteEntity = useCallback(
    (kind: EntityKind, name: string, subType?: string) => {
      switch (kind) {
        case "model":
          deleteModel(name);
          break;
        case "signal":
          if (subType) deleteSignal(subType, name);
          break;
        case "route":
          deleteRoute(name);
          break;
        case "plugin":
          if (subType) deletePlugin(name, subType);
          break;
      }
      setSelection(null);
    },
    [deleteModel, deleteSignal, deleteRoute, deletePlugin],
  );

  const handleUpdateModelFields = useCallback(
    (name: string, fields: DSLFieldObject) => {
      mutateModel(name, fields);
    },
    [mutateModel],
  );

  const handleAddModel = useCallback(
    (name: string, fields: DSLFieldObject) => {
      addModel(name, fields);
      setSelection({ kind: "model", name });
      setAddingEntity(null);
    },
    [addModel],
  );

  const handleUpdateSignalFields = useCallback(
    (signalType: string, name: string, fields: DSLFieldObject) => {
      mutateSignal(signalType, name, fields);
    },
    [mutateSignal],
  );

  const handleUpdatePluginFields = useCallback(
    (name: string, pluginType: string, fields: DSLFieldObject) => {
      mutatePlugin(name, pluginType, fields);
    },
    [mutatePlugin],
  );

  const handleAddSignal = useCallback(
    (signalType: string, name: string, fields: DSLFieldObject) => {
      addSignal(signalType, name, fields);
      setSelection({ kind: "signal", name });
      setAddingEntity(null);
    },
    [addSignal],
  );

  const handleAddPlugin = useCallback(
    (name: string, pluginType: string, fields: DSLFieldObject) => {
      addPlugin(name, pluginType, fields);
      setSelection({ kind: "plugin", name });
      setAddingEntity(null);
    },
    [addPlugin],
  );

  const handleUpdateRoute = useCallback(
    (name: string, input: RouteInput) => {
      mutateRoute(name, input);
    },
    [mutateRoute],
  );

  const handleAddRoute = useCallback(
    (name: string, input: RouteInput) => {
      addRoute(name, input);
      setSelection({ kind: "route", name });
      setAddingEntity(null);
    },
    [addRoute],
  );

  // --- Import Config handlers ---

  const handleOpenImport = useCallback(() => {
    setImportText("");
    setImportError(null);
    setImportUrl("");
    setImportUrlLoading(false);
    setShowImportModal(true);
    setTimeout(() => importTextareaRef.current?.focus(), 50);
  }, []);

  const handleImportConfirm = useCallback(() => {
    const yaml = importText.trim();
    if (!yaml) {
      setImportError("Please paste YAML content");
      return;
    }
    try {
      importYaml(yaml);
      compile();
      setShowImportModal(false);
      setImportText("");
      setImportError(null);
    } catch {
      setImportError(
        "Failed to import YAML. Use a full router config or routing fragment; only the routing section is imported into DSL.",
      );
    }
  }, [importText, importYaml, compile]);

  const handleImportFile = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        const text = ev.target?.result;
        if (typeof text === "string") {
          setImportText(text);
          setImportError(null);
        }
      };
      reader.readAsText(file);
      e.target.value = "";
    },
    [],
  );

  const handleImportUrl = useCallback(async () => {
    const url = importUrl.trim();
    if (!url) {
      setImportError("Please enter a URL");
      return;
    }
    try {
      new URL(url);
    } catch {
      setImportError("Invalid URL format");
      return;
    }
    setImportUrlLoading(true);
    setImportError(null);
    try {
      const resp = await fetch("/api/tools/fetch-raw", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      const data = await resp.json();
      if (data.error) {
        throw new Error(data.error);
      }
      if (!data.content?.trim()) {
        throw new Error("Remote returned empty content");
      }
      setImportText(data.content);
      setImportError(null);
    } catch (err) {
      setImportError(
        `Failed to fetch: ${err instanceof Error ? err.message : String(err)}`,
      );
    } finally {
      setImportUrlLoading(false);
    }
  }, [importUrl]);

  const [loadingFromRouter, setLoadingFromRouter] = useState(false);
  const handleLoadFromRouter = useCallback(async () => {
    setLoadingFromRouter(true);
    setImportError(null);
    try {
      await loadFromRouter();
      compile();
      setShowImportModal(false);
      setImportText("");
    } catch (err) {
      setImportError(
        `Failed to load from router: ${err instanceof Error ? err.message : String(err)}`,
      );
    } finally {
      setLoadingFromRouter(false);
    }
  }, [loadFromRouter, compile]);

  const handleRequestDeploy = useCallback(() => {
    if (deployDisabled) {
      return;
    }
    requestDeploy();
  }, [deployDisabled, requestDeploy]);

  // On first entry, load current router config and compile it by default.
  useEffect(() => {
    if (
      !wasmReady ||
      readonlyLoading ||
      dslSource.trim() ||
      autoLoadedDefaultConfigRef.current ||
      autoLoadingDefaultConfigRef.current
    ) {
      return;
    }

    autoLoadingDefaultConfigRef.current = true;
    let cancelled = false;
    const loadDefaultConfig = async () => {
      setLoadingFromRouter(true);
      setImportError(null);
      try {
        await loadFromRouter();
        if (!cancelled) {
          compile();
          autoLoadedDefaultConfigRef.current = true;
        }
      } catch (err) {
        console.error(
          "[BuilderPage] Failed to load default router config:",
          err,
        );
      } finally {
        autoLoadingDefaultConfigRef.current = false;
        if (!cancelled) {
          setLoadingFromRouter(false);
        }
      }
    };
    void loadDefaultConfig();
    return () => {
      cancelled = true;
    };
  }, [wasmReady, readonlyLoading, dslSource, loadFromRouter, compile]);

  // Diagnostic counts
  const errorCount = diagnostics.filter((d) => d.level === "error").length;
  const modelCount = ast?.models?.length ?? symbols?.models?.length ?? 0;
  const signalCount = ast?.signals?.length ?? symbols?.signals?.length ?? 0;
  const routeCount = ast?.routes?.length ?? symbols?.routes?.length ?? 0;
  const pluginCount = ast?.plugins?.length ?? symbols?.plugins?.length ?? 0;
  const isValid = errorCount === 0 && wasmReady;
  const lineCount = dslSource.split("\n").length;

  // Memoize selected entity from AST
  const selectedEntity = useMemo(() => {
    if (!selection || !ast) return null;
    switch (selection.kind) {
      case "model":
        return ast.models?.find((m) => m.name === selection.name) ?? null;
      case "signal":
        return ast.signals?.find((s) => s.name === selection.name) ?? null;
      case "route":
        return ast.routes?.find((r) => r.name === selection.name) ?? null;
      case "plugin":
        return ast.plugins?.find((p) => p.name === selection.name) ?? null;
      default:
        return null;
    }
  }, [selection, ast]);

  return (
    <div className={styles.page}>
      <BuilderToolbar
        dirty={dirty}
        mode={mode}
        wasmReady={wasmReady}
        wasmError={wasmError}
        dslSource={dslSource}
        loading={loading}
        deploying={deploying}
        deployDisabled={deployDisabled}
        deployDisabledReason={
          isReadonly
            ? "Deploy is unavailable in read-only mode"
            : readonlyLoading
              ? "Checking deploy permissions..."
              : undefined
        }
        guideOpen={guideOpen}
        outputPanelOpen={outputPanelOpen}
        onModeSwitch={handleModeSwitch}
        onImport={handleOpenImport}
        onCompile={compile}
        onRequestDeploy={handleRequestDeploy}
        onFormat={format}
        onValidate={validate}
        onToggleGuide={() => setGuideOpen(!guideOpen)}
        onToggleOutput={() => setOutputPanelOpen(!outputPanelOpen)}
        onReset={reset}
      />

      {/* Main Content — editor + output panel */}
      <div className={styles.content} ref={contentRef}>
        {/* Editor area (switches by mode) */}
        <div className={styles.editorArea}>
          {mode === "visual" && (
            <VisualMode
              ast={ast}
              dslSource={dslSource}
              diagnostics={diagnostics}
              selection={selection}
              onSelect={setSelection}
              sections={sections}
              onToggleSection={toggleSection}
              selectedEntity={selectedEntity}
              modelCount={modelCount}
              signalCount={signalCount}
              routeCount={routeCount}
              pluginCount={pluginCount}
              wasmReady={wasmReady}
              wasmError={wasmError}
              addingEntity={addingEntity}
              onSetAddingEntity={setAddingEntity}
              onDeleteEntity={handleDeleteEntity}
              onUpdateModelFields={handleUpdateModelFields}
              onUpdateSignalFields={handleUpdateSignalFields}
              onUpdatePluginFields={handleUpdatePluginFields}
              onAddModel={handleAddModel}
              onAddSignal={handleAddSignal}
              onAddPlugin={handleAddPlugin}
              onUpdateRoute={handleUpdateRoute}
              onAddRoute={handleAddRoute}
              errorCount={errorCount}
              isValid={isValid}
              onModeSwitch={handleModeSwitch}
            />
          )}
          {mode === "dsl" && (
            <div className={styles.dslModeContainer}>
              <DslEditorPage embedded hideOutput />
            </div>
          )}
          {mode === "nl" && (
            <div className={styles.nlPlaceholder}>
              <div className={styles.nlPlaceholderIcon}>🤖</div>
              <div className={styles.nlPlaceholderTitle}>
                Natural Language Mode
              </div>
              <div>
                Describe your routing configuration in plain English and let AI
                generate DSL for you.
              </div>
              <div
                style={{
                  fontSize: "var(--text-xs)",
                  color: "var(--color-text-muted)",
                }}
              >
                Coming soon — Phase 6
              </div>
            </div>
          )}
        </div>

        <BuilderOutputPanel
          open={outputPanelOpen}
          width={outputWidth}
          yamlOutput={yamlOutput}
          crdOutput={crdOutput}
          dslSource={dslSource}
          compileError={compileError}
          onDragStart={handleDragStart}
          onOpen={() => setOutputPanelOpen(true)}
          onClose={() => setOutputPanelOpen(false)}
        />
      </div>

      <BuilderStatusBar
        isValid={isValid}
        errorCount={errorCount}
        modelCount={modelCount}
        signalCount={signalCount}
        routeCount={routeCount}
        pluginCount={pluginCount}
        lineCount={lineCount}
        mode={mode}
      />

      {/* Hidden file input for YAML import */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".yaml,.yml,.json"
        style={{ display: "none" }}
        onChange={handleImportFile}
      />

      <BuilderImportModal
        open={showImportModal}
        importUrl={importUrl}
        importText={importText}
        importError={importError}
        importUrlLoading={importUrlLoading}
        loadingFromRouter={loadingFromRouter}
        importTextareaRef={importTextareaRef}
        onClose={() => setShowImportModal(false)}
        onImportUrlChange={(value) => {
          setImportUrl(value);
          setImportError(null);
        }}
        onImportTextChange={(value) => {
          setImportText(value);
          setImportError(null);
        }}
        onImportUrl={handleImportUrl}
        onSelectFile={() => fileInputRef.current?.click()}
        onLoadFromRouter={handleLoadFromRouter}
        onConfirm={handleImportConfirm}
      />

      <BuilderGuideDrawer
        open={guideOpen}
        width={guideWidth}
        isDragging={isGuideDragging}
        onClose={() => setGuideOpen(false)}
        onDragStart={handleGuideDragStart}
        onInsertSnippet={(snippet) => {
          if (mode !== "dsl") setMode("dsl");
          const store = useDSLStore.getState();
          const src = store.dslSource;
          store.setDslSource(
            src ? src.trimEnd() + "\n\n" + snippet + "\n" : snippet + "\n",
          );
          setGuideOpen(false);
        }}
      />

      <BuilderDeployConfirmModal
        open={showDeployConfirm}
        loading={deployPreviewLoading}
        error={deployPreviewError}
        currentYaml={deployPreviewCurrent}
        mergedYaml={deployPreviewMerged}
        onClose={dismissDeploy}
        onConfirm={executeDeploy}
      />

      <BuilderDeployToast
        deploying={deploying}
        deployStep={deployStep}
        deployResult={deployResult}
        onDismiss={dismissDeploy}
      />

      <BuilderDragOverlay active={isDragging || isGuideDragging} />
    </div>
  );
};


export default BuilderPage;
