import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import jsxA11y from "eslint-plugin-jsx-a11y";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist", "node_modules"] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
      "jsx-a11y": jsxA11y,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true },
      ],
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_" },
      ],
      // Temporarily relaxed rules for existing code - can be tightened later
      "@typescript-eslint/no-explicit-any": "warn",
      "@typescript-eslint/ban-ts-comment": "warn",
      "no-useless-escape": "warn",
      // Keep interactive controls reachable from the keyboard (#3515). These are
      // errors: a new static-element click handler fails lint rather than
      // relying on review to catch it. The files that still carry violations
      // from before the rules existed are listed in the override below.
      "jsx-a11y/click-events-have-key-events": "error",
      "jsx-a11y/no-static-element-interactions": "error",
    },
  },
  {
    // Keyboard-accessibility backlog from #3515. These files carried violations
    // before the two rules above existed; they stay at "warn" so CI is not
    // blocked on work this PR does not do, while every other file — and every
    // new file — is held to "error".
    //
    // This list only shrinks. Delete a path once its violations are fixed; do
    // not add one to quiet a new violation.
    //
    // 23 files, 10 reported violations when this list was written.
    files: [
      "src/components/ChatComposerAddMenu.tsx",
      "src/components/ChatTaskQueue.tsx",
      "src/components/ClawRoomTeamDetailsModal.tsx",
      "src/components/ExpressionBuilderCanvasEmptyState.tsx",
      "src/components/ExpressionBuilderInner.tsx",
      "src/components/ExpressionBuilderNodes.tsx",
      "src/components/ExpressionBuilderToolbox.tsx",
      "src/pages/ConfigPageConnectModelsDialogView.tsx",
      "src/pages/DslEditorPage.tsx",
      "src/pages/MLSetupBenchmarkStep.tsx",
      "src/pages/MLSetupPage.tsx",
      "src/pages/builderPageDashboardViews.tsx",
      "src/pages/builderPageFieldControls.tsx",
      "src/pages/builderPageGlobalSettingsObservabilitySections.tsx",
      "src/pages/builderPageGlobalSettingsRoutingSection.tsx",
      "src/pages/builderPageGuideDrawer.tsx",
      "src/pages/builderPageOutputPanel.tsx",
      "src/pages/builderPageRouteSharedControls.tsx",
      "src/pages/builderPageVisualShell.tsx",
      "src/pages/topology/components/CustomNodes/DecisionNode.tsx",
      "src/pages/topology/components/CustomNodes/PluginChainNode.tsx",
      "src/pages/topology/components/CustomNodes/SignalGroupNode.tsx",
      "src/pages/topology/components/ResultCard/ResultCard.tsx",
    ],
    rules: {
      "jsx-a11y/click-events-have-key-events": "warn",
      "jsx-a11y/no-static-element-interactions": "warn",
    },
  },
  {
    files: [
      "src/components/ExpressionBuilderNodes.tsx",
      "src/pages/*Support.tsx",
      "src/pages/builderPage*.tsx",
    ],
    rules: {
      "react-refresh/only-export-components": "off",
    },
  },
  {
    files: ["src/contexts/AuthContext.tsx"],
    rules: {
      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true, allowExportNames: ["useAuth"] },
      ],
    },
  }
);
