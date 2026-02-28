#!/usr/bin/env node
// wasm_node_test.js — Integration test for signal-compiler.wasm in Node.js
//
// Usage:
//   node wasm_node_test.js [path/to/signal-compiler.wasm]
//
// Requirements: Node.js 18+

"use strict";

const fs = require("fs");
const path = require("path");
const { performance } = require("perf_hooks");

// __dirname = src/semantic-router/cmd/wasm → project root = 4 levels up
const projectRoot = path.join(__dirname, "..", "..", "..", "..");
const wasmPath = process.argv[2] ||
  path.join(projectRoot, "dashboard", "frontend", "public", "signal-compiler.wasm");
const execJsPath = process.argv[3] ||
  path.join(projectRoot, "dashboard", "frontend", "public", "wasm_exec.js");

let passed = 0;
let failed = 0;

function assert(cond, msg) {
  if (!cond) {
    failed++;
    console.error(`  FAIL: ${msg}`);
  } else {
    passed++;
  }
}

async function main() {
  // Check files exist
  if (!fs.existsSync(wasmPath)) {
    console.error(`WASM binary not found: ${wasmPath}`);
    console.error("Run 'make build' in cmd/wasm/ first.");
    process.exit(2);
  }
  if (!fs.existsSync(execJsPath)) {
    console.error(`wasm_exec.js not found: ${execJsPath}`);
    process.exit(2);
  }

  // Load Go WASM support
  require(execJsPath);
  const go = new Go();

  // Load and instantiate WASM
  const wasmBuffer = fs.readFileSync(wasmPath);
  const wasmModule = await WebAssembly.compile(wasmBuffer);
  const instance = await WebAssembly.instantiate(wasmModule, go.importObject);

  // Run the Go main() — registers global functions
  go.run(instance);

  // Wait a tick for initialization
  await new Promise((r) => setTimeout(r, 50));

  // Verify global functions are registered
  console.log("=== Test: Global function registration ===");
  assert(typeof globalThis.signalCompile === "function", "signalCompile should be a function");
  assert(typeof globalThis.signalValidate === "function", "signalValidate should be a function");
  assert(typeof globalThis.signalDecompile === "function", "signalDecompile should be a function");
  assert(typeof globalThis.signalFormat === "function", "signalFormat should be a function");

  // Test: Compile valid DSL
  console.log("\n=== Test: signalCompile (valid DSL) ===");
  const dslSource = `SIGNAL keyword intent { operator: "any" keywords: ["hello", "world"] threshold: 0.8 }

ROUTE r1 {
  PRIORITY 1
  WHEN keyword("intent")
  MODEL "qwen"
}

BACKEND vllm_endpoint b1 { address: "127.0.0.1" port: 8000 }`;

  const t0 = performance.now();
  const compileRaw = globalThis.signalCompile(dslSource);
  const compileTime = performance.now() - t0;
  console.log(`  Compile time: ${compileTime.toFixed(2)}ms`);

  const compileResult = JSON.parse(compileRaw);
  assert(compileResult.yaml !== "", "YAML output should not be empty");
  assert(compileResult.crd !== "", "CRD output should not be empty");
  assert(compileResult.yaml.includes("intent"), "YAML should contain signal name");
  assert(compileResult.yaml.includes("keyword"), "YAML should contain signal type");
  console.log(`  YAML length: ${compileResult.yaml.length} chars`);

  // Test: Compile invalid DSL
  console.log("\n=== Test: signalCompile (invalid DSL) ===");
  const badResult = JSON.parse(globalThis.signalCompile("INVALID !!!"));
  assert(badResult.error !== "", "Should return error for invalid DSL");

  // Test: Validate clean input
  console.log("\n=== Test: signalValidate (clean) ===");
  const t1 = performance.now();
  const validateRaw = globalThis.signalValidate(dslSource);
  const validateTime = performance.now() - t1;
  console.log(`  Validate time: ${validateTime.toFixed(2)}ms`);

  const validateResult = JSON.parse(validateRaw);
  assert(validateResult.errorCount === 0, `Expected 0 errors, got ${validateResult.errorCount}`);

  // Test: Validate with constraint violation
  console.log("\n=== Test: signalValidate (constraint violation) ===");
  const badDsl = `SIGNAL keyword s1 { keywords: ["test"] threshold: 2.0 }
ROUTE r1 {
  PRIORITY 1
  WHEN keyword("undefined_ref")
  MODEL "m1"
}
BACKEND vllm_endpoint b1 { address: "127.0.0.1" port: 8000 }`;
  const badValidate = JSON.parse(globalThis.signalValidate(badDsl));
  assert(badValidate.diagnostics.length > 0, "Should have diagnostics for bad input");

  // Test: Decompile YAML → DSL
  console.log("\n=== Test: signalDecompile ===");
  const decompileRaw = globalThis.signalDecompile(compileResult.yaml);
  const decompileResult = JSON.parse(decompileRaw);
  assert(decompileResult.dsl !== "", "Decompiled DSL should not be empty");
  assert(!decompileResult.error, "Decompile should not error: " + decompileResult.error);
  assert(decompileResult.dsl.includes("SIGNAL"), "Decompiled DSL should contain SIGNAL keyword");
  assert(decompileResult.dsl.includes("ROUTE"), "Decompiled DSL should contain ROUTE keyword");

  // Test: Decompile invalid YAML
  console.log("\n=== Test: signalDecompile (invalid YAML) ===");
  const badDecompile = JSON.parse(globalThis.signalDecompile("{{{bad"));
  assert(badDecompile.error !== "", "Should return error for invalid YAML");

  // Test: Format
  console.log("\n=== Test: signalFormat ===");
  const formatRaw = globalThis.signalFormat(dslSource);
  const formatResult = JSON.parse(formatRaw);
  assert(formatResult.dsl !== "", "Formatted DSL should not be empty");
  assert(!formatResult.error, "Format should not error: " + formatResult.error);

  // Test: Format idempotency
  const format2 = JSON.parse(globalThis.signalFormat(formatResult.dsl));
  assert(format2.dsl === formatResult.dsl, "Format should be idempotent after first pass");

  // Test: Round-trip (DSL → YAML → DSL → YAML)
  console.log("\n=== Test: Round-trip ===");
  const rtDecompile = JSON.parse(globalThis.signalDecompile(compileResult.yaml));
  const rtRecompile = JSON.parse(globalThis.signalCompile(rtDecompile.dsl));
  assert(rtRecompile.yaml === compileResult.yaml,
    "Round-trip should produce identical YAML");

  // Performance summary
  console.log("\n=== Performance ===");
  const wasmSize = fs.statSync(wasmPath).size;
  console.log(`  WASM binary size: ${(wasmSize / 1024 / 1024).toFixed(2)} MB`);
  console.log(`  Compile time: ${compileTime.toFixed(2)}ms (target: <5ms)`);
  console.log(`  Validate time: ${validateTime.toFixed(2)}ms (target: <1ms)`);

  // Verdict
  console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
