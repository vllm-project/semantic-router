---
name: binding-ffi
category: fragment
description: Builds and maintains native Rust/C bindings and FFI layers that connect router-side classifiers and signal evaluation to compiled model runtimes. Use when adding or modifying native model bindings, updating FFI interfaces, or changing how the router calls into compiled classifier code.
---

# Binding FFI

## Trigger

- The primary skill adds or changes native model/runtime behavior

## Required Surfaces

- `native_binding`

## Conditional Surfaces

- `signal_runtime`
- `local_e2e`

## Stop Conditions

- Native code cannot be compiled or validated in the current environment

## Workflow

1. Read the Rust bindings playbook to understand the FFI interface patterns
2. Modify native binding code or FFI layer interfaces
3. Run `make test-binding-minimal` to validate basic binding functionality
4. Run `make test-binding-lora` if LoRA-related bindings are affected
5. Verify binding interface and router call sites stay aligned

## Must Read

- [docs/agent/playbooks/rust-bindings.md](../../../../docs/agent/playbooks/rust-bindings.md)

## Standard Commands

- `make test-binding-minimal`
- `make test-binding-lora`

## Acceptance

- Binding interface and router call sites stay aligned
