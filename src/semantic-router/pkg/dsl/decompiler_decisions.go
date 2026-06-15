package dsl

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (d *decompiler) decompileCandidateIteration(iter config.CandidateIterationConfig) {
	source := iter.Source
	if source == "models" {
		source = decompileCandidateIterationModelSource(iter.Models)
	}
	d.write("  FOR %s IN %s {\n", sanitizeName(iter.Variable), source)
	for _, output := range iter.Outputs {
		if output.Type == "model" {
			d.write("    MODEL %s\n", sanitizeName(output.Value))
		}
	}
	d.write("  }\n")
}

func decompileCandidateIterationModelSource(models []config.ModelRef) string {
	if len(models) == 0 {
		return "[]"
	}
	var sb strings.Builder
	sb.WriteString("[")
	for i, model := range models {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(strconv.Quote(model.Model))
		opts := candidateIterationModelRefOptions(&model)
		if opts != "" {
			sb.WriteString(" (")
			sb.WriteString(opts)
			sb.WriteString(")")
		}
	}
	sb.WriteString("]")
	return sb.String()
}

func candidateIterationModelRefOptions(model *config.ModelRef) string {
	var opts []string
	if model.UseReasoning != nil {
		opts = append(opts, fmt.Sprintf("reasoning = %t", *model.UseReasoning))
	}
	if model.ReasoningEffort != "" {
		opts = append(opts, fmt.Sprintf("effort = %q", model.ReasoningEffort))
	}
	if model.LoRAName != "" {
		opts = append(opts, fmt.Sprintf("lora = %q", model.LoRAName))
	}
	if model.Weight != 0 {
		opts = append(opts, fmt.Sprintf("weight = %s", strconv.FormatFloat(model.Weight, 'f', -1, 64)))
	}
	return strings.Join(opts, ", ")
}

func candidateIterationsCoverModelRefs(dec config.Decision) bool {
	// The MODEL omission optimization is only proven for one explicit-model
	// iteration that emits MODEL <iterator>. Multiple iterations require a
	// merge/order/dedup contract before they can safely cover ModelRefs.
	if len(dec.CandidateIterations) != 1 {
		return false
	}
	iter := dec.CandidateIterations[0]
	if iter.Source != "models" || !iterEmitsVariable(iter) {
		return false
	}
	if len(dec.ModelRefs) != len(iter.Models) {
		return false
	}
	for i := range dec.ModelRefs {
		if dec.ModelRefs[i].Model != iter.Models[i].Model ||
			dec.ModelRefs[i].LoRAName != iter.Models[i].LoRAName {
			return false
		}
	}
	return true
}

func decompileRuleNode(node *config.RuleCombination) string {
	if node == nil {
		return ""
	}

	// Leaf node — signal reference
	if node.Type != "" {
		return fmt.Sprintf("%s(%q)", node.Type, node.Name)
	}

	switch node.Operator {
	case "AND":
		// Flatten nested ANDs into a flat list: a AND b AND c
		parts := flattenRuleNode(node, "AND")
		return strings.Join(parts, " AND ")
	case "OR":
		// Flatten nested ORs into a flat list: (a OR b OR c)
		parts := flattenRuleNode(node, "OR")
		return "(" + strings.Join(parts, " OR ") + ")"
	case "NOT":
		if len(node.Conditions) == 1 {
			inner := decompileRuleNode(&node.Conditions[0])
			return "NOT " + inner
		}
	}

	// Fallback: join with operator
	parts := make([]string, 0, len(node.Conditions))
	for _, c := range node.Conditions {
		parts = append(parts, decompileRuleNode(&c))
	}
	if node.Operator != "" {
		return strings.Join(parts, " "+node.Operator+" ")
	}
	return strings.Join(parts, " AND ")
}

func flattenRuleNode(node *config.RuleCombination, op string) []string {
	if node.Operator == op {
		var parts []string
		for i := range node.Conditions {
			parts = append(parts, flattenRuleNode(&node.Conditions[i], op)...)
		}
		return parts
	}
	return []string{decompileRuleNode(node)}
}

func (d *decompiler) decompileAlgorithmFields(algo *config.AlgorithmConfig) string {
	if algo == nil {
		return ""
	}
	fields := d.algorithmToFields(algo)
	if len(fields) == 0 {
		return ""
	}

	var sb strings.Builder
	for _, key := range sortedKeys(fields) {
		fmt.Fprintf(&sb, "    %s: %s\n", key, formatDSLFieldValue(fields[key]))
	}
	return sb.String()
}

func formatDSLFieldValue(value Value) string {
	return formatPluginConfigValue(dslFieldValueFromValue(value).asInterface())
}

func decompileComposerObj(node *config.RuleCombination) string {
	if node == nil {
		return "{}"
	}
	if node.Type != "" {
		return fmt.Sprintf("{ type: %q, name: %q }", node.Type, node.Name)
	}
	var parts []string
	for i := range node.Conditions {
		parts = append(parts, decompileComposerObj(&node.Conditions[i]))
	}
	return fmt.Sprintf("{ operator: %q, conditions: [%s] }", node.Operator, strings.Join(parts, ", "))
}

func (d *decompiler) decompileDecisions() {
	for _, dec := range d.cfg.Decisions {
		d.decompileDecision(dec)
	}
}

func (d *decompiler) decompileDecision(dec config.Decision) {
	d.writeDecisionHeader(dec)
	d.write("  PRIORITY %d\n", dec.Priority)
	if dec.Tier != 0 {
		d.write("  TIER %d\n", dec.Tier)
	}
	if ruleExpr := decompileRuleNode(&dec.Rules); ruleExpr != "" {
		d.write("  WHEN %s\n", ruleExpr)
	}
	d.writeDecisionModels(dec)
	for _, iter := range dec.CandidateIterations {
		d.decompileCandidateIteration(iter)
	}
	d.writeDecisionAlgorithm(dec)
	d.writeDecisionPlugins(dec)
	for _, e := range dec.Emits {
		d.decompileEmit(e)
	}
	d.write("}\n\n")
}

func (d *decompiler) writeDecisionHeader(dec config.Decision) {
	if dec.Description != "" {
		d.write("ROUTE %s (description = %q) {\n", quoteName(dec.Name), dec.Description)
		return
	}
	d.write("ROUTE %s {\n", quoteName(dec.Name))
}

func (d *decompiler) writeDecisionModels(dec config.Decision) {
	if len(dec.ModelRefs) == 0 || candidateIterationsCoverModelRefs(dec) {
		return
	}
	d.write("  MODEL ")
	for i, mr := range dec.ModelRefs {
		if i > 0 {
			d.write(",\n        ")
		}
		d.write("%q", mr.Model)
		if opts := modelRefOptions(&mr, d.cfg.ModelConfig); opts != "" {
			d.write(" (%s)", opts)
		}
	}
	d.write("\n")
}

func (d *decompiler) writeDecisionAlgorithm(dec config.Decision) {
	if dec.Algorithm == nil || dec.Algorithm.Type == "" {
		return
	}
	d.write("  ALGORITHM %s", dec.Algorithm.Type)
	if algoFields := d.decompileAlgorithmFields(dec.Algorithm); algoFields != "" {
		d.write(" {\n%s  }\n", algoFields)
		return
	}
	d.write("\n")
}

func (d *decompiler) writeDecisionPlugins(dec config.Decision) {
	for _, p := range dec.Plugins {
		pluginFields := decompilePluginConfig(&p)
		if pluginFields != "" {
			d.write("  PLUGIN %s {\n%s  }\n", sanitizeName(p.Type), pluginFields)
			continue
		}
		d.write("  PLUGIN %s\n", sanitizeName(p.Type))
	}
}
