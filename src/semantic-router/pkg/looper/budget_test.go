/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package looper

import "testing"

func TestComputeBudgetIsZeroTreatsNilAsZero(t *testing.T) {
	var budget *ComputeBudget
	if !budget.IsZero() {
		t.Fatalf("nil budget should be zero")
	}
}

func TestComputeBudgetIsZeroTreatsUnsetFieldsAsZero(t *testing.T) {
	budget := &ComputeBudget{}
	if !budget.IsZero() {
		t.Fatalf("all-zero budget should be zero")
	}
}

func TestComputeBudgetIsZeroFalseWhenAnyLimitSet(t *testing.T) {
	cases := []ComputeBudget{
		{MaxPromptTokens: 1},
		{MaxCompletionTokens: 1},
		{MaxTotalTokens: 1},
		{MaxEstimatedCost: 0.01},
		{MaxWallTimeMs: 1},
	}
	for _, budget := range cases {
		if budget.IsZero() {
			t.Fatalf("budget %+v should not be zero", budget)
		}
	}
}
