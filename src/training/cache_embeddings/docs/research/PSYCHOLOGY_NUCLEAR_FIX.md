# Psychology NUCLEAR Strategy Fix

**Date**: 2026-01-14
**Issue**: Psychology negatives were too unrelated (jumping across subfields)
**Status**: ✅ FIXED - Updated to match medical/programming pattern

---

## Problem

The original psychology NUCLEAR strategy forced negatives to jump to completely different psychology subfields:

```
❌ OLD STRATEGY (TOO STRICT):
Anchor:   "Which signs point to an anxiety disorder?" (Clinical Psychology)
Negative: "How does the prefrontal cortex contribute to decision-making?" (Neuropsychology)
```

**User feedback**: "the negative should be related but not match the original, here it seems completely unrelated"

**Example provided (medical)**:
```
✅ CORRECT PATTERN:
Anchor:   "What are diagnostic methods for whooping cough?"
Positive: "How to diagnose Pertussis?"
Negative: "What are symptoms of Pertussis?"  # Same disease, different aspect
```

---

## Solution

Updated psychology NUCLEAR strategy to match medical/programming pattern:
- **Keep the same topic/entity** (e.g., anxiety disorder, working memory)
- **Change the aspect** (symptoms → causes → treatment → risk factors)
- **Don't jump to unrelated topics**

### NEW STRATEGY

**Step 1: Identify the Topic**
- Extract core topic: "anxiety disorder" → topic is "anxiety"
- Extract core topic: "working memory" → topic is "memory systems"

**Step 2: Shift the Aspect**
If anchor asks about → Negative asks about:
- Symptoms/Signs → Causes, Risk factors, Treatment, Prevention
- Diagnosis → Treatment, Prognosis, Epidemiology
- Treatment → Side effects, Efficacy, Contraindications
- Causes → Symptoms, Diagnosis, Prevention

**Step 3: Stay Related But Distinct**

✅ **DO**:
- "symptoms of anxiety disorder" → "treatment options for anxiety disorder"
- "symptoms of anxiety disorder" → "risk factors for developing anxiety"
- "working memory function" → "brain regions involved in working memory"

❌ **DON'T**:
- Jump to unrelated: "anxiety disorder" → "groupthink"
- Just reword: "symptoms of anxiety" → "signs of anxiety"

---

## Examples

### Example 1: Anxiety Disorder

```yaml
Anchor:   "What are symptoms of generalized anxiety disorder?"

Bad negatives:
  - "What are signs of generalized anxiety disorder?"  # Just rewording
  - "How does groupthink affect decision making?"      # Unrelated topic

Good negatives:
  - "What are the risk factors for developing generalized anxiety disorder?"
  - "How is generalized anxiety disorder treated with cognitive behavioral therapy?"
  - "What neurobiological mechanisms contribute to anxiety disorders?"
```

### Example 2: Working Memory

```yaml
Anchor:   "How does working memory function?"

Bad negatives:
  - "How does working memory operate?"           # Just rewording
  - "At what age do children develop theory of mind?"  # Unrelated topic

Good negatives:
  - "What brain regions are involved in working memory processing?"
  - "How does aging affect working memory performance?"
  - "What disorders impair working memory function?"
```

### Example 3: Operant Conditioning

```yaml
Anchor:   "What is operant conditioning?"

Bad negatives:
  - "What does operant conditioning mean?"         # Just rewording
  - "What role does dopamine play in reward processing?"  # Unrelated

Good negatives:
  - "How is operant conditioning applied in behavior therapy?"
  - "What are the neural mechanisms underlying operant conditioning?"
  - "When was operant conditioning first discovered and by whom?"
```

---

## Files Changed

### `src/training/cache_embeddings/domains/prompts.yaml`

**Lines 267-322**: Completely rewrote `negative_guidelines` for psychology
**Lines 324-359**: Updated `negative_examples` to match new pattern

**Key changes**:
1. Removed mandatory subfield jump requirement
2. Changed from "MAXIMUM separation" to "related but different aspect"
3. Updated verification questions to check for same topic, different aspect
4. Removed all examples of jumping across subfields

---

## Validation Plan

Before running full psychology generation (33,809 queries):

1. ✅ **Fix prompts.yaml** - DONE
2. ⏳ **Run small test** (5-10 queries) to validate negatives are "related but different aspect"
3. ⏳ **Manual review** of sample triplets
4. ⏳ **Run full generation** only after confirming negatives match expected pattern

---

## Expected Impact

### Before (OLD prompts - 1.5B model, wrong strategy)
- Generated: 12,049 triplets from 33,809 queries
- Augmentation: 0.36× (should be 5×)
- Placeholders: ~60% failure rate
- Negatives: Completely unrelated topics

### After (NEW prompts - 7B model, correct strategy)
- Expected: ~169,000 triplets from 33,809 queries
- Augmentation: ~5× (2 paraphrases × 3 negatives per query)
- Placeholders: <5% failure rate (validated on other domains)
- Negatives: Related topic, different aspect (matches medical pattern)

---

## Architecture Impact

This fix ensures psychology domain follows the same NUCLEAR pattern as medical and programming:
- **Medical**: Same disease, different aspect (diagnosis vs symptoms)
- **Programming**: Same library/framework, different feature (margins vs fonts)
- **Psychology**: Same disorder/concept, different aspect (symptoms vs treatment)

All domains now maintain **semantic relevance** while ensuring **aspect diversity** for effective contrastive learning.
