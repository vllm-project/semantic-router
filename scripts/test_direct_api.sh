#!/bin/bash
# Direct Classifier API Test Script
# Tests fact-check and feedback classifiers via their dedicated endpoints
# Shows actual model confidence scores (softmax probabilities)

ROUTER_URL="${ROUTER_URL:-http://localhost:8080}"

echo "# Direct Classifier API Test Results"
echo "**Date:** $(date)"
echo "**Router:** $ROUTER_URL"
echo ""

# ============================================================================
# 1. FACT CHECK CLASSIFIER
# ============================================================================
echo "## 1. Fact Check Classifier (/api/v1/classify/fact-check)"
echo ""
echo "| Query | Expected | Got | Label | Confidence | Result |"
echo "|-------|----------|-----|-------|------------|--------|"

declare -a fact_check_tests=(
    # Needs fact check (factual queries)
    "What is the capital of France?|true"
    "How many planets are in our solar system?|true"
    "When did World War II end?|true"
    "Who invented the telephone?|true"
    "What is the speed of light?|true"
    "What is the population of China?|true"
    "Who wrote Romeo and Juliet?|true"
    "What year did the Titanic sink?|true"
    "Which country has the largest population?|true"
    "What is the chemical formula for water?|true"
    "What is the boiling point of water?|true"
    "Who won the Nobel Prize in Physics last year?|true"
    # No fact check needed (creative/subjective)
    "Write a poem about nature|false"
    "Tell me a story about a dragon|false"
    "What do you think about modern art?|false"
    "Create a Python function to sort a list|false"
    "Write a SQL query to join two tables|false"
    "Give me cooking tips|false"
    "How can I improve my writing?|false"
    "What's the best way to learn programming?|false"
)

fc_correct=0
fc_total=0
for test in "${fact_check_tests[@]}"; do
    IFS='|' read -r query expected <<< "$test"
    resp=$(curl -s -X POST "$ROUTER_URL/api/v1/classify/fact-check" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$query\"}" 2>/dev/null)

    if [[ -z "$resp" ]] || echo "$resp" | grep -q '"error"'; then
        echo "| ${query:0:45} | $expected | ERROR | - | - | ❌ |"
        fc_total=$((fc_total + 1))
        continue
    fi

    needs_fact_check=$(echo "$resp" | jq -r 'if .needs_fact_check == true then "true" elif .needs_fact_check == false then "false" else "null" end')
    label=$(echo "$resp" | jq -r '.label // "unknown"')
    confidence=$(echo "$resp" | jq -r '.confidence // 0')
    conf_pct=$(awk "BEGIN {printf \"%.1f%%\", $confidence * 100}")

    if [[ "$needs_fact_check" == "$expected" ]]; then
        fc_correct=$((fc_correct + 1))
        mark="✅"
    else
        mark="❌"
    fi
    fc_total=$((fc_total + 1))
    echo "| ${query:0:45} | $expected | $needs_fact_check | ${label:0:22} | $conf_pct | $mark |"
done

echo ""
fc_pct=$(awk "BEGIN {printf \"%.0f\", $fc_correct/$fc_total*100}")
echo "**Fact Check Accuracy: $fc_correct/$fc_total ($fc_pct%)**"
echo ""

# ============================================================================
# 2. FACT CHECK CLASSIFIER - MULTILINGUAL
# ============================================================================
echo "## 2. Fact Check Classifier - Multilingual"
echo ""
echo "| Lang | Query | Expected | Got | Confidence | Result |"
echo "|------|-------|----------|-----|------------|--------|"

declare -a ml_fact_check_tests=(
    # Chinese - Needs fact check
    "ZH|法国的首都是什么？|true"
    "ZH|世界上人口最多的国家是哪个？|true"
    "ZH|世界上最高的山是什么？|true"
    "ZH|地球上有多少个大洲？|true"
    "ZH|第二次世界大战什么时候结束？|true"
    "ZH|美国第一任总统是谁？|true"
    "ZH|光速是多少？|true"
    "ZH|水的化学式是什么？|true"
    "ZH|太阳系有多少颗行星？|true"
    # Chinese - No fact check needed
    "ZH|写一首关于自然的诗|false"
    "ZH|给我讲一个关于龙的故事|false"
    "ZH|创建一个Python函数来排序列表|false"
    "ZH|你对现代艺术有什么看法？|false"
    "ZH|哪种编程语言最好？|false"
    # English - Additional
    "EN|What is the GDP of Japan?|true"
    "EN|How many continents are there?|true"
    "EN|What is the longest river in Africa?|true"
    "EN|Write a short story about time travel|false"
    "EN|Which programming language is the best?|false"
    "EN|Do you prefer cats or dogs?|false"
)

ml_fc_correct=0
ml_fc_total=0
for test in "${ml_fact_check_tests[@]}"; do
    IFS='|' read -r lang query expected <<< "$test"
    resp=$(curl -s -X POST "$ROUTER_URL/api/v1/classify/fact-check" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$query\"}" 2>/dev/null)

    if [[ -z "$resp" ]] || echo "$resp" | grep -q '"error"'; then
        echo "| $lang | ${query:0:30} | $expected | ERROR | - | ❌ |"
        ml_fc_total=$((ml_fc_total + 1))
        continue
    fi

    needs_fact_check=$(echo "$resp" | jq -r 'if .needs_fact_check == true then "true" elif .needs_fact_check == false then "false" else "null" end')
    confidence=$(echo "$resp" | jq -r '.confidence // 0')
    conf_pct=$(awk "BEGIN {printf \"%.1f%%\", $confidence * 100}")

    if [[ "$needs_fact_check" == "$expected" ]]; then
        ml_fc_correct=$((ml_fc_correct + 1))
        mark="✅"
    else
        mark="❌"
    fi
    ml_fc_total=$((ml_fc_total + 1))
    echo "| $lang | ${query:0:30} | $expected | $needs_fact_check | $conf_pct | $mark |"
done

echo ""
ml_fc_pct=$(awk "BEGIN {printf \"%.0f\", $ml_fc_correct/$ml_fc_total*100}")
echo "**Multilingual Fact Check Accuracy: $ml_fc_correct/$ml_fc_total ($ml_fc_pct%)**"
echo ""

# ============================================================================
# 3. USER FEEDBACK CLASSIFIER
# ============================================================================
echo "## 3. User Feedback Classifier (/api/v1/classify/user-feedback)"
echo ""
echo "| Query | Expected | Got | Confidence | Result |"
echo "|-------|----------|-----|------------|--------|"

declare -a user_feedback_tests=(
    # Satisfied
    "Thank you, that was very helpful!|satisfied"
    "Perfect, that answers my question|satisfied"
    "Great explanation, thanks!|satisfied"
    "Excellent answer, I understand now|satisfied"
    "That's exactly what I needed|satisfied"
    # Need clarification
    "I don't understand your explanation|need_clarification"
    "Can you explain that in simpler terms?|need_clarification"
    "What do you mean by that?|need_clarification"
    "I'm confused, can you clarify?|need_clarification"
    "Could you elaborate on that?|need_clarification"
    # Wrong answer
    "That's incorrect, the answer should be different|wrong_answer"
    "Your previous response was wrong|wrong_answer"
    "That's not right, please try again|wrong_answer"
    "This answer is inaccurate|wrong_answer"
    "You made a mistake in your response|wrong_answer"
    # Want different
    "Can you give me another answer?|want_different"
    "I'd like a different approach|want_different"
    "Show me an alternative solution|want_different"
    "Can you try a different method?|want_different"
    "Give me a different perspective|want_different"
)

uf_correct=0
uf_total=0
for test in "${user_feedback_tests[@]}"; do
    IFS='|' read -r query expected <<< "$test"
    resp=$(curl -s -X POST "$ROUTER_URL/api/v1/classify/user-feedback" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$query\"}" 2>/dev/null)

    if [[ -z "$resp" ]] || echo "$resp" | grep -q '"error"'; then
        echo "| ${query:0:50} | $expected | ERROR | - | ❌ |"
        uf_total=$((uf_total + 1))
        continue
    fi

    feedback_type=$(echo "$resp" | jq -r '.feedback_type // "unknown"')
    confidence=$(echo "$resp" | jq -r '.confidence // 0')
    conf_pct=$(awk "BEGIN {printf \"%.1f%%\", $confidence * 100}")

    if [[ "$feedback_type" == "$expected" ]]; then
        uf_correct=$((uf_correct + 1))
        mark="✅"
    else
        mark="❌"
    fi
    uf_total=$((uf_total + 1))
    echo "| ${query:0:50} | $expected | $feedback_type | $conf_pct | $mark |"
done

echo ""
uf_pct=$(awk "BEGIN {printf \"%.0f\", $uf_correct/$uf_total*100}")
echo "**User Feedback Accuracy: $uf_correct/$uf_total ($uf_pct%)**"
echo ""

# ============================================================================
# 4. USER FEEDBACK CLASSIFIER - MULTILINGUAL
# ============================================================================
echo "## 4. User Feedback Classifier - Multilingual"
echo ""
echo "| Lang | Query | Expected | Got | Confidence | Result |"
echo "|------|-------|----------|-----|------------|--------|"

declare -a ml_feedback_tests=(
    # Chinese - Need clarification
    "ZH|我不明白你之前的回答|need_clarification"
    "ZH|你能用更简单的话解释吗？|need_clarification"
    "ZH|你这是什么意思？|need_clarification"
    "ZH|我对你的解释感到困惑|need_clarification"
    "ZH|你能澄清一下你说的话吗？|need_clarification"
    # Chinese - Wrong answer
    "ZH|这是不正确的，答案不同|wrong_answer"
    "ZH|你之前的回答是错的|wrong_answer"
    "ZH|这不对，请再试一次|wrong_answer"
    "ZH|这个答案不准确|wrong_answer"
    "ZH|你的回答有错误|wrong_answer"
    # Chinese - Want different
    "ZH|你能给我另一个答案吗？|want_different"
    "ZH|我想要一个不同的方法|want_different"
    "ZH|给我看一个替代方案|want_different"
    "ZH|你能尝试不同的方法吗？|want_different"
    "ZH|给我一个不同的视角|want_different"
    # Chinese - Satisfied
    "ZH|谢谢，这正是我需要的|satisfied"
    "ZH|完美，这回答了我的问题|satisfied"
    "ZH|很好的解释，谢谢！|satisfied"
    "ZH|这非常有帮助，谢谢|satisfied"
    "ZH|很好的答案，我现在明白了|satisfied"
)

ml_uf_correct=0
ml_uf_total=0
for test in "${ml_feedback_tests[@]}"; do
    IFS='|' read -r lang query expected <<< "$test"
    resp=$(curl -s -X POST "$ROUTER_URL/api/v1/classify/user-feedback" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$query\"}" 2>/dev/null)

    if [[ -z "$resp" ]] || echo "$resp" | grep -q '"error"'; then
        echo "| $lang | ${query:0:25} | $expected | ERROR | - | ❌ |"
        ml_uf_total=$((ml_uf_total + 1))
        continue
    fi

    feedback_type=$(echo "$resp" | jq -r '.feedback_type // "unknown"')
    confidence=$(echo "$resp" | jq -r '.confidence // 0')
    conf_pct=$(awk "BEGIN {printf \"%.1f%%\", $confidence * 100}")

    if [[ "$feedback_type" == "$expected" ]]; then
        ml_uf_correct=$((ml_uf_correct + 1))
        mark="✅"
    else
        mark="❌"
    fi
    ml_uf_total=$((ml_uf_total + 1))
    echo "| $lang | ${query:0:25} | $expected | $feedback_type | $conf_pct | $mark |"
done

echo ""
ml_uf_pct=$(awk "BEGIN {printf \"%.0f\", $ml_uf_correct/$ml_uf_total*100}")
echo "**Multilingual User Feedback Accuracy: $ml_uf_correct/$ml_uf_total ($ml_uf_pct%)**"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "## Summary"
echo ""
echo "| Classifier | Accuracy | Status |"
echo "|------------|----------|--------|"

for name_pct_correct_total in \
    "Fact Check (EN)|$fc_pct|$fc_correct|$fc_total" \
    "Fact Check (Multilingual)|$ml_fc_pct|$ml_fc_correct|$ml_fc_total" \
    "User Feedback (EN)|$uf_pct|$uf_correct|$uf_total" \
    "User Feedback (Multilingual)|$ml_uf_pct|$ml_uf_correct|$ml_uf_total"; do
    IFS='|' read -r name pct correct total <<< "$name_pct_correct_total"
    if [[ $pct -ge 90 ]]; then status="✅ Excellent"
    elif [[ $pct -ge 70 ]]; then status="✅ Good"
    elif [[ $pct -ge 50 ]]; then status="⚠️ Partial"
    else status="❌ Limited"; fi
    echo "| $name | $correct/$total ($pct%) | $status |"
done
