"""Original training utterances for terse feedback and contrastive intentions.

These are training examples, not a benchmark. Each translated pair is one
semantic family. All inputs are interpreted as the current user follow-up;
the classifier cannot recover conversational facts absent from that text.
"""

# Native Chinese punctuation is part of the training text.
# ruff: noqa: RUF001

import argparse
import hashlib
import json
from pathlib import Path

CASES = {
    "SAT": [
        ("All sorted here now.", "现在这边全都正常了。"),
        ("That was the missing piece.", "之前缺的就是这一环。"),
        (
            "Ran it on my example and got the expected output.",
            "我拿自己的例子跑过了，输出符合预期。",
        ),
        (
            "I can finally explain it back to someone else.",
            "这下我终于也能把它讲给别人听了。",
        ),
        ("No follow-up needed from me.", "我这边没有要继续追问的了。"),
        ("Approved; let's use this version.", "通过，就采用这个版本。"),
        ("That saved me a lot of work.", "这替我省了不少事。"),
        (
            "I had doubts before, but this checks out.",
            "之前我有些怀疑，不过核实下来确实没问题。",
        ),
        ("Please leave it exactly as it is.", "保持现在这样就可以了。"),
        ("We're aligned now.", "这次咱们的理解一致了。"),
        (
            "This fits our constraints and our audience.",
            "这个既符合我们的限制，也适合我们的读者。",
        ),
        ("You've given me enough to move ahead.", "你给的信息已经足够我往下推进了。"),
    ],
    "NEED_CLARIFICATION": [
        (
            "I can follow the recipe, but I don't know why it works.",
            "操作步骤我会照着做，但还不知道为什么有效。",
        ),
        ("What does x stand for in that formula?", "那个公式里的 x 代表什么？"),
        (
            "Help me unpack the assumption before we continue.",
            "继续之前先帮我把这个假设讲透。",
        ),
        (
            "Hold on, where did that denominator come from?",
            "等一下，这个分母是怎么来的？",
        ),
        (
            "Which part of the example illustrates that rule?",
            "例子里的哪一部分体现了那条规则？",
        ),
        (
            "I'm learning this for the first time; could you slow down?",
            "我第一次学这个，能讲慢一点吗？",
        ),
        (
            "I agree with the outcome, but cannot explain the mechanism.",
            "结果我认同，但机制还没搞清楚。",
        ),
        (
            "When you say it is stable, stable in what sense?",
            "你说它稳定，具体是哪个意义上的稳定？",
        ),
        (
            "I'm stuck on the notation rather than the answer itself.",
            "我卡在符号的含义上，而不是答案本身。",
        ),
        (
            "Could you illustrate the distinction using a concrete case?",
            "能用一个具体案例说明两者的区别吗？",
        ),
        (
            "There are two interpretations of your sentence in my mind. Which did you intend?",
            "你这句话我能理解成两种意思，你指的是哪一种？",
        ),
        (
            "Before applying it, I need a more intuitive explanation.",
            "实际使用之前，我需要一个更直观的解释。",
        ),
    ],
    "WRONG_ANSWER": [
        (
            "That function doesn't exist in the library version I specified.",
            "我指定的库版本里没有这个函数。",
        ),
        (
            "It returns an empty list even though the example has a match.",
            "例子里明明有匹配项，它却返回空列表。",
        ),
        ("You counted that column twice.", "你把那一列算了两遍。"),
        (
            "The quotation you supplied is not in the linked article.",
            "你给出的引文不在所链接的文章里。",
        ),
        ("You reversed the cause and the effect.", "你把原因和结果颠倒了。"),
        (
            "The names in your summary belong to different people.",
            "你总结中的名字对应的是不同的人。",
        ),
        ("This still crashes at the same line.", "它还是在同一行崩溃。"),
        (
            "You treated a monthly rate as an annual rate.",
            "你把月度数值当成年度数值用了。",
        ),
        (
            "The required field is absent from the JSON you produced.",
            "你生成的 JSON 缺少必填字段。",
        ),
        (
            "Your response violates the condition I explicitly stated.",
            "你的回复不符合我明确给出的条件。",
        ),
        (
            "That claim is invented. There is no such event in the timeline.",
            "这个说法是编造的，时间线里没有那件事。",
        ),
        (
            "It isn't a matter of style: the returned value is wrong.",
            "这不是风格问题，返回值本身就是错的。",
        ),
    ],
    "WANT_DIFFERENT": [
        (
            "Keep the conclusions and turn them into slides.",
            "保留结论，把它整理成演示文稿。",
        ),
        ("More casual voice, please.", "语气再随意一点吧。"),
        ("Drop the rhymes and try free verse.", "不用押韵了，试试自由诗。"),
        (
            "The code works, but I'd rather avoid that dependency.",
            "代码能用，不过我想避开这个依赖。",
        ),
        (
            "Both are sound; show me a third option with less setup.",
            "这两个都可行，再给个准备工作更少的选项。",
        ),
        (
            "Let's keep the facts and change the target audience to children.",
            "保留事实，把目标读者换成儿童。",
        ),
        (
            "I understood it. What I need now is a one-page handout.",
            "我已经理解了，现在需要的是一页讲义。",
        ),
        (
            "Can we achieve the same effect without using a table?",
            "不用表格能达到同样效果吗？",
        ),
        (
            "Use a warmer opening and a shorter closing.",
            "开头写得亲切一点，结尾短一些。",
        ),
        (
            "The estimate is reasonable. Explore a more ambitious plan too.",
            "估算挺合理的，也探讨一下更大胆的方案吧。",
        ),
        (
            "Give it another shot with a minimalist visual style.",
            "再试一版极简视觉风格。",
        ),
        (
            "Don't add more explanation; reorganize it as a checklist.",
            "不用增加解释，重新组织成检查清单即可。",
        ),
    ],
}


def rows():
    result = []
    for label, pairs in CASES.items():
        for index, pair in enumerate(pairs):
            for language, text in zip(["en", "zh"], pair, strict=True):
                key = hashlib.sha256(text.encode()).hexdigest()
                result.append(
                    {
                        "id": f"vela-feedback-contrast:{key}",
                        "group_id": f"vela-feedback-contrast:{label}:{index}",
                        "text": text,
                        "label": label,
                        "source": "authored_feedback_contrasts",
                        "language": language,
                        "length_bucket": "authored_short",
                        "label_provenance": "original training author judgment",
                    }
                )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite training data")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows())
    )


if __name__ == "__main__":
    main()
