"""Original matched-intent training examples; never an independent benchmark.

Each topic and its translations share one group, including all five intentions.
The difference is whether the current text evaluates, clarifies or revises a
preceding answer, or supplies a new task. No absent conversation is invented.
"""

# Preserve the natural punctuation in bilingual training utterances.
# ruff: noqa: RUF001

import argparse
import hashlib
import json
from pathlib import Path

from .vela_contract import VELA_LABEL2ID

CASES = [
    {
        "NO_FEEDBACK": (
            "Which microscope setting controls the brightness of the image?",
            "显微镜的哪个设置控制图像亮度？",
        ),
        "SAT": (
            "The specimen is visible now. Your adjustment solved it.",
            "现在能看清标本了，你建议的调整解决了问题。",
        ),
        "NEED_CLARIFICATION": (
            "You mentioned numerical aperture. What exactly does that term mean here?",
            "你提到了数值孔径，这个词在这里具体是什么意思？",
        ),
        "WRONG_ANSWER": (
            "You told me to turn a knob that this microscope doesn't have.",
            "你让我转的那个旋钮，这台显微镜根本没有。",
        ),
        "WANT_DIFFERENT": (
            "Those instructions are clear; turn them into a card I can tape beside the microscope.",
            "操作说明很清楚，把它改成能贴在显微镜旁边的小卡片吧。",
        ),
    },
    {
        "NO_FEEDBACK": (
            "At what time of year should an apple orchard be pruned?",
            "苹果园一般在一年中的什么时候修剪？",
        ),
        "SAT": (
            "This gives me a workable pruning schedule. I'll follow it.",
            "这个修剪时间表可以实际执行，我就按它来。",
        ),
        "NEED_CLARIFICATION": (
            "Why does your pruning plan leave that branch in place?",
            "你的修剪方案为什么要保留那根枝条？",
        ),
        "WRONG_ANSWER": (
            "You read the tree age incorrectly; it is two years old, not twenty.",
            "你看错树龄了，是两年，不是二十年。",
        ),
        "WANT_DIFFERENT": (
            "Keep the same timing, but make a version for a smaller crew.",
            "保留同样的时间安排，再做一个适合更少人手的版本。",
        ),
    },
    {
        "NO_FEEDBACK": (
            "Draft an invoice for three consulting sessions at 80 euros each.",
            "拟一份发票，内容是三次咨询，每次八十欧元。",
        ),
        "SAT": (
            "The invoice totals now reconcile with my records.",
            "这次发票合计与我的记录对上了。",
        ),
        "NEED_CLARIFICATION": (
            "I can't tell why you put that amount in the tax column.",
            "我看不出你为什么把那笔金额放在税费栏。",
        ),
        "WRONG_ANSWER": (
            "You charged for four sessions although I specified three.",
            "我说的是三次，你却按四次收费了。",
        ),
        "WANT_DIFFERENT": (
            "The numbers are fine; switch the invoice to a landscape layout.",
            "数字没问题，把发票换成横向版式。",
        ),
    },
    {
        "NO_FEEDBACK": (
            "What distinguishes metamorphic rock from sedimentary rock?",
            "变质岩与沉积岩有什么区别？",
        ),
        "SAT": (
            "I can identify the samples with that explanation. Much appreciated.",
            "根据你的解释，我能辨认这些样本了，很感谢。",
        ),
        "NEED_CLARIFICATION": (
            "When you say recrystallization, are the minerals melting first? I'm confused.",
            "你说的重结晶，是矿物先熔化了吗？这点我没弄懂。",
        ),
        "WRONG_ANSWER": (
            "The sample in the photo is marble; you called it sandstone.",
            "照片里的样本是大理石，你却说成了砂岩。",
        ),
        "WANT_DIFFERENT": (
            "I understand the distinction. Present it as a dialogue between two museum guides.",
            "区别我理解了，请用两位博物馆讲解员对话的形式呈现。",
        ),
    },
    {
        "NO_FEEDBACK": (
            "How does room temperature affect a sourdough starter?",
            "室温怎样影响天然酵母？",
        ),
        "SAT": (
            "The starter rose after I followed those steps. It worked.",
            "照着这些步骤操作后酵母发起来了，确实有效。",
        ),
        "NEED_CLARIFICATION": (
            "I need help understanding the feeding ratio you used.",
            "我需要你帮我理解一下所用的喂养比例。",
        ),
        "WRONG_ANSWER": (
            "Your ingredient list omits the flour used in step two.",
            "你的配料表漏掉了第二步要用的面粉。",
        ),
        "WANT_DIFFERENT": (
            "That recipe is valid, but I'd prefer one that fits an overnight schedule.",
            "这个配方可行，不过我更想要适合隔夜操作的版本。",
        ),
    },
    {
        "NO_FEEDBACK": (
            "Why do violin players apply rosin to the bow?",
            "小提琴演奏者为什么给琴弓擦松香？",
        ),
        "SAT": (
            "That fixed the slipping bow. I'm happy with the result.",
            "琴弓打滑的问题解决了，我对结果很满意。",
        ),
        "NEED_CLARIFICATION": (
            "Could you clarify what you mean by a light coat in that instruction?",
            "能解释一下那条说明里的薄薄一层究竟指多少吗？",
        ),
        "WRONG_ANSWER": (
            "You answered about guitar strings when I asked about a violin bow.",
            "我问的是小提琴弓，你却回答了吉他弦。",
        ),
        "WANT_DIFFERENT": (
            "Keep your advice, but rewrite it in the voice of a patient music teacher.",
            "建议保留，但请改成耐心的音乐老师会使用的语气。",
        ),
    },
    {
        "NO_FEEDBACK": (
            "Calculate the orbital period of a satellite at the given altitude.",
            "计算给定高度下卫星的轨道周期。",
        ),
        "SAT": (
            "I checked the units and the result matches the reference. Good work.",
            "我核对了单位，结果也与参考值一致，做得不错。",
        ),
        "NEED_CLARIFICATION": (
            "Where did the cube in your equation come from?",
            "你那个等式里的三次方是怎么来的？",
        ),
        "WRONG_ANSWER": (
            "You used altitude instead of distance from the center in the formula.",
            "你在公式里把距中心的距离误用了高度。",
        ),
        "WANT_DIFFERENT": (
            "The derivation is sound; give me a graphical explanation of the same relationship.",
            "推导没问题，再用图形说明同一个关系吧。",
        ),
    },
    {
        "NO_FEEDBACK": (
            "Write a fictional parcel-tracking update for a package delayed by snow.",
            "写一则虚构的包裹物流更新，说明包裹因下雪延误。",
        ),
        "SAT": (
            "This message strikes exactly the reassuring tone I wanted.",
            "这条消息的安抚语气正合我意。",
        ),
        "NEED_CLARIFICATION": (
            "What did you mean by an exception in the tracking message?",
            "你在物流消息里说的异常具体是什么意思？",
        ),
        "WRONG_ANSWER": (
            "The tracking number in your draft doesn't match the number I provided.",
            "你草稿中的单号与我提供的单号不一致。",
        ),
        "WANT_DIFFERENT": (
            "Use the same information in a more matter-of-fact notification.",
            "使用相同的信息，改写成更客观直接的通知。",
        ),
    },
    {
        "NO_FEEDBACK": (
            "Invent a cooperative board game involving migrating birds.",
            "设计一个关于候鸟迁徙的合作桌游。",
        ),
        "SAT": (
            "We tried the rules and everyone enjoyed the game.",
            "我们试着按这些规则玩了，大家都很喜欢。",
        ),
        "NEED_CLARIFICATION": (
            "I'm not following when the second player is allowed to move.",
            "我没弄明白第二位玩家什么时候可以行动。",
        ),
        "WRONG_ANSWER": (
            "Your scoring example contradicts the rule you stated above it.",
            "你的计分例子与它上面写的规则矛盾了。",
        ),
        "WANT_DIFFERENT": (
            "The game works, but create a quicker variant for a lunch break.",
            "游戏能玩，不过再设计一个适合午休的快速变体。",
        ),
    },
    {
        "NO_FEEDBACK": (
            "Write a welcoming toast for a newly opened neighborhood bakery.",
            "为一家新开业的社区面包店写一段欢迎祝酒词。",
        ),
        "SAT": (
            "That wording feels natural to say aloud. I'll use it.",
            "这段话念起来很自然，我会采用。",
        ),
        "NEED_CLARIFICATION": (
            "Could you explain the metaphor in the final sentence of your toast?",
            "能解释一下你那段祝酒词最后一句的比喻吗？",
        ),
        "WRONG_ANSWER": (
            "You congratulated the business on an anniversary; it is opening today.",
            "你祝贺的是周年纪念，可这家店今天才开业。",
        ),
        "WANT_DIFFERENT": (
            "Make this version more playful and remove the formal salutation.",
            "把这一版写得活泼些，去掉正式称呼。",
        ),
    },
    {
        "NO_FEEDBACK": (
            "How do the front and rear gears on a bicycle work together?",
            "自行车的前后变速齿轮如何配合？",
        ),
        "SAT": (
            "Following your adjustment stopped the chain rubbing.",
            "按你的方法调整后，链条不再摩擦了。",
        ),
        "NEED_CLARIFICATION": (
            "I'm unsure which direction you mean by outward in step three.",
            "第三步里说的向外究竟是哪个方向，我不太确定。",
        ),
        "WRONG_ANSWER": (
            "That adjustment made the problem worse; the chain now falls off.",
            "那样调整后问题更严重了，链条现在会掉下来。",
        ),
        "WANT_DIFFERENT": (
            "The repair instructions make sense. Give me a version that avoids specialist tools.",
            "维修说明我看懂了，再给一个不用专门工具的方案。",
        ),
    },
    {
        "NO_FEEDBACK": (
            "Build a timezone conversion worksheet for an online conference.",
            "为线上会议制作一张时区转换工作表。",
        ),
        "SAT": (
            "Everyone's local time is correct now. Thanks for resolving it.",
            "现在每个人的当地时间都正确了，谢谢你解决这个问题。",
        ),
        "NEED_CLARIFICATION": (
            "Why did you add an extra hour for that city in your worksheet?",
            "为什么你在工作表里给那个城市额外加了一小时？",
        ),
        "WRONG_ANSWER": (
            "You applied daylight saving time to a location that doesn't observe it.",
            "你给一个不实行夏令时的地区应用了夏令时。",
        ),
        "WANT_DIFFERENT": (
            "The times are right; reorganize the worksheet by speaker instead of city.",
            "时间是对的，把工作表改成按讲者排列，不按城市排列。",
        ),
    },
]


def records():
    result = []
    for index, scenario in enumerate(CASES):
        if set(scenario) != set(VELA_LABEL2ID):
            raise ValueError("Every matched scenario must contain all five intentions")
        for label, pair in scenario.items():
            for language, text in zip(("en", "zh"), pair, strict=True):
                result.append(
                    {
                        "id": f"feedback-applicability-train:{hashlib.sha256(text.encode()).hexdigest()}",
                        "group_id": f"feedback-applicability-train:{index}",
                        "label": label,
                        "text": text,
                        "language": language,
                        "source": "original_five_intent_training_contrasts",
                        "source_split": "train",
                        "length_bucket": "authored_short",
                        "label_provenance": "original model-authored training judgment",
                    }
                )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing overwrite of training material")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in records())
    )


if __name__ == "__main__":
    main()
