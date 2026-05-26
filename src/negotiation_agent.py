"""模拟谈判对话Agent - AI扮演企业负责人，与招商人员进行多轮谈判演练"""
from openai import OpenAI
from src.enterprise_tool import EnterpriseTool
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


NEGOTIATION_SYSTEM_PROMPT = """你正在扮演一家企业的负责人，与地方政府招商人员进行投资谈判。

## 角色设定
- 你代表的企业背景和信息由用户提供
- 你需要从企业角度出发，关心土地、税收、人才、配套等实际问题
- 谈判风格：务实、专业，愿意探讨合作可能性但不轻易让步

## 谈判规则
1. 根据企业实际情况提出合理诉求
2. 对于招商方的提议，给出有依据的回应
3. 适时表达满意或继续磋商的意愿
4. 每个回复后，以 [态度] 标记当前态度：积极/观望/需要更多条件

## 输出格式
以企业负责人身份直接回复，语言简洁专业。
"""


class NegotiationAgent:
    """模拟谈判Agent：AI扮演企业负责人进行多轮对话"""

    def __init__(self):
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        self.enterprise_tool = EnterpriseTool()
        self.conversation = []

    def start(self, enterprise_name: str) -> str:
        """初始化谈判，获取企业画像并生成开场白"""
        profile = self.enterprise_tool.get_profile_text(enterprise_name)

        # 构建初始提示
        self.conversation = [
            {"role": "system", "content": NEGOTIATION_SYSTEM_PROMPT},
            {"role": "user", "content": f"""请根据以下企业信息，以该企业负责人的身份开始谈判对话。

{profile}

招商人员即将与你接洽。请从企业角度出发，介绍贵公司的基本情况和投资意向，表达核心关注点。用中文回复。"""},
        ]

        # 调用LLM生成开场白
        resp = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=self.conversation,
            temperature=0.7,
            max_tokens=300,
        )
        reply = resp.choices[0].message.content
        self.conversation.append({"role": "assistant", "content": reply})
        return reply

    def respond(self, user_message: str) -> str:
        """招商人员发送消息，企业负责人回复"""
        self.conversation.append({"role": "user", "content": user_message})

        # 保持上下文窗口（最近10轮）
        ctx = self.conversation[-20:]

        resp = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=ctx,
            temperature=0.7,
            max_tokens=300,
        )
        reply = resp.choices[0].message.content
        self.conversation.append({"role": "assistant", "content": reply})
        return reply

    def get_history(self) -> list[dict]:
        """获取对话历史"""
        return [m for m in self.conversation if m["role"] != "system"]
