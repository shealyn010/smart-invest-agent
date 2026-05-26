"""企业画像工具 - 模拟企查查等外部API，返回本地脱敏JSON数据"""
import json
from pathlib import Path
from config import ENTERPRISE_DIR


class EnterpriseTool:
    """模拟外部企业信息查询API，返回预设的脱敏结构化数据"""

    def __init__(self):
        data_path = ENTERPRISE_DIR / "enterprises.json"
        with open(data_path) as f:
            raw = json.load(f)
        self.enterprises = raw["enterprises"]
        self._index = {e["name"]: e for e in self.enterprises}
        self._index.update({e["short_name"]: e for e in self.enterprises})

    def search(self, keyword: str) -> list[dict]:
        """模糊搜索企业"""
        results = []
        for e in self.enterprises:
            if keyword in e["name"] or keyword in e["short_name"] or keyword in e["industry"]:
                results.append(e)
        return results

    def get_profile(self, name: str) -> dict | None:
        """获取企业完整画像"""
        return self._index.get(name)

    def list_all(self) -> list[dict]:
        """列出所有企业"""
        return self.enterprises

    def get_profile_text(self, name: str) -> str:
        """生成企业画像文本，供 Agent 使用"""
        e = self._index.get(name)
        if not e:
            return f"未找到「{name}」的企业信息。"

        return f"""企业名称：{e['name']}
所属行业：{e['industry']}
所在城市：{e['city']}
注册资本：{e['registered_capital']}
员工规模：{e['employees']}
2024年营收：{e['revenue_2024']}
主营业务：{e['main_business']}
近期投资动态：{e['investment_history']}
核心关注点：{', '.join(e['key_concerns'])}
扩张意向：{e['expansion_plan']}"""
