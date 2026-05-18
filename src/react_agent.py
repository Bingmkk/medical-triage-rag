"""
ReAct范式实现模块
ReAct = Reasoning + Acting
让LLM通过思考-行动循环来解决医学分诊问题
"""
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import BaseSingleActionAgent
from langchain.callbacks.base import BaseCallbackManager
from langchain.tools import BaseTool
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
import re
class ActionType(Enum):
    """行动类型枚举"""
    SEARCH_KG = "search_knowledge_graph"  # 搜索知识图谱
    SEARCH_VECTOR = "search_vector_store"  # 搜索向量存储
    ANALYZE_SYMPTOMS = "analyze_symptoms"  # 分析症状
    ASSESS_URGENCY = "assess_urgency"      # 评估紧急程度
    RECOMMEND_DEPARTMENT = "recommend_department"  # 推荐科室
    SMART_REFERRAL = "smart_referral"      # 智能分流
    FINISH = "finish"  # 完成分诊


class ReActTool(BaseTool):
    """ReAct工具基类"""
    name: str
    description: str
    action_type: ActionType

    def _run(self, query: str) -> str:
        """执行工具"""
        raise NotImplementedError("子类必须实现_run方法")


class KnowledgeGraphSearchTool(ReActTool):
    """知识图谱搜索工具"""

    def __init__(self, knowledge_graph):
        self.name = "search_knowledge_graph"
        self.description = "搜索医学知识图谱，获取疾病、症状、药物、科室等信息"
        self.action_type = ActionType.SEARCH_KG
        self.knowledge_graph = knowledge_graph

    def _run(self, query: str) -> str:
        """搜索知识图谱"""
        try:
            result = self.knowledge_graph.get_entity_by_name(query.strip())
            if result:
                relations = self.knowledge_graph.get_disease_relations(result["id"])
                parts = [f"找到实体: {result['name']} ({result['type']})"]

                if result.get("properties", {}).get("description"):
                    parts.append(f"描述: {result['properties']['description']}")

                if relations["symptoms"]:
                    symptoms = "、".join([s["name"] for s in relations["symptoms"]])
                    parts.append(f"常见症状: {symptoms}")

                if relations["drugs"]:
                    drugs = "、".join([d["name"] for d in relations["drugs"]])
                    parts.append(f"常用药物: {drugs}")

                if relations["treatments"]:
                    treatments = "、".join([t["name"] for t in relations["treatments"]])
                    parts.append(f"治疗方法: {treatments}")

                if relations["departments"]:
                    depts = "、".join([d["name"] for d in relations["departments"]])
                    parts.append(f"就诊科室: {depts}")

                return "\n".join(parts)
            else:
                return f"在知识图谱中未找到: {query}"
        except Exception as e:
            return f"搜索失败: {str(e)}"


class VectorSearchTool(ReActTool):
    """向量存储搜索工具"""

    def __init__(self, vector_store):
        self.name = "search_vector_store"
        self.description = "搜索医学文档向量库，获取相关医学知识"
        self.action_type = ActionType.SEARCH_VECTOR
        self.vector_store = vector_store

    def _run(self, query: str) -> str:
        """搜索向量存储"""
        try:
            docs = self.vector_store.similarity_search(query.strip(), k=3)
            if docs:
                results = []
                for i, doc in enumerate(docs, 1):
                    snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    results.append(f"{i}. {snippet}")
                return "\n\n".join(results)
            else:
                return f"未找到相关医学文档: {query}"
        except Exception as e:
            return f"搜索失败: {str(e)}"


class SmartReferralReActTool(ReActTool):
    """智能分流工具 - 根据排班和专业推荐最合适的医生"""

    def __init__(self, hospital_loader):
        from .smart_referral import SmartReferralTool
        self.name = "smart_referral"
        self.description = "根据排班时间、医生专业和当前时间智能推荐最合适的医生"
        self.action_type = ActionType.SMART_REFERRAL
        self.hospital_loader = hospital_loader
        self.referral_tool = SmartReferralTool(hospital_loader)

    def _run(self, query: str) -> str:
        """执行智能分流"""
        try:
            parts = query.strip().split("|")
            if len(parts) >= 2:
                symptom = parts[0].strip()
                department = parts[1].strip()
            else:
                symptom = query.strip()
                department = "内科"
            
            result = self.referral_tool.analyze_and_recommend(
                symptom_description=symptom,
                department_name=department
            )
            
            if not result.get("success"):
                return f"智能分流失败: {result.get('error', '未知错误')}"
            
            output_parts = []
            output_parts.append(f"科室: {result['department']['name']}")
            output_parts.append(f"位置: {result['department']['location']}")
            output_parts.append(f"门牌号: {result['department']['room_number']}")
            
            if result.get("recommended_doctor"):
                doc = result["recommended_doctor"]
                output_parts.append(f"\n推荐医生: {doc['name']} ({doc['title']})")
                output_parts.append(f"专业: {doc['specialty']}")
                output_parts.append(f"排班: {doc['schedule']}")
                output_parts.append(f"当前是否出诊: {'是' if doc['is_available_now'] else '否'}")
                output_parts.append(f"专业匹配度: {doc['specialty_match_score']*100:.0f}%")
                
                if doc.get("next_available_time"):
                    output_parts.append(f"下次出诊: {doc['next_available_time']['message']}")
            
            if result.get("available_doctors_now"):
                output_parts.append(f"\n当前可挂号医生 ({len(result['available_doctors_now'])}人):")
                for doc in result["available_doctors_now"][:3]:
                    output_parts.append(f"  - {doc['name']} ({doc['title']}): {doc['specialty']}")
            
            output_parts.append(f"\n分流建议:\n{result['referral_message']}")
            
            return "\n".join(output_parts)
            
        except Exception as e:
            return f"智能分流失败: {str(e)}"


class ReActPromptTemplate(StringPromptTemplate):
    """ReAct提示词模板"""

    def format(self, **kwargs) -> str:
        """格式化提示词"""
        intermediate_steps = kwargs.pop("intermediate_steps", [])
        thoughts = ""

        for action, observation in intermediate_steps:
            thoughts += f"思考: {action.log}\n行动: {action.tool}({action.tool_input})\n结果: {observation}\n\n"

        template = f"""你是一个专业的医学分诊助手，使用ReAct范式进行推理。

医学分诊需要通过以下步骤完成：
1. 理解患者描述的症状
2. 搜索相关医学知识（知识图谱和文档）
3. 分析症状与疾病的关联
4. 评估紧急程度
5. 推荐就诊科室
6. 给出处置建议

可用工具：
- search_knowledge_graph(query): 搜索医学知识图谱
- search_vector_store(query): 搜索医学文档向量库
- finish(result): 完成分诊并输出结果

请按照以下格式输出：
思考: [你的思考过程]
行动: [工具名]([参数])

当你已经收集到足够信息并可以给出最终分诊结果时，请使用finish工具。

患者描述: {kwargs['input']}

历史思考与行动:
{thoughts}

现在开始分诊："""

        return template


class ReActMedicalAgent(BaseSingleActionAgent):
    """基于ReAct范式的医学分诊Agent"""

    llm_chain: LLMChain
    tools: List[ReActTool]
    max_iterations: int = 10

    @property
    def input_keys(self) -> List[str]:
        return ["input"]

    def plan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs
    ) -> Union[AgentAction, AgentFinish]:
        """规划下一步行动"""
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"思考: {action.log}\n行动: {action.tool}({action.tool_input})\n结果: {observation}\n\n"

        prompt = f"""你是一个专业的医学分诊助手，使用ReAct范式进行推理。

医学分诊需要通过以下步骤完成：
1. 理解患者描述的症状
2. 搜索相关医学知识（知识图谱和文档）
3. 分析症状与疾病的关联
4. 评估紧急程度
5. 推荐就诊科室
6. 给出处置建议

可用工具：
- search_knowledge_graph(query): 搜索医学知识图谱
- search_vector_store(query): 搜索医学文档向量库
- finish(result): 完成分诊并输出结果

请按照以下格式输出：
思考: [你的思考过程]
行动: [工具名]([参数])

当你已经收集到足够信息并可以给出最终分诊结果时，请使用finish工具。

患者描述: {kwargs['input']}

历史思考与行动:
{thoughts}

现在开始分诊："""

        response = self.llm_chain.run(input=kwargs['input'], intermediate_steps=intermediate_steps)

        return self._parse_response(response, intermediate_steps)

    def _parse_response(self, response: str, intermediate_steps: List) -> Union[AgentAction, AgentFinish]:
        """解析LLM响应"""
        lines = response.strip().split('\n')

        thought = ""
        action = ""

        for line in lines:
            if line.startswith("思考:") or line.startswith("Thought:"):
                thought = line[3:].strip()
            elif line.startswith("行动:") or line.startswith("Action:"):
                action = line[3:].strip()

        if not action:
            return AgentFinish(
                return_values={"result": response},
                log=response
            )

        if action.startswith("finish("):
            result = action[7:-1].strip()
            return AgentFinish(
                return_values={"result": result},
                log=f"思考: {thought}\n行动: finish\n结果: {result}"
            )

        match = re.match(r"(\w+)\((.+)\)", action)
        if match:
            tool_name = match.group(1)
            tool_input = match.group(2).strip("'\"")

            return AgentAction(
                tool=tool_name,
                tool_input=tool_input,
                log=thought
            )

        return AgentAction(
            tool="search_knowledge_graph",
            tool_input=kwargs.get('input', ''),
            log=f"无法解析响应，默认搜索症状"
        )

    async def aplan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs
    ) -> Union[AgentAction, AgentFinish]:
        """异步规划"""
        return self.plan(intermediate_steps, **kwargs)


class ReActTriageAgent:
    """ReAct分诊Agent"""

    def __init__(self, llm_client, knowledge_graph=None, vector_store=None, hospital_loader=None):
        self.llm_client = llm_client
        self.knowledge_graph = knowledge_graph
        self.vector_store = vector_store
        self.hospital_loader = hospital_loader
        self.tools = self._initialize_tools()
        self.thought_history = []

    def _initialize_tools(self) -> List[ReActTool]:
        """初始化工具"""
        tools = []

        if self.knowledge_graph:
            tools.append(KnowledgeGraphSearchTool(self.knowledge_graph))

        if self.vector_store:
            tools.append(VectorSearchTool(self.vector_store))

        if self.hospital_loader:
            tools.append(SmartReferralReActTool(self.hospital_loader))

        return tools

    def triage(self, symptom_description: str) -> Dict[str, Any]:
        """使用ReAct范式进行分诊"""
        self.thought_history = []
        iteration = 0
        max_iterations = 10

        while iteration < max_iterations:
            iteration += 1

            thought, action, action_type, query = self._generate_thought(symptom_description)

            self.thought_history.append({
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "action_type": action_type.value if action_type else "unknown"
            })

            if action_type == ActionType.FINISH:
                result = self._finish_triage(thought, symptom_description)
                return result

            observation = self._execute_action(action_type, query)

            self.thought_history[-1]["observation"] = observation

            if "finish" in action.lower():
                result = self._finish_triage(observation, symptom_description)
                return result

        return {
            "symptom_description": symptom_description,
            "result": "分诊过程超时，请简化症状描述后重试",
            "thought_history": self.thought_history,
            "urgency_level": 4
        }

    def _generate_thought(self, symptom_description: str) -> Tuple[str, str, ActionType, str]:
        """生成思考和行动"""
        history_str = "\n".join([
            f"{h['iteration']}. 思考: {h['thought']}, 行动: {h['action']}"
            for h in self.thought_history
        ])

        prompt = f"""你是一个专业的医学分诊助手，使用ReAct范式进行推理。

任务：根据患者描述进行医学分诊

患者描述: {symptom_description}

可用工具：
1. search_knowledge_graph(query) - 搜索医学知识图谱
2. search_vector_store(query) - 搜索医学文档向量库
3. smart_referral(症状|科室) - 智能分流，根据排班和专业推荐最合适的医生
4. finish(result) - 完成分诊并输出结果

已执行的步骤：
{history_str}

请按照以下格式输出（只输出一行）：
思考: [你的思考] | 行动: [工具名]([参数])

当你已经收集到足够信息并可以给出最终分诊结果时，请使用finish工具。

输出示例：
思考: 患者描述胸痛，需要先搜索知识图谱了解相关疾病 | 行动: search_knowledge_graph(胸痛)
思考: 已确定科室为心内科，需要智能分流推荐医生 | 行动: smart_referral(胸痛|心内科)
思考: 已收集到足够信息，可以给出分诊结果 | 行动: finish(紧急程度：紧急，建议科室：心内科)
"""

        response = self.llm_client.generate(prompt)

        lines = response.strip().split('\n')
        thought = ""
        action = ""

        for line in lines:
            if "思考:" in line and "行动:" in line:
                parts = line.split("|")
                for part in parts:
                    if "思考:" in part:
                        thought = part.replace("思考:", "").strip()
                    elif "行动:" in part:
                        action = part.replace("行动:", "").strip()
                break

        action_type, query = self._parse_action(action)

        return thought, action, action_type, query

    def _parse_action(self, action: str) -> Tuple[ActionType, str]:
        """解析行动"""
        if not action:
            return ActionType.SEARCH_KG, ""

        if action.startswith("finish("):
            return ActionType.FINISH, action[7:-1].strip()

        if action.startswith("search_knowledge_graph("):
            query = action[23:-1].strip("'\"")
            return ActionType.SEARCH_KG, query

        if action.startswith("search_vector_store("):
            query = action[22:-1].strip("'\"")
            return ActionType.SEARCH_VECTOR, query

        if action.startswith("smart_referral("):
            query = action[15:-1].strip("'\"")
            return ActionType.SMART_REFERRAL, query

        return ActionType.SEARCH_KG, action

    def _execute_action(self, action_type: ActionType, query: str) -> str:
        """执行行动"""
        if action_type == ActionType.SEARCH_KG and self.knowledge_graph:
            tool = KnowledgeGraphSearchTool(self.knowledge_graph)
            return tool._run(query)

        if action_type == ActionType.SEARCH_VECTOR and self.vector_store:
            tool = VectorSearchTool(self.vector_store)
            return tool._run(query)

        if action_type == ActionType.SMART_REFERRAL and self.hospital_loader:
            tool = SmartReferralReActTool(self.hospital_loader)
            return tool._run(query)

        return f"无法执行行动: {action_type}"

    def _finish_triage(self, result: str, symptom_description: str) -> Dict[str, Any]:
        """完成分诊"""
        urgency_info = self._extract_urgency(result)

        return {
            "symptom_description": symptom_description,
            "result": result,
            "thought_history": self.thought_history,
            "urgency_level": urgency_info["level"],
            "urgency_name": urgency_info["name"],
            "urgency_color": urgency_info["color"]
        }

    def _extract_urgency(self, text: str) -> Dict[str, Any]:
        """提取紧急程度"""
        text_lower = text.lower()

        if any(keyword in text_lower for keyword in ["危急", "立即", "拨打120", "危及生命"]):
            return {"level": 1, "name": "危急", "color": "red"}
        elif any(keyword in text_lower for keyword in ["紧急", "尽快", "严重"]):
            return {"level": 2, "name": "紧急", "color": "orange"}
        elif any(keyword in text_lower for keyword in ["次紧急"]):
            return {"level": 3, "name": "次紧急", "color": "yellow"}
        else:
            return {"level": 4, "name": "非紧急", "color": "green"}

    def get_thought_history(self) -> List[Dict[str, Any]]:
        """获取思考历史"""
        return self.thought_history


def create_react_triage_agent(llm_client, knowledge_graph=None, vector_store=None, hospital_loader=None) -> ReActTriageAgent:
    """创建ReAct分诊Agent"""
    return ReActTriageAgent(llm_client, knowledge_graph, vector_store, hospital_loader)
