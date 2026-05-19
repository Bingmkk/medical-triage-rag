"""
Web API路由模块
"""
from flask import Blueprint, request, jsonify, Response
from typing import Dict, Any, Generator
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_engine import RAGEngine, create_optimized_rag_engine, TRIAGE_PROMPT_TEMPLATE, TRIAGE_PROMPT_NO_KG
from src.symptom_clarifier import SymptomClarifier
from src.hospital_loader import HospitalDataLoader
from src.smart_referral import SmartReferralTool
from src.department_recommender import enrich_departments_with_referral
import traceback

api_bp = Blueprint('api', __name__, url_prefix='/api')

optimized_rag_engine = None
hospital_loader = None
smart_referral_tool = None


def initialize_agents():
    """初始化Agent"""
    global optimized_rag_engine, hospital_loader, smart_referral_tool

    if optimized_rag_engine is None:
        print("=" * 60)
        print("正在初始化医学分诊系统...")
        print("=" * 60)

        try:
            optimized_rag_engine = create_optimized_rag_engine()
            print("✓ RAG引擎已初始化")
        except Exception as e:
            print(f"✗ RAG引擎初始化失败: {e}")
            import traceback
            traceback.print_exc()
            optimized_rag_engine = None

        try:
            hospital_loader = HospitalDataLoader()
            print("✓ 医院数据加载器已初始化")
        except Exception as e:
            print(f"✗ 医院数据加载器初始化失败: {e}")
            hospital_loader = None

        try:
            smart_referral_tool = SmartReferralTool(hospital_loader)
            print("✓ 智能分流工具已初始化")
        except Exception as e:
            print(f"✗ 智能分流工具初始化失败: {e}")
            smart_referral_tool = None

        print("=" * 60)
        print("Agent初始化完成！")
        print("=" * 60)


def _build_department_recommendations(
    symptom_description: str,
    analysis: str = "",
    urgency_level: int = 4,
    limit: int = 5,
):
    """统一构建科室推荐列表（含智能分流）"""
    if not hospital_loader:
        return [], None

    departments = hospital_loader.recommend_departments(
        symptom_text=symptom_description,
        analysis=analysis,
        urgency_level=urgency_level,
        limit=limit,
    )

    if smart_referral_tool:
        return enrich_departments_with_referral(
            symptom_description, departments, smart_referral_tool
        )
    return departments, None


@api_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "service": "Medical Triage RAG System",
        "version": "1.0.0"
    })


def _triage_stream(symptom_description: str, k: int = 5) -> Generator[str, None, None]:
    """流式分诊生成器"""
    if not optimized_rag_engine:
        yield "data: " + json.dumps({"type": "error", "content": "RAG引擎未初始化"}, ensure_ascii=False) + "\n\n"
        return

    try:
        yield "data: " + json.dumps({"type": "status", "content": "正在检索相关知识..."}, ensure_ascii=False) + "\n\n"

        context = optimized_rag_engine._retrieve_context(symptom_description, use_knowledge_graph=True, k=k)
        yield "data: " + json.dumps({"type": "status", "content": "知识检索完成"}, ensure_ascii=False) + "\n\n"

        hospital_context = optimized_rag_engine._get_hospital_context(symptom_description)

        if context and context != "知识库中暂无相关信息，建议咨询专业医生获取准确诊断。":
            triage_prompt = TRIAGE_PROMPT_TEMPLATE.format(
                context=context,
                hospital_context=hospital_context,
                symptom_description=symptom_description
            )
        else:
            triage_prompt = TRIAGE_PROMPT_NO_KG.format(
                hospital_context=hospital_context,
                symptom_description=symptom_description
            )

        yield "data: " + json.dumps({"type": "status", "content": "正在生成分析..."}, ensure_ascii=False) + "\n\n"

        full_answer = ""
        for chunk in optimized_rag_engine.llm_client.generate_stream(triage_prompt):
            if chunk:
                full_answer += chunk
                yield "data: " + json.dumps({"type": "chunk", "content": chunk}, ensure_ascii=False) + "\n\n"

        urgency_level = optimized_rag_engine._extract_urgency_level(full_answer)

        recommended_departments_detail, smart_referral_result = _build_department_recommendations(
            symptom_description=symptom_description,
            analysis=full_answer,
            urgency_level=urgency_level["level"],
        )
        primary_department = (
            recommended_departments_detail[0]["name"]
            if recommended_departments_detail
            else None
        )

        yield "data: " + json.dumps({
            "type": "done",
            "content": full_answer,
            "urgency_level": urgency_level["level"],
            "urgency_name": urgency_level["name"],
            "urgency_color": urgency_level["color"],
            "primary_department": primary_department,
            "recommended_departments_detail": recommended_departments_detail,
            "hospital_context": hospital_context,
            "smart_referral": smart_referral_result
        }, ensure_ascii=False) + "\n\n"

    except Exception as e:
        yield "data: " + json.dumps({"type": "error", "content": str(e)}, ensure_ascii=False) + "\n\n"


@api_bp.route('/triage/stream', methods=['POST'])
def triage_stream():
    """
    流式医学分诊接口

    请求体：
    {
        "symptom_description": "患者症状描述"
    }

    返回：SSE流式响应
    """
    try:
        data = request.get_json()

        if not data or 'symptom_description' not in data:
            return jsonify({
                "success": False,
                "error": "缺少症状描述"
            }), 400

        symptom_description = data['symptom_description']

        if not symptom_description or len(symptom_description.strip()) == 0:
            return jsonify({
                "success": False,
                "error": "症状描述不能为空"
            }), 400

        if len(symptom_description) > 1000:
            return jsonify({
                "success": False,
                "error": "症状描述过长，请控制在1000字以内"
            }), 400

        initialize_agents()

        return Response(
            _triage_stream(symptom_description),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )

    except Exception as e:
        print(f"流式分诊出错: {e}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"流式分诊出错: {str(e)}"
        }), 500


@api_bp.route('/triage', methods=['POST'])
def triage():
    """
    医学分诊接口

    请求体：
    {
        "symptom_description": "患者症状描述"
    }

    返回：
    {
        "success": true,
        "data": {
            "symptom_description": "...",
            "analysis": "...",
            "urgency_level": 2,
            "urgency_name": "紧急",
            "urgency_color": "orange",
            "recommended_departments_detail": [...],
            "hospital_context": "..."
        }
    }
    """
    try:
        data = request.get_json()

        if not data or 'symptom_description' not in data:
            return jsonify({
                "success": False,
                "error": "缺少症状描述"
            }), 400

        symptom_description = data['symptom_description']

        if not symptom_description or len(symptom_description.strip()) == 0:
            return jsonify({
                "success": False,
                "error": "症状描述不能为空"
            }), 400

        if len(symptom_description) > 1000:
            return jsonify({
                "success": False,
                "error": "症状描述过长，请控制在1000字以内"
            }), 400

        initialize_agents()

        if not optimized_rag_engine:
            return jsonify({
                "success": False,
                "error": "RAG引擎未初始化"
            }), 500

        result = optimized_rag_engine.triage(symptom_description, k=5)

        if not result.get("success"):
            return jsonify({
                "success": False,
                "error": result.get("error", "分诊失败")
            }), 500

        recommended_departments_detail, smart_referral_result = _build_department_recommendations(
            symptom_description=symptom_description,
            analysis=result.get("analysis", ""),
            urgency_level=result["urgency_level"]["level"],
        )
        primary_department = (
            recommended_departments_detail[0]["name"]
            if recommended_departments_detail
            else None
        )

        return jsonify({
            "success": True,
            "data": {
                "symptom_description": result["symptom_description"],
                "analysis": result["analysis"],
                "urgency_level": result["urgency_level"]["level"],
                "urgency_name": result["urgency_level"]["name"],
                "urgency_color": result["urgency_level"]["color"],
                "primary_department": primary_department,
                "recommended_departments_detail": recommended_departments_detail,
                "hospital_context": result.get("hospital_context", ""),
                "context": result.get("context", ""),
                "smart_referral": smart_referral_result
            }
        })

    except Exception as e:
        print(f"分诊出错: {e}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"分诊过程出错: {str(e)}"
        }), 500


@api_bp.route('/triage/batch', methods=['POST'])
def batch_triage():
    """批量分诊接口"""
    try:
        data = request.get_json()

        if not data or 'symptoms' not in data:
            return jsonify({"success": False, "error": "缺少症状列表"}), 400

        symptoms = data['symptoms']

        if not isinstance(symptoms, list) or len(symptoms) == 0:
            return jsonify({"success": False, "error": "症状列表必须是非空数组"}), 400

        if len(symptoms) > 20:
            return jsonify({"success": False, "error": "每次最多支持20个症状"}), 400

        initialize_agents()

        if not optimized_rag_engine:
            return jsonify({"success": False, "error": "RAG引擎未初始化"}), 500

        results = []
        for symptom in symptoms:
            result = optimized_rag_engine.triage(symptom, k=5)

            if result.get("success"):
                recommended_departments_detail = []
                if hospital_loader:
                    recommended_departments_detail = hospital_loader.recommend_departments(
                        symptom_text=symptom,
                        analysis=result.get("analysis", ""),
                        urgency_level=result["urgency_level"]["level"],
                        limit=5,
                    )

                results.append({
                    "symptom_description": result["symptom_description"],
                    "analysis": result["analysis"],
                    "urgency_level": result["urgency_level"]["level"],
                    "urgency_name": result["urgency_level"]["name"],
                    "urgency_color": result["urgency_level"]["color"],
                    "recommended_departments_detail": recommended_departments_detail[:5]
                })

        return jsonify({"success": True, "data": results})

    except Exception as e:
        return jsonify({"success": False, "error": f"批量分诊出错: {str(e)}"}), 500


@api_bp.route('/chat', methods=['POST'])
def chat():
    """对话接口"""
    try:
        data = request.get_json()

        if not data or 'message' not in data:
            return jsonify({"success": False, "error": "缺少消息内容"}), 400

        message = data['message']

        initialize_agents()

        if not optimized_rag_engine:
            return jsonify({"success": False, "error": "RAG引擎未初始化"}), 500

        result = optimized_rag_engine.query(message, use_knowledge_graph=True, k=5)

        if not result.get("success"):
            return jsonify({"success": False, "error": result.get("error", "查询失败")}), 500

        return jsonify({
            "success": True,
            "data": {
                "response": result["answer"],
                "hospital_context": result.get("hospital_context", "")
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"对话处理出错: {str(e)}"}), 500


@api_bp.route('/history', methods=['GET'])
def get_history():
    """获取对话历史"""
    return jsonify({"success": True, "data": [], "message": "当前版本暂不支持历史记录"})


@api_bp.route('/smart-referral', methods=['POST'])
def smart_referral():
    """智能分流接口"""
    try:
        from src.smart_referral import create_smart_referral_tool

        data = request.get_json()

        if not data or 'symptom_description' not in data:
            return jsonify({"success": False, "error": "缺少症状描述"}), 400

        symptom_description = data['symptom_description']
        department_name = data.get('department_name', '内科')

        initialize_agents()

        if not hospital_loader:
            return jsonify({"success": False, "error": "医院数据未初始化"}), 500

        referral_tool = create_smart_referral_tool(hospital_loader)

        result = referral_tool.analyze_and_recommend(
            symptom_description=symptom_description,
            department_name=department_name
        )

        return jsonify({"success": True, "data": result})

    except Exception as e:
        return jsonify({"success": False, "error": f"智能分流出错: {str(e)}"}), 500


@api_bp.route('/doctor/<doctor_name>', methods=['GET'])
def get_doctor_info(doctor_name: str):
    """获取医生信息"""
    try:
        initialize_agents()

        if not hospital_loader:
            return jsonify({"success": False, "error": "医院数据未初始化"}), 500

        doctor_info = hospital_loader.get_doctor_info(doctor_name)

        if not doctor_info:
            return jsonify({"success": False, "error": f"未找到医生: {doctor_name}"}), 404

        return jsonify({"success": True, "data": doctor_info})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


clarification_sessions = {}


@api_bp.route('/clarify/start', methods=['POST'])
def start_clarification():
    """开始症状梳理"""
    try:
        from src.symptom_clarifier import SymptomClarifier

        data = request.get_json()

        if not data or 'symptom_description' not in data:
            return jsonify({"success": False, "error": "请提供症状描述"}), 400

        symptom_description = data['symptom_description']

        session = SymptomClarifier()
        result = session.start_clarification(symptom_description)

        import uuid
        session_id = str(uuid.uuid4())
        clarification_sessions[session_id] = session

        response_data = {"session_id": session_id, **result}

        return jsonify({"success": True, "data": response_data})

    except Exception as e:
        return jsonify({"success": False, "error": f"症状梳理出错: {str(e)}"}), 500


@api_bp.route('/clarify/answer', methods=['POST'])
def answer_clarification():
    """回答症状梳理问题"""
    try:
        data = request.get_json()

        if not data or 'session_id' not in data or 'answer' not in data:
            return jsonify({"success": False, "error": "缺少必要参数"}), 400

        session_id = data['session_id']
        answer = data['answer']

        if session_id not in clarification_sessions:
            return jsonify({"success": False, "error": "会话不存在或已过期"}), 404

        session = clarification_sessions[session_id]
        result = session.continue_clarification(answer)

        return jsonify({"success": True, "data": result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route('/clarify/status/<session_id>', methods=['GET'])
def get_clarification_status(session_id: str):
    """获取症状梳理会话状态"""
    try:
        if session_id not in clarification_sessions:
            return jsonify({"success": False, "error": "会话不存在或已过期"}), 404

        session = clarification_sessions[session_id]

        return jsonify({
            "success": True,
            "data": {
                "session_id": session_id,
                "round": session.current_round,
                "max_rounds": session.max_rounds,
                "history_length": len(session.conversation_history)
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route('/clarify/confirm', methods=['POST'])
def confirm_clarification():
    """确认症状梳理结果"""
    try:
        data = request.get_json()

        if not data or 'session_id' not in data:
            return jsonify({"success": False, "error": "缺少会话ID"}), 400

        session_id = data['session_id']

        if session_id not in clarification_sessions:
            return jsonify({"success": False, "error": "会话不存在或已过期"}), 404

        session = clarification_sessions[session_id]
        clarified = session.get_clarified_symptom()

        return jsonify({
            "success": True,
            "data": {
                "clarified_symptom": clarified,
                "ready_for_triage": True
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route('/history/clear', methods=['POST'])
def clear_history():
    """清除对话历史"""
    return jsonify({"success": True, "message": "当前版本暂不支持历史记录"})


@api_bp.route('/statistics', methods=['GET'])
def get_statistics():
    """获取统计信息"""
    return jsonify({
        "success": True,
        "data": {
            "engine": "RAGEngine",
            "features": ["hybrid_search", "rerank", "knowledge_graph", "hospital_recommendation"],
            "message": "统计功能开发中"
        }
    })


@api_bp.route('/models', methods=['GET'])
def list_models():
    """列出可用的LLM模型"""
    try:
        from src.llm_client import MedicalLLMClient

        models = MedicalLLMClient.list_available_models()
        return jsonify({
            "success": True,
            "data": {
                "available_models": models,
                "default_model": "qwen"
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route('/triage/react', methods=['POST'])
def react_triage():
    """ReAct分诊接口"""
    return triage()


@api_bp.route('/hospital', methods=['GET'])
def get_hospital_info():
    """获取医院信息"""
    try:
        initialize_agents()

        if not hospital_loader:
            return jsonify({"success": False, "error": "医院数据未初始化"}), 500

        hospital_data = hospital_loader.get_all_departments()

        return jsonify({"success": True, "data": hospital_data})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route('/department/recommend', methods=['POST'])
def recommend_department():
    """科室推荐接口（结合症状与可选分诊分析）"""
    try:
        data = request.get_json() or {}
        symptom = data.get("symptom") or data.get("symptom_description", "")
        analysis = data.get("analysis", "")
        urgency_level = int(data.get("urgency_level", 4))

        if not symptom or not str(symptom).strip():
            return jsonify({"success": False, "error": "缺少症状描述"}), 400

        initialize_agents()

        if not hospital_loader:
            return jsonify({"success": False, "error": "医院数据未初始化"}), 500

        departments, smart_referral_result = _build_department_recommendations(
            symptom_description=symptom,
            analysis=analysis,
            urgency_level=urgency_level,
        )

        return jsonify({
            "success": True,
            "data": {
                "symptom": symptom,
                "departments": departments,
                "smart_referral": smart_referral_result,
            },
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route('/departments', methods=['GET'])
def get_departments():
    """获取科室列表"""
    try:
        initialize_agents()

        if not hospital_loader:
            return jsonify({"success": False, "error": "医院数据未初始化"}), 500

        symptom = request.args.get('symptom', '')

        if symptom:
            analysis = request.args.get("analysis", "")
            urgency_level = int(request.args.get("urgency_level", 4))
            recommendations = hospital_loader.recommend_departments(
                symptom_text=symptom,
                analysis=analysis,
                urgency_level=urgency_level,
                limit=5,
            )
            return jsonify({
                "success": True,
                "data": {
                    "symptom": symptom,
                    "recommendations": recommendations
                }
            })
        else:
            hospital_data = hospital_loader.get_all_departments()
            return jsonify({
                "success": True,
                "data": {
                    "departments": hospital_data["departments"]
                }
            })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
