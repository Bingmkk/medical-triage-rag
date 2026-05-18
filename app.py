"""
医学分诊RAG系统 - Web应用主文件
"""
import sys
import os
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template
from flask_cors import CORS
from api.routes import api_bp
from config import WEB_CONFIG


def create_app():
    """创建Flask应用"""
    # 全局注入：让 json.dumps 能处理 numpy 类型
    _original_default = json.JSONEncoder.default
    def _numpy_safe_default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return _original_default(self, obj)
    json.JSONEncoder.default = _numpy_safe_default

    app = Flask(__name__,
                template_folder='templates',
                static_folder='static')

    CORS(app)

    app.config['JSON_AS_ASCII'] = False
    app.config['JSON_SORT_KEYS'] = False

    app.register_blueprint(api_bp)

    @app.route('/')
    def index():
        """首页 - 分诊主页面"""
        return render_template('index.html')

    return app


if __name__ == '__main__':
    app = create_app()

    print("=" * 60)
    print("医学分诊RAG系统启动中...")
    print("=" * 60)
    print(f"服务地址: http://{WEB_CONFIG['host']}:{WEB_CONFIG['port']}")
    print(f"调试模式: {WEB_CONFIG['debug']}")
    print("=" * 60)
    print("\n页面路由：")
    print("  • 首页:          http://localhost:5000/   (完整分诊流程)")
    print("\nAPI端点：")
    print("  • POST /api/triage             - 医学分诊")
    print("  • POST /api/clarify/start      - 开始症状追问")
    print("  • POST /api/clarify/answer     - 回答追问")
    print("  • POST /api/clarify/confirm    - 确认追问结果")
    print("  • GET  /api/health             - 健康检查")
    print("  • GET  /api/hospital           - 医院信息")
    print("  • GET  /api/departments        - 科室列表")
    print("=" * 60)

    app.run(
        host=WEB_CONFIG['host'],
        port=WEB_CONFIG['port'],
        debug=WEB_CONFIG['debug']
    )
