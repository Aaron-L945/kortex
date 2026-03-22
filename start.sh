#!/bin/bash
# 启动脚本：在两个终端分别运行 API 和前端

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# 检查 .env 文件
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "⚠️  请先编辑 .env 文件，填写 ANTHROPIC_API_KEY"
    exit 1
fi

echo "🚀 启动 FastAPI 后端（端口 8000）..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

sleep 2

echo "🎨 启动 Streamlit 前端（端口 8501）..."
streamlit run frontend/app.py --server.port 8501 &
FRONTEND_PID=$!

echo ""
echo "✅ 系统已启动："
echo "   API 文档:   http://localhost:8000/docs"
echo "   前端界面:   http://localhost:8501"
echo ""
echo "按 Ctrl+C 停止所有服务"

trap "kill $API_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
