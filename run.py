# 生鲜 AI 品控 — 一键启动脚本

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.app import demo

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )
