#!/usr/bin/env python3
"""启动 PaRL Web 前端服务。"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "web.server:app",
        host="127.0.0.1",
        port=7860,
        reload=False,
    )
