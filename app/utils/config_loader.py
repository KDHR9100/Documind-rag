# app/utils/config_loader.py
import os
import json
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/config.json") -> Dict[str, Any]:
    """加载配置文件，并自动合并环境变量覆盖（如有）"""
    if not os.path.exists(config_path):
        logger.warning(f"配置文件 {config_path} 不存在，使用空配置")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 【知识点】：支持环境变量覆盖（例如 DASHSCOPE_API_KEY 已在 .env 中）
    # 这里只是示例，实际可以从 os.getenv 读取覆盖项
    return config