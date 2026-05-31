"""
日志配置工具函数

提供统一的日志配置，支持控制台和文件输出，日志轮转等功能
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    设置logger，支持多种配置选项

    Args:
        name: logger名称，通常使用 __name__
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径，如果为None则不输出到文件
        console_output: 是否输出到控制台，默认True
        log_format: 自定义日志格式，默认为带时间戳的格式

    Returns:
        配置好的logger实例

    Example:
        >>> logger = setup_logger(__name__, level="INFO", log_file="app.log")
        >>> logger.info("这是一条信息日志")
    """
    # 获取或创建logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 默认日志格式
    if log_format is None:
        log_format = "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"

    # 控制台输出
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        # 确保目录存在
        log_dir = Path(log_file).parent
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """
    获取已配置好的logger，自动生成日志文件名

    Args:
        name: logger名称，通常使用 __name__
        log_dir: 日志目录，默认在当前目录创建logs子目录

    Returns:
        配置好的logger实例

    Example:
        >>> # 量化交易脚本使用示例
        >>> logger = get_logger(__name__, log_dir="./logs")
        >>> logger.info("选股开始")
    """
    logger = logging.getLogger(name)

    # 如果logger已经有handler，直接返回
    if logger.handlers:
        return logger

    # 设置日志级别
    logger.setLevel(logging.INFO)

    # 默认日志目录
    if log_dir is None:
        log_dir = "logs"

    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"{name.replace('.', '_')}_{timestamp}.log")

    # 使用setup_logger配置
    return setup_logger(
        name=name,
        level="INFO",
        log_file=log_file,
        console_output=True,
    )


def set_log_level(level: str):
    """
    动态设置日志级别

    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.getLogger().setLevel(getattr(logging, level.upper()))
