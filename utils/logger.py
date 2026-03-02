import logging
import sys
from datetime import datetime
import os

def setup_logger(name: str = "Qwen3_Medical_SFT", log_level: str = "INFO", log_dir: str = "logs"):
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_level: 日志级别
        log_dir: 日志保存目录
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 创建控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # 创建文件handler
    log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加handler
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # 添加wandb/swanlab等集成
    logger.propagate = False
    
    return logger

class LoggerMixin:
    """为类提供日志功能的Mixin类"""
    
    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = setup_logger(self.__class__.__name__)
        return self._logger

def log_execution_time(func):
    """装饰器：记录函数执行时间"""
    def wrapper(*args, **kwargs):
        logger = setup_logger()
        start_time = datetime.now()
        logger.info(f"开始执行函数: {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"函数 {func.__name__} 执行完成，耗时: {duration:.2f} 秒")
            return result
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行失败: {str(e)}")
            raise
    
    return wrapper