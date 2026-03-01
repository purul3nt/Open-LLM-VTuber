# config_manager/system.py
from pydantic import Field, model_validator
from typing import Dict, ClassVar, Optional, List
from .i18n import I18nMixin, Description


class VNReaderConfigSchema(I18nMixin):
    """VN reader settings (monitor, ROI for dialogue box)."""

    monitor_index: int = Field(1, alias="monitor_index")
    roi_rel: List[float] = Field(
        default=[0.30, 0.45, 0.80, 0.70],
        alias="roi_rel",
    )
    min_change_chars: int = Field(8, alias="min_change_chars")


class SystemConfig(I18nMixin):
    """System configuration settings."""

    conf_version: str = Field(..., alias="conf_version")
    host: str = Field(..., alias="host")
    port: int = Field(..., alias="port")
    config_alts_dir: str = Field(..., alias="config_alts_dir")
    tool_prompts: Dict[str, str] = Field(..., alias="tool_prompts")
    proactive_speak_cooldown_seconds: float = Field(
        99.0, alias="proactive_speak_cooldown_seconds"
    )
    enable_proxy: bool = Field(False, alias="enable_proxy")
    enable_vn_reader: bool = Field(False, alias="enable_vn_reader")
    vn_reader_config: Optional[VNReaderConfigSchema] = Field(
        default=None, alias="vn_reader_config"
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "conf_version": Description(en="Configuration version", zh="配置文件版本"),
        "host": Description(en="Server host address", zh="服务器主机地址"),
        "port": Description(en="Server port number", zh="服务器端口号"),
        "config_alts_dir": Description(
            en="Directory for alternative configurations", zh="备用配置目录"
        ),
        "tool_prompts": Description(
            en="Tool prompts to be inserted into persona prompt",
            zh="要插入到角色提示词中的工具提示词",
        ),
        "proactive_speak_cooldown_seconds": Description(
            en="Minimum seconds between proactive-speak LLM calls (0 = no cooldown)",
            zh="主动发言 LLM 调用之间的最小间隔秒数（0=无冷却）",
        ),
        "enable_proxy": Description(
            en="Enable proxy mode for multiple clients",
            zh="启用代理模式以支持多个客户端使用一个 ws 连接",
        ),
        "enable_vn_reader": Description(
            en="Enable Visual Novel screen-capture reader and auto-advance",
            zh="启用视觉小说屏幕捕获阅读器和自动推进",
        ),
        "vn_reader_config": Description(
            en="VN reader monitor and ROI (dialogue box region)",
            zh="VN阅读器显示器和区域",
        ),
    }

    @model_validator(mode="after")
    def check_port(cls, values):
        port = values.port
        if port < 0 or port > 65535:
            raise ValueError("Port must be between 0 and 65535")
        return values
