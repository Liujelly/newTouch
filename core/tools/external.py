"""外部工具实现: 天气查询。

天气使用 wttr.in 免费接口，无需 key。
当前时间通过 prompt 注入（v2.34），无需工具。
"""
from __future__ import annotations

import httpx

from .registry import register

# ── 天气查询 (wttr.in，免费无需 key) ──────────────────────────
async def _get_weather(location: str = "auto") -> str:
    url = f"https://wttr.in/{location}?format=j1&lang=zh"

    # 重试最多2次（总共3次请求）
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(
                timeout=10.0,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0"}
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()

            cur = data["current_condition"][0]
            desc = cur["weatherDesc"][0]["value"]
            temp = cur["temp_C"]
            feels = cur["FeelsLikeC"]
            today = data["weather"][0]
            rain_chance = today["hourly"][4].get("chanceofrain", "?")

            # 如果是自动定位，尝试获取城市名
            location_name = location
            if location == "auto" and "nearest_area" in data:
                try:
                    location_name = data["nearest_area"][0]["areaName"][0]["value"]
                except (KeyError, IndexError):
                    location_name = "当前位置"

            return (f"{location_name} 当前 {desc}，{temp}°C（体感 {feels}°C），"
                    f"今日降雨概率 {rain_chance}%")

        except httpx.TimeoutException:
            if attempt < 2:
                continue  # 重试
            return f"天气查询超时（已重试{attempt+1}次）"
        except httpx.HTTPStatusError as e:
            return f"天气查询失败: HTTP {e.response.status_code}"
        except (KeyError, IndexError) as e:
            return f"天气数据解析失败: 缺少字段 {e}"
        except Exception as e:
            error_detail = f"{type(e).__name__}: {str(e)}" if str(e) else type(e).__name__
            if attempt < 2:
                continue  # 重试
            return f"天气查询失败: {error_detail}"

    return "天气查询失败: 重试次数已用尽"


register({
    "name": "get_weather",
    "description": "查询某地天气（不传 location 则用 auto 自动定位）",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "城市名，如 Beijing；留空自动定位"},
        },
        "required": [],
    },
}, _get_weather)
