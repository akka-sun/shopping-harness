"""根据最新 observation 守卫模型的工具调用。

ShopSimulator 的按钮和商品只在当前页面有效。动作守卫在请求到达环境前检查
这一点，并把拒绝原因反馈给模型，避免非法点击污染环境状态。
"""

import json
import re

from shopping_grpo.environment.product_id import PRODUCT_ID_CAPTURE
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS, tool_call_to_action


RUNTIME_GUARD_FIELD = "runtime_action_guard"
NAVIGATION_BUTTONS = {
    "description",
    "features",
    "reviews",
    "attributes",
    "next >",
    "< prev",
    "back to search",
    "buy now",
}
TOOL_ARGUMENT_NAMES = {
    tool["function"]["name"]: set(tool["function"]["parameters"].get("properties", {}))
    for tool in SHOP_TOOL_SCHEMAS
}


def action_reject_reason(name, arguments, observation, *, instruction=""):
    """返回动作拒绝原因；``None`` 表示允许执行。

检查顺序很重要：先拦截 schema 外字段，再处理无需页面状态的动作，最后只允许
点击最新 observation 中仍然存在的目标。
"""
    extra_argument_names = _schema_extra_argument_names(name, arguments)
    if extra_argument_names:
        return "schema_extra_arguments:" + ",".join(extra_argument_names)
    if name == "think":
        return None
    if name == "finish_without_purchase":
        if arguments.get("reason") != "no_suitable_product":
            return "invalid_finish_reason"
        return None
    if name == "search_products":
        if "搜索功能是否可用: False" in observation:
            return "search_not_available_on_current_page"
        return None
    if name == "buy_now":
        conflict = purchase_option_conflict(instruction, observation)
        if conflict is not None:
            return "buy_option_conflict:" + conflict
    if not observation:
        return "missing_previous_observation"
    if name == "open_product":
        asin = str(arguments.get("asin", ""))
        if asin not in product_ids(observation):
            return "click_not_in_previous_observation"
        return None

    try:
        action = tool_call_to_action(name, arguments)
    except Exception:
        return "unknown_or_invalid_tool"
    if not isinstance(action, str) or not action.startswith("click[") or not action.endswith("]"):
        return "click_not_in_previous_observation"
    target = action[6:-1].casefold()
    if name == "select_option" and target in NAVIGATION_BUTTONS:
        return "select_option_is_navigation_button"
    if target not in {button.casefold() for button in clickable_buttons(observation)}:
        return "click_not_in_previous_observation"
    return None


def purchase_option_conflict(instruction, observation):
    """Return a public, high-confidence selected-option conflict, if any.

    This deliberately avoids a goal/target lookup.  It compares only the user's
    literal request with labels on the current product page, and declines to
    decide when the evidence is ambiguous.  The narrow margin means it cannot
    turn a preference such as a colour mention into an unconditional block when
    several visible variants match the request equally well.
    """
    request = _option_text(instruction)
    if not request:
        return None
    selected = _observation_option_map(observation, "selected_options")
    available = _observation_option_map(observation, "available_options")
    if not selected or not available:
        return None
    for axis, selected_value in selected.items():
        choices = available.get(axis)
        if not isinstance(choices, list) or len(choices) < 2:
            continue
        scored = []
        for value in choices:
            if not isinstance(value, str):
                continue
            score = _literal_option_match_score(request, _option_text(value))
            scored.append((score, value))
        if not scored:
            continue
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_value = scored[0]
        selected_score = _literal_option_match_score(request, _option_text(selected_value))
        # At least a 5-character literal phrase is required.  Requiring a
        # unique winner and a two-character advantage prevents generic tokens
        # ("红色", "礼盒", "大号") from over-ruling an already plausible choice.
        if (
            best_value != selected_value
            and best_score >= 5
            and best_score >= selected_score + 2
            and (len(scored) == 1 or best_score > scored[1][0])
        ):
            return f"{axis}:selected={selected_value};visible_exact_match={best_value}"
    return None


def _observation_option_map(observation, field):
    match = re.search(rf"(?m)^{re.escape(field)}:\s*(\{{[^\n]*\}})", str(observation or ""))
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _option_text(value):
    # Chinese characters, Latin/digits, and the common CJK extension range are
    # retained; packaging punctuation and axes are intentionally ignored.
    return "".join(re.findall(r"[\w\u3400-\u9fff]", str(value).casefold()))


def _literal_option_match_score(request, option):
    """Length of the strongest literal option phrase contained in request."""
    if not request or not option:
        return 0
    best = 0
    # Option labels are short.  Enumerating substrings makes matching stable for
    # punctuation differences such as "卡通-永结同心款" vs "【卡通-永结同心】".
    for start in range(len(option)):
        for end in range(start + 3, len(option) + 1):
            if end - start <= best:
                continue
            if option[start:end] in request:
                best = end - start
    return best


def _schema_extra_argument_names(name, arguments):
    """只拒绝 schema 未声明字段；缺少必填字段仍由工具动作转换报错。"""
    allowed_names = TOOL_ARGUMENT_NAMES.get(name)
    if allowed_names is None or not isinstance(arguments, dict):
        return []
    return sorted(set(arguments) - allowed_names)


def action_guard_tool_message(tool_call, reason, observation):
    """构造标准 tool error observation，让 Agent 感知调用未被执行。"""
    targets = clickable_buttons(observation)
    asins = product_ids(observation)
    parts = []
    if asins:
        parts.append("可打开的商品 ASIN: " + ", ".join(asins[:20]))
    if targets:
        parts.append("可点击按钮: " + ", ".join(targets[:30]))
    allowed = "；".join(parts) or "当前页面没有可用点击目标"
    return_tools = []
    normalized_targets = {target.casefold() for target in targets}
    if "< prev" in normalized_targets:
        return_tools.append("prev_page")
    if "back to search" in normalized_targets:
        return_tools.append("back_to_search")
    only_return_buttons = bool(normalized_targets) and normalized_targets <= {"< prev", "back to search"}
    if only_return_buttons:
        correction = f"你处于信息子页，下一步只能调用 {' 或 '.join(return_tools)}。"
    elif str(reason) == "forced_buy_only":
        correction = (
            "目标规格已经选择完成。不要再次选择规格或浏览页面，下一步只能调用 buy_now。"
        )
    elif str(reason).startswith("buy_option_conflict:"):
        marker = "visible_exact_match="
        exact = str(reason).split(marker, 1)[1] if marker in str(reason) else ""
        correction = (
            "当前 buy_now 被拒绝，因为用户明确要求的规格与已选规格冲突。"
            f"请先调用 select_option，value 必须精确填写当前可见的目标选项：{exact}。"
            "选择成功后再重新检查页面，不能重复调用 buy_now。"
        )
    else:
        correction = "下一步只能从当前页面列出的目标中选择。"
    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id"),
        "name": (tool_call.get("function") or {}).get("name"),
        RUNTIME_GUARD_FIELD: True,
        "content": (
            f"上一工具调用被本地动作守卫拒绝（{reason}），未执行。"
            f"{correction}{allowed}。不要重复使用历史页面目标；只调用一个合法工具。"
        ),
    }


def product_ids(observation):
    """提取当前 observation 中可打开的商品 ID，并保持出现顺序。"""
    return list(
        dict.fromkeys(
            re.findall(
                rf"(?m)^\d+\|({PRODUCT_ID_CAPTURE})\|",
                observation,
            )
        )
    )


def clickable_buttons(observation):
    """读取 observation footer 中当前页面实际可点击的按钮。"""
    match = re.search(r"可点击的按钮:\s*(\[[^\n]*\])", observation)
    if not match:
        return []
    try:
        buttons = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    return [button for button in buttons if isinstance(button, str)]


def available_tool_names(observation):
    """Return public tools that can pass the current observation guard."""
    names = {"finish_without_purchase"}
    if "搜索功能是否可用: False" not in str(observation or ""):
        names.add("search_products")
    if product_ids(observation):
        names.add("open_product")
    buttons = {button.casefold() for button in clickable_buttons(observation)}
    visible_product_ids = {product_id.casefold() for product_id in product_ids(observation)}
    mapping = {
        "description": "view_description",
        "features": "view_features",
        "reviews": "view_reviews",
        "attributes": "view_attributes",
        "next >": "next_page",
        "< prev": "prev_page",
        "back to search": "back_to_search",
        "buy now": "buy_now",
    }
    names.update(tool for button, tool in mapping.items() if button in buttons)
    if buttons - NAVIGATION_BUTTONS - visible_product_ids:
        names.add("select_option")
    return names
