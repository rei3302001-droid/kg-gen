from typing import List, Optional
from pathlib import Path
import dspy
import litellm
from pydantic import BaseModel


class TextEntities(dspy.Signature):
    __doc__ = """从原文中提取关键实体用于构建知识图谱。

你需要做的是：
1. 仔细阅读原文（可能是英文）
2. 识别出所有重要的实体
3. 将每个实体翻译为简洁自然的简体中文后输出

## 实体类别
- 人物：包括以头衔、角色、别名指代的人（"旅店老板"、"男孩"、"国王"）
- 组织：团体、机构、派系
- 地点：场所、建筑、房间、地理特征（"旅店"、"后房"、"高塔"）
- 物品：有重要意义的物件、武器、工具（"剑"、"钥匙"、"硬币"、"蜡烛"）
- 事件/时间：具体事件、节日、时间段（"伐木之夜"）
- 概念：重要思想、传统或抽象实体

## 提取规则
1. 提取所有具名角色以及用角色/头衔/别名描述的人物
2. 提取重要地点包括房间和建筑
3. 提取对叙事关键的重要物品和工具
4. 提取命名的事件和节日
5. 每个实体只用最自然的中文表达，不重复
6. 专有名词合理翻译（如 Taborlin the Great → 伟大的塔博林，Waystone Inn → 路石旅店）"""

    source_text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")


class ConversationEntities(dspy.Signature):
    """Extract key entities from the conversation Extracted entities are subjects or objects.
    Consider both explicit entities and participants in the conversation.
    This is for an extraction task, please be THOROUGH and accurate."""

    source_text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")


class EntitiesResponse(BaseModel):
    """Structured response for entity extraction."""

    entities: List[str]


def _load_entities_prompt() -> str:
    """Load the entities prompt template from file."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "entities.txt"
    return prompt_path.read_text()


def _get_entities_litellm(
    input_data: str,
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
) -> List[str]:
    prompt_template = _load_entities_prompt()
    user_prompt = f"""
Here is the text to extract entities from:

<article>
{input_data}
</article>
    """

    # Build schema with additionalProperties: false (required by OpenAI)
    schema = EntitiesResponse.model_json_schema()
    schema["additionalProperties"] = False

    kwargs = {
        "model": model,
        "input": [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "entities_response",
                "schema": schema,
                "strict": True,
            }
        },
    }

    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    response = litellm.responses(**kwargs)
    # print(response.model_dump_json(indent=2))
    parsed = EntitiesResponse.model_validate_json(response.output[-1].content[0].text)
    return parsed.entities


def get_entities(
    input_data: str,
    is_conversation: bool = False,
    use_litellm_prompt: bool = False,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
) -> List[str]:
    if use_litellm_prompt and not is_conversation:
        return _get_entities_litellm(
            input_data,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
        )

    extract = (
        dspy.Predict(ConversationEntities)
        if is_conversation
        else dspy.Predict(TextEntities)
    )
    result = extract(source_text=input_data)
    return result.entities
