from dotenv import load_dotenv
load_dotenv()

from kg_gen.models import Graph  # noqa: F401
from kg_gen import KGGen
import json  # noqa: F401
import os  # noqa: F401

text = """
恶魔之地

那是一个伐木之夜，往常的人群聚集在路石旅店。五个人算不上什么人群，但时局如此，路石旅店这些天也就只能见到五个人了。

老科布正充当着说书人和忠告提供者的角色。吧台边的男人们啜饮着酒，聆听着。后房里，一个年轻的旅店老板站在门后视线之外，微笑着听一个熟悉故事的细节。

"当他醒来时，伟大的塔博林发现自己被锁在一座高塔里。他们拿走了他的剑，剥去了他所有的工具：钥匙、硬币和蜡烛都不见了。但这还不是最糟的，你知道吗……"科布停顿了一下以增强效果，"……因为墙上的灯正燃烧着蓝色的火焰！"

格雷厄姆、杰克和谢普彼此点了点头。这三个朋友从小一起长大，听着科布的故事，无视他的忠告。

科布仔细打量着他小小听众中那个更新、更专心的成员——铁匠学徒。"你知道那意味着什么吗，小子？"尽管铁匠学徒比在场任何人都高一个头，但所有人都叫他"小子"。小地方就是这样，他很可能一直被称为"小子"，直到他的胡子长满，或者他为此把某人的鼻子打出血。
"""

kg = KGGen(
    model="deepseek/deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.7,
    disable_cache=True,
)
# with open("tests/data/kingkiller_chapter_one.txt", "r", encoding="utf-8") as f:
#     text = f.read()

graph = kg.generate(
    input_data=text,
    context="Kingkiller Chronicles",
    output_folder="./examples/",
    deduplication_method=None,
)
# with open("./examples/graph.json", "r") as f:
#     graph = Graph(**json.load(f))

KGGen.visualize(graph, "./examples/basic-graph.html", True)
