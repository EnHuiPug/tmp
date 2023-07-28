## 用户问题类别的prompt模板
quesionType_prompt_template = "判断给定的用户问题“{}”属于以下类别中的哪一个类别？类别是股票分析,板块分析,大盘分析,基金分析,其他金融产品分析,股票筛选,板块筛选,基金筛选,其他金融产品筛选,资产配置,股票信息查询,板块信息查询,市场信息查询,基金信息查询,其他金融产品信息查询,资讯查询,其他信息查询"
## 用户问题中NER的prompt模板
entity_prompt_template = "找出用户问题“{}”中的基金、股票、人名、地名、指标类的实体并用json的列表返回结果，key为entityType和entityName。"
## 用户问题相关问的prompt模板
relatedMatchingprompt_template = "根据用户问题“{}”返回跟它相关的潜在的后续问题。"
## 用户问题兜底问的prompt模板
bottomLineReply_prompt_template = "你的身份是灵犀客服。灵犀客服可7*24小时不停歇的为客户提供智能选股、智能诊股、资产分析等智能化服务。是客户的智能服务助手，可以为客户答疑解惑，全方位满足客户需要，为用户提供一站式智能服务。请理解客户问题，识别问题关键信息，并回复客户暂时无法回答，用语要求拟人的口吻，表明自己的身份，亲切温和可爱。客户问题：{}"

## 通用prompt模板
common_prompt_template = "根据问题“{}”，进行回答。"


class Wrapper:
    def __init__(self, question) -> None:
        self.question = question
        self.quesionType_prompt = self.wrap_quesionType_prompt()
        self.entity_prompt = self.wrap_entity_prompt()
        self.relatedMatching_prompt = self.wrap_relatedMatching_prompt()
        self.bottomLineReply_prompt = self.wrap_bottomLineReply_prompt()
        self.common_prompt = self.wrap_common_prompt()

    def wrap_quesionType_prompt(self):
        return quesionType_prompt_template.format(self.question)

    def wrap_entity_prompt(self):
        return entity_prompt_template.format(self.question)

    def wrap_relatedMatching_prompt(self):
        return relatedMatchingprompt_template.format(self.question)

    def wrap_bottomLineReply_prompt(self):
        return bottomLineReply_prompt_template.format(self.question)

    def wrap_common_prompt(self):
        return common_prompt_template.format(self.question)