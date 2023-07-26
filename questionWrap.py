

## 用户问题类别的prompt模板
quesionType_prompt_template = "判断给定的用户问题“{}”属于什么类别？"
## 用户问题中NER的prompt模板
entity_prompt_template = "找出用户问题“{}”中的实体并用json的列表返回结果，key为entityType和entityName。"
## 用户问题相关问的prompt模板
relatedMatchingprompt_template = "根据用户问题“{}”返回跟它相关的潜在的后续问题。"
## 用户问题兜底问的prompt模板
bottomLineReply_prompt_template = "理解客户问题“{}”，根据问题中的信息，组织语言委婉地回复客户你没有足够的知识和能力进行回答。"


class Wrapper:
    def __init__(self, question) -> None:
        self.question = question
        self.quesionType_prompt = self.wrap_quesionType_prompt()
        self.entity_prompt = self.wrap_entity_prompt()
        self.relatedMatching_prompt = self.wrap_relatedMatching_prompt()
        self.bottomLineReply_prompt = self.wrap_bottomLineReply_prompt()

    def wrap_quesionType_prompt(self):
        return quesionType_prompt_template.format(self.question)

    def wrap_entity_prompt(self):
        return entity_prompt_template.format(self.question)

    def wrap_relatedMatching_prompt(self):
        return relatedMatchingprompt_template.format(self.question)

    def wrap_bottomLineReply_prompt(self):
        return bottomLineReply_prompt_template.format(self.question)


