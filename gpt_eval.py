def gpt4_evaluate(query, document):
    prompt = f"""
    评估以下文档是否回答了查询问题。
    查询：{query}
    文档：{document}
    请评估文档的相关性并选择一个类别：
    - CORRECT：文档直接回答了查询，信息准确完整
    - INCORRECT：文档不相关或包含错误信息
    - AMBIGUOUS：文档部分相关但信息不够完整
    只返回类别名称，不需要解释。
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()
