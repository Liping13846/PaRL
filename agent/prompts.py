QUERY_PARSE_SYSTEM = """你是学术论文检索助手。请将用户查询解析为 JSON，字段包括：
- intent: semantic_search | metadata_search | navigational
- constraints: 对象，可含 topic, method, dataset, year, venue, cites 等
- sub_queries: 可独立检索的子查询列表（2-4条）
- expanded_terms: 缩写展开和同义词
只输出 JSON，不要其他文字。"""

QUERY_PARSE_USER = "用户查询：{query}"

GENERATE_SEARCH_QUERIES = """请根据用户学术查询生成若干互斥的搜索词，用于论文检索。
优先覆盖不同角度，每行一个，格式为 [Search]查询词[
用户查询：{query}
已解析子查询：{sub_queries}
"""

SELECT_PAPER = """你是 AI 领域专家，正在调研：{user_query}
请判断下面这篇论文是否满足用户查询要求。

论文标题：{title}
论文摘要：{abstract}

输出格式：
Decision: True/False
Reason: ...
Decision:"""

# 对齐 PaSa-7B Selector 训练格式（agent_prompt.json -> get_selected）
SELECT_PAPER_PASA = """You are an elite researcher in the field of AI, conducting research on {user_query}. Evaluate whether the following paper fully satisfies the detailed requirements of the user query and provide your reasoning. Ensure that your decision and reasoning are consistent.

Searched Paper:
Title: {title}
Abstract: {abstract}

User Query: {user_query}

Output format: Decision: True/False
Reason:...
Decision:"""

# 对齐 PaSa-7B Crawler 训练格式（agent_prompt.json -> generate_query）
GENERATE_SEARCH_QUERIES_PASA = """Please generate some mutually exclusive queries in a list to search the relevant papers according to the User Query. Searching for survey papers would be better.
User Query: {user_query}"""

REFLECT_SEARCH = """你是学术检索策略助手。根据当前检索状态，判断是否继续搜索。

用户查询：{query}
子查询：{sub_queries}
已召回论文数：{recall_count}
高分论文数：{high_score_count}
剩余 API 预算：{api_budget}

请输出 JSON：
{{
  "continue": true/false,
  "reason": "...",
  "new_queries": ["..."]
}}
"""

CLUSTER_SUMMARY = """请根据以下论文列表，按主题聚类并各写一句归纳。
用户查询：{query}
论文列表：
{papers}

输出 JSON：
{{
  "clusters": [
    {{"name": "主题名", "summary": "一句话归纳", "paper_ids": ["..."]}}
  ]
}}
"""
