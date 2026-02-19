from langchain_cohere import CohereRerank

reranker = CohereRerank(
    model="rerank-english-v3.0",
    top_n=10
)

#Rerank会返回相关性分数
results = reranker.rerank(documents, query)

#根据分数分类
for doc in results:
    if doc['relevance_score'] > 0.8:
        doc['category'] = 'CORRECT'
    elif doc['relevance_score'] < 0.3:
        doc['category'] = 'INCORRECT'
    else:
        doc['category'] = 'AMBIGUOUS'
