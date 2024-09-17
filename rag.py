import time

# search
def search(query, encoder, index_name, es_client):
    # Encode the query
    query_vector = encoder.encode(query).tolist()

    # Construct the search query
    search_query = {
        "size": 5,  # Limit the number of results
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'text_vector') + 1.0",
                    "params": {
                        "query_vector": query_vector
                    }
                }
            }
        }
    }

    # Execute the search query
    results = es_client.search(index=index_name, body=search_query)

    return results['hits']['hits']

# prompt
def build_prompt(query, search_results):
    prompt_template = """
    You're a podcast chat bot. Answer the QUESTION based on the CONTEXT from the RESULTS database.
    Use only the facts from the CONTEXT when answering the QUESTION.

    QUESTION: {question}
    
    CONTEXT: 
    {context}
    """.strip()
    
    context = ""
    
    for doc in search_results:
        context = context + f"{doc['_source']['text']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()

    return prompt

# generate
def llm(openai_client, prompt):
    response = openai_client.chat.completions.create(
        model='gpt-4o',
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    return response.choices[0].message.content

# rag 
def rag(query, sentence_encoder, oa_client, index_name, es_client):
    query = query
    search_results = search(query, sentence_encoder, index_name, es_client)
    prompt = build_prompt(query, search_results)
    answer = llm(oa_client, prompt)
    
    for word in answer.split():
        yield word + " "
        time.sleep(0.05)