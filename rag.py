import time

# search
# def search(query, encoder, index_name, es_client):
#     # Encode the query
#     query_vector = encoder.encode(query).tolist()

#     # Construct the search query
#     search_query = {
#         "size": 5,  # Limit the number of results
#         "query": {
#             "script_score": {
#                 "query": {
#                     "match_all": {}
#                 },
#                 "script": {
#                     "source": "cosineSimilarity(params.query_vector, 'text_vector') + 1.0",
#                     "params": {
#                         "query_vector": query_vector
#                     }
#                 }
#             }
#         }
#     }

#     # Execute the search query
#     results = es_client.search(index=index_name, body=search_query)

#     return results['hits']['hits']

# search
# def search(query, index, num_results):
#     boost = {'text':3.0}
#     results = index.search(
#         query=query,
#         boost_dict=boost, 
#         num_results=num_results
#     )
    
#     return results


def search(query, **kwargs):
    vector_db = kwargs['vector_db']

    if vector_db == "1. Minsearch":
        boost = {'text':3.0}
        results = kwargs['index'].search(
            query=query,
            boost_dict=boost, 
            num_results=kwargs['num_results']
        )
    elif vector_db=="2. Elasticsearch":
        # Encode the query
        if kwargs['sentence_encoder'] == "1. T5":
            query_vector = kwargs['encoder'].encode(query).tolist()
        elif kwargs['sentence_encoder'] == "2. OpenAI":
            query_vector = kwargs['client'].embeddings.create(model=kwargs['model'], input=query)

        # Construct the search query
        search_query = {
            "size": kwargs['num_results'],  # Limit the number of results
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
        results = kwargs['vector_db_client'].search(index=kwargs['index_name'], body=search_query)
        results = results['hits']['hits']
    
    return results

# # prompt
# def build_prompt(query, search_results):
#     prompt_template = """
#     You're a podcast chat bot. Answer the QUESTION based on the CONTEXT from the RESULTS database.
#     Use only the facts from the CONTEXT when answering the QUESTION.

#     QUESTION: {question}
    
#     CONTEXT: 
#     {context}
#     """.strip()
    
#     context = ""
    
#     for doc in search_results:
#         context = context + f"{doc['_source']['text']}\n\n"
    
#     prompt = prompt_template.format(question=query, context=context).strip()

#     return prompt

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
    
    for search_result in search_results:
        doc = search_result['_source']['text'] if '_source' in search_result.keys() else search_result['text']
        # context = context + f"{doc['_source']['text']}\n\n"
        context = context + f"{doc}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()

    return prompt

# generate
def llm(prompt, **kwargs):
    if kwargs['llm_option'] == "1. GPT-4o":
        outputs = kwargs['llm_client'].chat.completions.create(
            model='gpt-4o',
            messages=[{'role': 'user', 'content': prompt}]
        )
        response = outputs.choices[0].message.content
    elif kwargs['llm_option'] == "2. FLAN-5":
        inputs = kwargs['llm_tokenizer'](prompt, return_tensors="pt")
        outputs = kwargs['llm_client'].generate(
            inputs["input_ids"], 
            max_length=100,
            num_beams=5,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=0.95,    
            )
        response = kwargs['llm_tokenizer'].decode(outputs[0], skip_special_tokens=True)
        
    return response

# # rag 
# def rag(query, sentence_encoder, oa_client, index_name, es_client):
#     query = query
#     search_results = search(query, sentence_encoder, index_name, es_client)
#     prompt = build_prompt(query, search_results)
#     answer = llm(oa_client, prompt)
    
#     for word in answer.split():
#         yield word + " "
#         time.sleep(0.05)


# rag 
def rag(query, **kwargs):
    vector_db = kwargs['vector_db']
    index_name = kwargs['index_name']
    index = kwargs['index']
    vector_db_client = kwargs['vector_db_client'] if 'vector_db_client' in kwargs.keys() else None
    sentence_encoder = kwargs['sentence_encoder']
    encoder = kwargs['encoder'] if 'encoder' in kwargs.keys() else None
    client = kwargs['embedding_client'] if 'embedding_client' in kwargs.keys() else None
    model = kwargs['embedding_model'] if 'embedding_model' in kwargs.keys() else None

    search_results = search(query, vector_db=vector_db, index_name=index_name, sentence_encoder=sentence_encoder, index=index, vector_db_client=vector_db_client, encoder=encoder, model=model, client=client,num_results=5)

    prompt = build_prompt(query, search_results)

    llm_option = kwargs['llm_option']
    llm_client = kwargs['llm_client']
    llm_tokenizer = kwargs['llm_tokenizer'] if 'llm_tokenizer' in kwargs.keys() else None

    answer = llm(prompt, llm_option=llm_option, llm_client=llm_client, llm_tokenizer=llm_tokenizer)
    
    for word in answer.split():
        yield word + " "
        time.sleep(0.05)        


# # rag 
# def rag(query, index, llm_client):
#     search_results = search(query, index=index, num_results=5)
#     prompt = build_prompt(query, search_results)
#     answer = llm(prompt, client=llm_client)
    
#     for word in answer.split():
#         yield word + " "
#         time.sleep(0.05)
