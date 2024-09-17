# deep-pod

## tasks

- [x] create transcribe function in utils 
- [x] modify utils.user_interface to take specific episode  
- [x] ~~split create index from rag~~
- [x] create get index function that can be used by rag, should take index name as parameter
- [x] write problem description
- [x] rag flow
- [ ] retrieval evaluation
- [ ] rag evaluation
- [x] interface
- [ ] ingestion
- [ ] monitoring
- [ ] containerization
- [ ] reproducibility
- [ ] hybrid search (keyword + semantic)
- [ ] document re-ranking 
- [ ] user query rewriting 
- [x] deploy to cloud
- [x] use gliner to list named entities
- [ ] local transcription 
- [ ] cpu / gpu transcription
- [x] gliner: print top 5 entities per topic in table
- [ ] open source llms
- [ ] gliner optimization
- [ ] create a dynamic update state function
- [ ] remove the transcribe episode function from the interface

## criteria
Problem description:  
    2 points: The problem is well-described and it's clear what problem the project solves
RAG flow: 
    2 points: Both a knowledge base and an LLM are used in the RAG flow
Retrieval evaluation: -- ok
    2 points: Multiple retrieval approaches are evaluated, and the best one is used
RAG evaluation: -- 
    2 points: Multiple RAG approaches are evaluated, and the best one is used
Interface: -- 
    2 points: UI (e.g., Streamlit), web application (e.g., Django), or an API (e.g., built with FastAPI)
Ingestion pipeline: 
    1 point: Semi-automated ingestion of the dataset into the knowledge base, e.g., with a Jupyter notebook
    2 points: Automated ingestion with a Python script or a special tool (e.g., Mage, dlt, Airflow, Prefect)
Monitoring: 
    1 point: User feedback is collected OR there's a monitoring dashboard
    2 points: User feedback is collected and there's a dashboard with at least 5 charts
Containerization: 
    2 points: Everything is in docker-compose
Reproducibility: 
    2 points: Instructions are clear, the dataset is accessible, it's easy to run the code, and it works. The versions for all dependencies are specified.
Best practices: 
    Hybrid search: combining both text and vector search (at least evaluating it) (1 point)
    Document re-ranking (1 point)
    User query rewriting (1 point)
Bonus points (not covered in the course):
    Deployment to the cloud (2 points)

## Problem description

This is deep-pod 🎙️, a streamlit app that allows you to interact with your podcast through:

1. Chat 💬
2. Summary 📝
3. Topic detection 🔍

**Chat 💬**

For this functionality, I built a RAG pipeline using the podcast's transcript. 

*Data Ingestions and Transcription*

Podcast episodes are downloaded in one of two ways:

1. By providing a URL for the desired episode
2. By providing the name of a podcast, for which the latest episode will be downloaded

After the mp3 file is downloaded, I proceed to the transcription and indexing processes.

I'm using Replicate's 'incredibly fast whisper' to transcribe the mp3 files 🎵. Replicate provides access to LLMs (and GPUs) using APIs. The incredibly fast whisper costs approximately $0.0079 per run. (in 4 days it cost me $2.14) (https://lnkd.in/eQwqjWEN). Note: Even with Replicate's GPU backed models, transcription can be slow especially for long format podcasts. One trick is to shrink the audio file size by decreasing the file's bit rate and using a mono stream. For a 25 min podcast it can take between 30-45 seconds to transcribe the episode. 

The returned object contains the transcript as a string and as a list of sentences, thus I no longer need to chunk the text (at least for now; I'm considering trying thematic or sentiment chunking to weed out ads).

For more details, check out the RAG flow section below.

**Summary 📝**

Good old web scrapping 🕸️

**Topic detection 🔍**

I'm using GLiNER for named entity recognition 🧠 (link: https://lnkd.in/ePMmR2hN). GLiNER is a very strong technique that can detect any kind of topic using bidirectional encoders to process the contexts and to facilitate parallel entity extraction. 

However, I noticed that its NER detection can be impacted if it's given a large text, on the other hand, it is slow and extra granular when given smaller chunks. This part is still a work in progress. 🚧

**You can try the app here: https://lnkd.in/eqCv-xZC 🚀**

To use it, you will need:

- OpenAI API key 🔑
- Replicate API key 🔑
- Elasticsearch API key 🔑
- Elasticsearch Cloud ID ☁️

## RAG flow

*Search*

For indexing the text, I'm using an Elasticsearch cloud index ☁️ (I'm taking advantage of the 14 day trial period, it costs $95 per month, will look for alternatives) (https://lnkd.in/eVqkyg9s) 

When a user provides a query in the chat bar, the query is encoded and a semantic search is conducted against the index to retrieve the top 5 documents (chunked by sentence).

*Prompt*

A prompt that includes the search query and the top 5 documents is constructed.

*Text generation*

And for text generation, I'm using GPT 4o 🤖 (Less than $1 over the past 4 days) (https://lnkd.in/e9fiapjS). The prompt is passed to the completion API and the contents are retrieved and presented to the user.

## Retrieval evaluation

⚠️ In progress (...)

## RAG evaluation

⚠️ In progress (...)

## Interface

A streamlit interface is built on top of the app. Through the interface, users can download the podcast to interact with it through chat, summary, or topic modelling. 

## Ingestion pipeline 

Data ingestion is handled with a python script. The script does two things:

1- Uses the iTunes API to search and download for the requested episode
2- Creates an Elasticsearch index with the encoded text

## Monitoring

⚠️ In progress (...)

## Containerization 

⚠️ In progress (...)

## Reproducibility 

⚠️ In progress (...)

## Best practices

⚠️ In progress (...)

## Bonus points (not covered in the course)

⚠️ In progress (...)


### steps
1. download episode
2. shrink mp3 file 
3. transcribe with replicate
4. batch and encode 
5. search
6. prompt
7. generate
8. rag 

### commands
```bash
docker run -it \
    --rm \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.15.0
```
**with increased memory limit**
```bash
docker network create elastic

docker run -it \
    --rm \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" \
    --memory=4g \
    docker.elastic.co/elasticsearch/elasticsearch:8.15.0

docker rm elasticsearch
```

```bash
pip install --upgrade transformers accelerate langchain langchain-text-splitters openai pydub feedparser librosa faster-whisper pydub torch sentence_transformers==2.7.0 elasticsearch=8.15.0 replicate spacy boto3 gliner gliner-spacy sentencepiece streamlit beautifulsoup4

python -m spacy download en_core_web_sm
```

**git**
echo "# deep-pod" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:el-grudge/deep-pod.git
git push -u origin main

**requirements**
pip freeze | grep -E 'transformers|accelerate|langchain|langchain-text-splitters|openai|pydub|feedparser|librosa|faster-whisper|torch|sentence-transformers|elasticsearch|replicate|spacy|boto3|gliner|gliner-spacy|sentencepiece|streamlit|beautifulsoup4' > requirements.txt

**ffmpeg**
ffmpeg -i output.webm -filter_complex \
"[0:v]trim=0:20,setpts=PTS-STARTPTS[v1]; \
 [0:v]trim=59:95,setpts=PTS-STARTPTS[v2]; \
 [0:v]trim=215,setpts=PTS-STARTPTS[v3]; \
 [v1][v2][v3]concat=n=3:v=1:a=0[vout]" \
-map "[vout]" -c:v libvpx-vp9 -crf 30 -b:v 0 output_trimmed_video.webm

