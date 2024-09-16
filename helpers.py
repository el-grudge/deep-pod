import os
from pydub import AudioSegment
import io
import time
from gliner import GLiNER
from collections import defaultdict

def shrink_mp3(mp3_file):
    # Split the file path into directory, base filename, and extension
    directory, filename = os.path.split(mp3_file)
    basename, ext = os.path.splitext(filename)

    # Create a new filename with the suffix '_smaller'
    new_filename = f"{basename}_smaller{ext}"

    # Create the new file path
    new_file_path = os.path.join(directory, new_filename)

    # decrease size
    audio = AudioSegment.from_mp3(mp3_file)
    # Set desired sample rate and bit depth to control size
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(16 // 8)  # 8 bits = 1 byte    
    audio = audio.set_channels(1)  # Convert to mono
    audio.export(new_file_path, format="mp3", bitrate="128k")

    # Read the local audio file in binary mode
    with open(new_file_path, "rb") as f:
        audio_blob = io.BytesIO(f.read())  # Use BytesIO to create a file-like object

    return audio_blob

def transcribe(replicate_client, mp3_file):

    audio_blob = shrink_mp3(mp3_file)

    # create sepaarte function
    start_time = time.time()
    output = replicate_client.run(
        "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c",
        input={
            "task": "transcribe",
            "audio": audio_blob,
            "language": "None",
            "timestamp": "chunk",
            "batch_size": 64,
            "diarise_audio": False
        }
    )
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time to run the command: {execution_time} seconds")

    return output

# Function to rank and return top words based on average score and count
def get_top_words_per_topic(topics, top_n=15):
    grouped = defaultdict(lambda: defaultdict(list))

    for item in topics:
        grouped[item['label']][item['text']].append(item['score'])

    entities = [
        {label: [
            {
                'text': text,
                'total_score': sum(scores),
                'count': len(scores)
            } for text, scores in texts.items()
        ]} for label, texts in grouped.items()
    ]

    sorted_data = defaultdict(list)

    # Sort each category by count and average_score (both descending)
    for item in entities:
        for label, values in item.items():
            sorted_values = sorted(values, key=lambda x: (-x['count'], -x['total_score']))
            sorted_data[label].extend(sorted_values)

    # Prepare the final dictionary with the top 5 words based on sorted order
    top_words_per_topic = {}

    for label, items in sorted_data.items():
        top_words_per_topic[label] = [item['text'] for item in items[:top_n]]  # Get top 5 words

    return top_words_per_topic

def list_topics(documents, labels):
    # list of topics in bar 
    model = GLiNER.from_pretrained("urchade/gliner_base")
    
    chunks = documents['chunks']
    entities = []

    for chunk in chunks:
        entities += model.predict_entities(chunk['text'], labels)

    return get_top_words_per_topic(entities)
    # return entities


