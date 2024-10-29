import requests
from bs4 import BeautifulSoup
import openai
from flask import Flask, request, jsonify, render_template
import re
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from pydub import AudioSegment 
import os
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file

app = Flask(__name__)
CORS(app)
load_dotenv()

# Path for storing the generated file
GENERATED_FILE_PATH = 'generated_responses.xlsx'
# Helper function to fetch page and parse it with BeautifulSoup
def get_soup(url):
    response = requests.get(url)
    if response.status_code == 200:
        return BeautifulSoup(response.text, 'html.parser')
    else:
        print(f"Failed to fetch page: {url}")
        return None
def scrape_profile(profile_url):
    print(f"Fetching profile: {profile_url}")
    soup = get_soup(profile_url)

    if soup:
        # Extract relevant information such as bio, experience, etc.
        profile_data = [tag.get_text(strip=True) for tag in soup.find_all(['p', 'span', 'h3', 'h2'])]

        return profile_data
    else:
        return ["Failed to fetch profile details."]

def scrape_all_pages(query):
    url_dict = {
        "exits": "https://capria.vc/portfolio-exits/",
        "investing-partners": "https://capria.vc/investing-partners/",
        "Africa": "https://capria.vc/entire-portfolio/?jsf=jet-engine:entire-grid&tax=pregion:43",
        "south-east-asia": "https://capria.vc/entire-portfolio/?jsf=jet-engine:entire-grid&tax=pregion:67",
        "Latam": "https://capria.vc/entire-portfolio/?jsf=jet-engine:entire-grid&tax=pregion:52",
        "India": "https://capria.vc/entire-portfolio/?jsf=jet-engine:entire-grid&tax=pregion:48",
        "founders": "https://capria.vc/founders/",
        "careers": "https://capria.vc/careers/",
        "genai": "https://capria.vc/genai/",
        "news": "https://capria.vc/news/",
        "team": "https://capria.vc/team/",
        "homepage": "https://capria.vc/",
    }

    all_page_data = {}

    for page_type, page_url in url_dict.items():
        soup = get_soup(page_url)
        if soup:
            page_data = []
            
            # For team or leadership pages
            if page_type in ["all-team", "team"]:
                for member in soup.find_all('a'):
                    name_tag = member.find('span', class_="pp-first-text")
                    position_tag = member.find('span', class_="pp-second-text")
                    profile_link = member['href'] if member.has_attr('href') else None
                    
                    if name_tag and position_tag and profile_link:
                        name = name_tag.get_text(strip=True)
                        position = position_tag.get_text(strip=True)
                        
                        # Scrape the detailed profile page
                        detailed_profile = scrape_profile(profile_link)
                        
                        # Append the data along with the URL to page_data
                        page_data.append({
                            'name': name,
                            'position': position,
                            'profile_link': profile_link,
                            'detailed_info': detailed_profile,
                            'source_url': page_url  # Include source URL
                        })
            else:
                page_data = [section.get_text(strip=True) for section in soup.find_all(['h1', 'h2', 'p', 'h3', 'span'])]
                
            # Include the source URL in the response
            all_page_data[page_type] = {
                'data': page_data if page_data else [f"No information found on {page_type} page."],
                'source_url': page_url
            }
        else:
            all_page_data[page_type] = {
                'data': [f"Failed to retrieve {page_type} page."],
                'source_url': page_url
            }
    
    return all_page_data

# ChatGPT interaction function
# openai.api_key = os.getenv('OPENAI_API_KEY')

def chat_with_gpt(prompt):
    response = lm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a knowledgeable assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content  # Directly access content


# Handle numeric queries (now tied to "how many")
def handle_numeric_query(query):
    numeric_data = []
    main_url = "https://capria.vc/"
    
    # Scrape only the main page for numeric data
    soup = get_soup(main_url)
    if soup:
        numeric_data += [tag.get_text(strip=True) for tag in soup.find_all(['span','p', 'h1', 'h2']) if any(char.isdigit() for char in tag.get_text(strip=True))]

    return numeric_data if numeric_data else ["No numeric information found."]

# Handle general queries

def handle_general_query(query):
    all_pages_data = scrape_all_pages(query)

    combined_data = ""
    for page, content in all_pages_data.items():
        data = content['data']
        source_url = content['source_url']
        
        if page in ["all-team", "team"]:
            for member in data:
                if isinstance(member, dict):
                    combined_data += (f"\nName: {member['name']}\nPosition: {member['position']}\nProfile: "
                                      f"<a href='{member['profile_link']}' target='_blank'>{member['profile_link']}</a>\n"
                                      f"Details: {' '.join(member['detailed_info'])}\nSource: "
                                      f"<a href='{source_url}' target='_blank'>{source_url}</a>\n\n")
                else:
                    combined_data += f"\n{page} Info:\n{member}\nSource: <a href='{source_url}' target='_blank'>{source_url}</a>\n\n"
        else:
            combined_data += f"\n{page}:\n{' '.join(data)}\nSource: <a href='{source_url}' target='_blank'>{source_url}</a>\n\n"

    prompt = (
        f"Based on the data scraped from Capria's website, answer the following query concisely:\n"
        f"Query: {query}\n\n"
        f"Data:\n{combined_data}\n\n"
        "The response should be clear and accurate from the scraped data. It should also reference the source link of the page where the data was obtained."
    )

    return chat_with_gpt(prompt)




# Main query handling function
def main(query):
    return handle_general_query(query)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/query', methods=['POST'])
def handle_profile_query():
    data = request.get_json()
    query = data.get('query')
    response = main(query)
    
    return jsonify({'response': response})


import os

import difflib
import difflib

# New function to evaluate generated answers with Ground Truth using ChatGPT
def evaluate_with_gpt(generated_answer, ground_truth):
    prompt = (
        f"Compare the following:\n"
        f"Ground Truth: {ground_truth}\n"
        f"Generated Answer: {generated_answer}\n\n"
        "Do they convey the same meaning or same answer they have? if the simillarity occurs then Respond with either 'True' or 'False'. Try to check more accurately "
    )
    
    evaluation_response = chat_with_gpt(prompt)
    
    # Assuming GPT will output either 'True' or 'False' as plain text
    return evaluation_response.strip()

@app.route('/api/upload', methods=['POST'])
def handle_file_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        df = pd.read_excel(file)
    except Exception as e:
        return jsonify({'error': f'Failed to read the Excel file: {str(e)}'}), 400

    # Trim whitespace from column names
    df.columns = df.columns.str.strip()

    # Ensure the 'Questions' and 'GroundTruth' columns exist
    if 'Questions' not in df.columns or 'GroundTruth' not in df.columns:
        return jsonify({'error': 'Columns "Questions" and "GroundTruth" must be present in the uploaded file.'}), 400
    
    questions = df['Questions'].tolist()
    ground_truths = df['GroundTruth'].tolist()  # Read the GroundTruth column
    responses = []

    for question in questions:
        response = main(question)  # Call your existing main function to get the generated answer
        responses.append({'question': question, 'answer': response})

    # Create a new DataFrame to store the results
    results_df = pd.DataFrame({
        'Questions': questions,
        'Ground Truth': ground_truths,
        'Generated Answer': [res['answer'] for res in responses]
    })

    # Evaluate the similarity using the ChatGPT model
    true_false_list = []

    for gt, generated in zip(ground_truths, responses):
        gpt_evaluation = evaluate_with_gpt(generated['answer'], gt)  # Get GPT's evaluation
        true_false_list.append(gpt_evaluation)

    results_df['True or False'] = true_false_list  # Add the new column

    # **Calculate the Accuracy**:
    total_questions = len(results_df)
    true_count = true_false_list.count('True')
    accuracy = (true_count / total_questions) * 100

    # Append the accuracy result at the bottom
    accuracy_row = pd.DataFrame({
        'Questions': ['Accuracy'],
        'Ground Truth': [''],
        'Generated Answer': [''],
        'True or False': [f'{accuracy:.2f}%']
    })

    results_df = pd.concat([results_df, accuracy_row], ignore_index=True)

    # Save to a new Excel file
    output_file_path = 'generated_responses.xlsx'
    results_df.to_excel(output_file_path, index=False)

    return jsonify({'responses': responses, 'download_link': output_file_path, 'accuracy': f'{accuracy:.2f}%'})


# Route to download the generated Excel file
@app.route('/download', methods=['GET'])
def download_file():
    return send_file('generated_responses.xlsx', as_attachment=True)


import whisper


# Whisper mo del initialization
# Load larger Whisper model for better accuracy
# model = whisper.load_model("medium")  # or "large"

api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("No OPENAI_API_KEY set for Flask application")

# Initialize OpenAI client with the API key from the .env file
lm_client = openai.OpenAI(api_key=api_key)
# Folder for uploading audio files
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Convert audio to WAV format if necessary


def convert_to_wav(file_path):
    file_ext = file_path.rsplit('.', 1)[1].lower()
    if file_ext != 'wav':
        try:
            audio = AudioSegment.from_file(file_path)
            wav_file_path = file_path.rsplit('.', 1)[0] + '.wav'
            audio.export(wav_file_path, format='wav')
            return wav_file_path
        except Exception as e:
            print(f"Conversion error: {e}")
            return file_path  # Return the original path on error
    return file_path

# Transcribe function using Whisper

# Transcribe function using Whisper

def preprocess_audio(file_path):
    # Load audio file
    audio = AudioSegment.from_wav(file_path)
    
    # Noise reduction logic (for example)
    # audio = noise_reduction_function(audio)
    
    # Export processed audio
    processed_path = file_path.replace('.wav', '_processed.wav')
    audio.export(processed_path, format='wav')
    
    return processed_path

import os

def transcribe(audio_file_path):
    if not os.path.isfile(audio_file_path):
        print("Error: File does not exist.")
        return "File does not exist."

    try:
        with open(audio_file_path, 'rb') as audio_file:
            response = lm_client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file, 
                response_format="verbose_json",
                language="en"
            )
            transcription = response.text
    except Exception as e:
        print(f"Error during transcription: {e}")
        transcription = "Error occurred during transcription."

    return transcription

# Route for transcribing audio

# Route for transcribing audio
@app.route('/transcribe_audio', methods=['POST'])
def transcribe_audio():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file uploaded.'}), 400

    audio_file = request.files['audio_file']

    if audio_file.filename == '':
        return jsonify({'error': 'No file selected for upload.'}), 400

    # Secure the filename and save the uploaded file
    filename = secure_filename(audio_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(file_path)

    # Convert to WAV if necessary
    file_path = convert_to_wav(file_path)

    # Transcribe the audio file
    transcription = transcribe(file_path)

    # Clean up by removing the saved file after transcription
    os.remove(file_path)

    return jsonify({
        'transcription': transcription
    })


if __name__ == "__main__":
    app.run(debug=True)  