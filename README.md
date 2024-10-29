
# Capria Web Bot

**Capria Web Bot** is a Flask-based web application that interacts with users by answering their queries using data scraped from web pages and OpenAI's GPT model. It also processes batch questions from uploaded Excel files, returning generated responses in the same format. The chatbot scrapes specific URLs, processes information, and leverages OpenAI’s GPT for intelligent responses.

## Features

- **Web Scraping**: Uses BeautifulSoup to scrape data from specific pages (e.g., Capria’s website) and store relevant information.
- **Natural Language Processing (NLP)**: Integrates OpenAI's GPT model to handle user queries and provide intelligent responses.
- **Batch Processing of Excel Files**: Processes batch queries from an uploaded Excel file and generates responses for each question.
- **Flask Framework**: Provides REST API endpoints for interaction and a simple front-end for file uploads.

## Prerequisites

- **Python 3.12** or higher
- **Flask**: Python web framework
- **Gunicorn**: WSGI HTTP Server for Python
- **Gevent**: Asynchronous I/O library for Python
- **nginx** (optional): For production deployment as a reverse proxy
- **Weaviate**: Vector database used for efficient document search and retrieval
- **OpenAI API**: Integrate with OpenAI for enhanced query interpretation

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/Capria-web-bot.git
   ```

2. Navigate into the project directory:

   ```bash
   cd Capria-web-bot
   ```

3. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Set up the environment variables for the OpenAI API and other services:

   - Create a `.env` file in the root directory of the project with the following content:

     ```bash
     OPENAI_API_KEY=your_openai_api_key
     WEAVIATE_URL=your_weaviate_url
     ```

## Running the Application

### Local Development

1. Start the Flask development server:

   ```bash
   python app.py
   ```

   The app will be available at `http://127.0.0.1:8000/`.

2. You can access the bot and interact via the browser or through `curl`.

### Production Deployment

For production, it's recommended to use **Gunicorn** with **nginx** as a reverse proxy.

1. Run the Gunicorn server:

   ```bash
   gunicorn -k gevent -w 1 -b 0.0.0.0:8000 app:app --timeout 120
   ```

2. Optionally, configure **nginx** to route traffic to the Gunicorn server. Example `nginx` configuration:

   ```nginx
   server {
       listen 80;
       server_name your_domain.com;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

## Systemd Service Setup

To run the application as a background service using **systemd**:

1. Create a new service file:

   ```bash
   sudo nano /etc/systemd/system/capria_web.service
   ```

2. Add the following configuration to the service file:

   ```ini
   [Unit]
   Description=Gunicorn instance for Capria Web Bot
   After=network.target

   [Service]
   User=ubuntu
   Group=www-data
   WorkingDirectory=/home/ubuntu/Capria-web-bot
   Environment="PATH=/home/ubuntu/Capria-web-bot/venv/bin"
   ExecStart=/home/ubuntu/Capria-web-bot/venv/bin/gunicorn -k gevent -w 1 -b 0.0.0.0:8000 app:app --timeout 120

   [Install]
   WantedBy=multi-user.target
   ```

3. Reload systemd and start the service:

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl start capria_web.service
   sudo systemctl enable capria_web.service
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please fork the repository and submit a pull request with your changes. Make sure to add tests for any new features or bug fixes.
