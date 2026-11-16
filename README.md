# 🎙️ AI Apocalypse? The Podcast Studio

A Python application that transforms research papers about artificial intelligence into a conversational podcast episode. The application uses OpenAI to generate a structured two-host script and create spoken audio for the final episode.

# About the Project

AI Apocalypse? The Podcast examines concerns about artificial intelligence, including:


Potentially harmful effects of AI chatbots
Existential risks and the possibility of AI going rogue
Evidence versus speculation in discussions about AI
Practical implications for professionals and society

The podcast uses two hosts, Ana and Rui, who discuss the source papers in an informative, balanced, and conversational style.

The project is designed for AI-curious professionals and non-experts who want to understand AI risks without unnecessary technical language or sensationalism.

# Features
Extracts text from PDF research papers
Supports multiple PDF source documents
Generates a structured podcast script using the OpenAI API
Uses two podcast hosts: Ana and Rui
Generates spoken audio for each host
Combines the audio segments into one MP3 file
Provides a Gradio web interface
Displays the generated transcript
Displays the sources used in the episode
Allows users to upload their own PDF sources
Allows users to load a previously generated episode

#Project Structure

Podcast Project/
├── source_material/
│   ├── AI and the End of the World.pdf
│   ├── AI going rogue.pdf
│   └── Potentially Harmful Consequences of AI.pdf
├── src/
│   ├── data_processor.py
│   ├── hosts.py
│   ├── llm_processor.py
│   ├── main.py
│   └── tts_generator.py
├── outputs/
│   ├── episode_script.json
│   ├── episode_sources.json
│   └── final_episode.mp3
├── requirements.txt
├── README.md
├── .gitignore
└── .env

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"
```
## Installation

Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run

Put source material in a text file and run:

```bash
python src/main.py input.txt --output outputs/episode.mp3
```

Generated audio is kept in `outputs/`, which is ignored by Git except for its `.gitkeep` placeholder.

### Install FFmpeg for audio generation

Podcast Studio uses `pydub` and FFmpeg to combine Ana and Rui’s audio clips
into one MP3.

1. Install the Python dependencies from the project folder:

   ```bash
   pip install -r requirements.txt
   ```

2. Install FFmpeg separately. On Windows, extract the FFmpeg download and
   add the `bin` folder that contains `ffmpeg.exe` to your Windows `PATH`.
3. Reopen the VS Code terminal and check that FFmpeg is available:

   ```bash
   ffmpeg -version
   ```

   If the terminal shows version information, you’re ready to generate audio.
   The finished episode is saved to `outputs/final_episode.mp3`.

## Sources


The podcast episode is generated from three PDF papers stored in
`source_material/`. The hosts are instructed to credit authors and
publications aloud when discussing the papers.

| File | Authors | Publication | Type |
|---|---|---|---|
| AI and the End of the World.pdf | Alexander Rogul | Working paper, Bentley University, 2026 | Working paper |
| AI going rogue.pdf | Jascha Bareis, Clemens Ackerl, Reinhard Heil | AI and Ethics, 2026 | Peer-reviewed narrative review |
| Potentially Harmful Consequences of AI.pdf | Sidse Godske Olsen, Christian Jon Reinecke-Tellefsen, Søren Dinesen Østergaard | Acta Psychiatrica Scandinavica, 2026 | Peer-reviewed research letter |

### Full references

1. Rogul, A. (2026). *AI & the End of the World: What the Data Say*. Working paper, Bentley University.

2. Bareis, J., Ackerl, C., & Heil, R. (2026). *AI going rogue? An integrative narrative review of the tacit assumptions underlying existential AI-risks*. *AI and Ethics, 6*, 152. https://doi.org/10.1007/s43681-025-00928-w

3. Olsen, S. G., Reinecke-Tellefsen, C. J., & Østergaard, S. D. (2026). *Potentially harmful consequences of artificial intelligence (AI) chatbot use among patients with mental illness: Early data from a large psychiatric service system*. *Acta Psychiatrica Scandinavica, 153*, 301–303. https://doi.org/10.1111/acps.70068

### How source material is used

1. `src/data_processor.py` finds the PDF files in `source_material/`.
2. It extracts readable text from each PDF.
3. The extracted text is combined and sent to the language model.
4. `src/llm_processor.py` generates a structured JSON podcast script.
5. The JSON script is sent to `src/tts_generator.py` for audio generation.

The current script structure contains:

- the episode title;
- speaker segments;
- the speaker name (`Ana` or `Rui`);
- the spoken text for each segment.
