import html
import json
import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

# Allow running as `python src/main.py` (or VS Code's Run button) as well as
# `python -m src.main`: put the project folder on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

from src.data_processor import from_multiple_pdfs
from src.llm_processor import generate_podcast_script
from src.tts_generator import combine_audio_segments, generate_script_segments


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "source_material"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def create_openai_client() -> OpenAI:
    """Load the local environment file and create an OpenAI client."""

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY was not found.")

    return OpenAI(api_key=api_key)


def run_pipeline(
    episode_title: str = "The Future of AI: Catastrophe or Control?",
    pdf_files: list[str | Path] | None = None,
    progress=None,
) -> Path:
    """Run the complete podcast-generation pipeline.

    progress, if given, is called as progress(fraction, description) at each
    stage so a UI can show how far along the episode is.
    """

    def report(fraction: float, description: str) -> None:
        print(f"[{fraction:>4.0%}] {description}")
        if progress is not None:
            progress(fraction, desc=description)

    report(0.02, "Connecting to OpenAI...")
    client = create_openai_client()
    if pdf_files is None:
        pdf_files = sorted(SOURCE_DIR.glob("*.pdf"))

    report(0.05, "Reading source PDFs...")
    content = from_multiple_pdfs(pdf_files)

    source_text = content["text"]
    source_details = "\n".join(
        (
            f"- Title: {source['full_title']}\n"
            f"  Authors: {source['author']}\n"
            f"  Year: {source['year']}\n"
            f"  Type: {source['type']}"
        )
        for source in content["sources"]
    )

    report(0.10, "Writing the podcast script (this can take a minute)...")
    podcast_script = generate_podcast_script(
        client=client,
        source_text=source_text,
        episode_title=episode_title,
        source_details=source_details,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_script_path = OUTPUT_DIR / "episode_script.json"
    saved_audio_path = OUTPUT_DIR / "final_episode.mp3"
    saved_sources_path = OUTPUT_DIR / "episode_sources.json"

    # Build everything in a temporary folder first. The existing episode is
    # replaced only after both the new script and new audio are ready.
    with tempfile.TemporaryDirectory(
        prefix="podcast-build-",
        dir=OUTPUT_DIR,
    ) as temporary_folder:
        temporary_folder = Path(temporary_folder)
        temporary_script_path = temporary_folder / "episode_script.json"
        temporary_audio_path = temporary_folder / "final_episode.mp3"
        temporary_sources_path = temporary_folder / "episode_sources.json"

        temporary_script_path.write_text(
            json.dumps(podcast_script, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_sources_path.write_text(
            json.dumps(content["sources"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Audio is the slowest step, so it gets 30%-90% of the bar.
        report(0.30, "Recording voices...")

        def on_segment(done: int, total: int) -> None:
            report(
                0.30 + 0.60 * done / total,
                f"Recording voices: segment {done} of {total}",
            )

        segment_paths = generate_script_segments(
            client=client,
            script=podcast_script,
            output_dir=temporary_folder / "segments",
            on_progress=on_segment,
        )

        report(0.93, "Mixing the final episode...")
        combine_audio_segments(
            segment_paths=segment_paths,
            output_path=temporary_audio_path,
        )

        # Both files exist only after all generation steps above succeed.
        os.replace(temporary_script_path, saved_script_path)
        os.replace(temporary_audio_path, saved_audio_path)
        os.replace(temporary_sources_path, saved_sources_path)

    print(f"Saved script to: {saved_script_path}")
    print(f"Saved source metadata to: {saved_sources_path}")
    print(f"Generated audio segments: {len(segment_paths)}")
    print(f"Saved final audio to: {saved_audio_path}")

    return saved_audio_path


def format_script_for_display(script: dict) -> str:
    """Convert the structured script into readable Markdown."""

    lines = [f"# {script['title']}", ""]

    for segment in script["segments"]:
        lines.append(f"**{segment['speaker']}:** {segment['text']}")
        lines.append("")

    return "\n".join(lines)


def format_source_references(sources: list[dict]) -> str:
    """Format source metadata for display in the Gradio interface."""

    if not sources:
        return "No PDF sources selected."

    lines = []

    for index, source in enumerate(sources, start=1):
        title = source.get("full_title") or source.get("file_name", "Unknown PDF")
        author = source.get("author") or "Author unavailable"
        details = f"{index}. **{author}** — *{title}*"

        if source.get("publication"):
            details += f" — {source['publication']}"

        if source.get("year"):
            details += f" ({source['year']})"

        lines.append(details)

    return "\n\n".join(lines)


def selected_pdf_paths(pdf_files: list[str | Path] | None) -> list[Path]:
    """Use uploaded PDFs, or fall back to the bundled source PDFs."""

    if pdf_files:
        return [Path(pdf_file) for pdf_file in pdf_files]

    return sorted(SOURCE_DIR.glob("*.pdf"))


def preview_sources(pdf_files: list[str | Path] | None) -> str:
    """Read source metadata locally for the UI without making an API call."""

    try:
        content = from_multiple_pdfs(selected_pdf_paths(pdf_files))
    except (FileNotFoundError, ValueError) as error:
        return f"**Source error:** {error}"

    return format_source_references(content["sources"])


def load_existing_episode():
    """Load the already-generated transcript and MP3 for the interface."""

    script_path = OUTPUT_DIR / "episode_script.json"
    audio_path = OUTPUT_DIR / "final_episode.mp3"
    sources_path = OUTPUT_DIR / "episode_sources.json"

    if not script_path.exists():
        raise FileNotFoundError(f"Transcript not found: {script_path}")

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    script = json.loads(script_path.read_text(encoding="utf-8"))
    transcript = format_script_for_display(script)

    if sources_path.exists():
        sources = json.loads(sources_path.read_text(encoding="utf-8"))
    else:
        sources = from_multiple_pdfs(selected_pdf_paths(None))["sources"]

    # A readable copy of the transcript for the download button.
    transcript_path = OUTPUT_DIR / "episode_transcript.md"
    transcript_path.write_text(transcript, encoding="utf-8")

    return (
        transcript,
        str(audio_path),
        format_source_references(sources),
        str(transcript_path),
    )


# Generation stages shown in the status card, with the progress fraction at
# which each one starts (matching the report() calls in run_pipeline).
PIPELINE_STEPS = [
    ("Read sources", 0.0),
    ("Write script", 0.10),
    ("Record voices", 0.30),
    ("Mix episode", 0.93),
]


def render_status(
    state: str,
    message: str,
    fraction: float = 0.0,
    elapsed: float | None = None,
) -> str:
    """Build the HTML status card.

    state is one of: idle, running, done, error.
    """

    message = html.escape(message)

    if state != "running":
        icon = {"idle": "○", "done": "✓", "error": "!"}[state]
        return (
            f'<div class="status-card status-{state}">'
            f'<div class="status-line"><span class="status-icon">{icon}</span>'
            f"<span>{message}</span></div></div>"
        )

    active = max(
        index
        for index, (_, start) in enumerate(PIPELINE_STEPS)
        if fraction >= start
    )
    steps_html = "".join(
        f'<li class="step step-'
        f'{"done" if index < active else "active" if index == active else "todo"}">'
        f'<span class="step-dot"></span>{name}</li>'
        for index, (name, _) in enumerate(PIPELINE_STEPS)
    )
    elapsed_text = ""
    if elapsed is not None:
        minutes, seconds = divmod(int(elapsed), 60)
        elapsed_text = f'<span class="status-time">{minutes}:{seconds:02d}</span>'

    return f"""
    <div class="status-card status-running">
      <div class="status-line">
        <span class="status-spinner"></span>
        <span class="status-message">{message}</span>
        <span class="status-percent">{fraction:.0%}</span>
        {elapsed_text}
      </div>
      <div class="status-track">
        <div class="status-fill" style="width: {fraction * 100:.1f}%"></div>
      </div>
      <ol class="status-steps">{steps_html}</ol>
    </div>
    """


def launch_interface():
    """Create the Podcast Studio interface."""

    import gradio as gr

    def episode_outputs(status_html, button_label="Generate Podcast Episode"):
        """Outputs for a finished (or failed) run: re-enable the button and
        fill the player, transcript, sources and download buttons."""

        transcript, audio_path, source_references, transcript_path = (
            load_existing_episode()
        )
        return (
            status_html,
            gr.Button(value=button_label, interactive=True),
            audio_path,
            transcript,
            source_references,
            gr.DownloadButton(value=audio_path, visible=True),
            gr.DownloadButton(value=transcript_path, visible=True),
        )

    def failed_outputs(message):
        return (
            render_status("error", message),
            gr.Button(value="Try Again", interactive=True),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )

    def create_episode(pdf_files):
        """Generate a fresh episode, streaming progress into the status card."""

        pdf_paths = selected_pdf_paths(pdf_files)
        updates = queue.Queue()
        result = {}

        def report(fraction, desc=""):
            updates.put((fraction, desc))

        def worker():
            # Always run the complete pipeline. This makes fresh API calls and
            # reflects any changes to the source PDFs or prompt.
            try:
                run_pipeline(pdf_files=pdf_paths, progress=report)
            except Exception as error:  # reported back to the UI below
                result["error"] = error

        started = time.monotonic()
        fraction, message = 0.0, "Starting..."
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        yield (
            render_status("running", message, fraction, 0),
            gr.Button(value="Generating...", interactive=False),
            None,
            "",
            preview_sources(pdf_files),
            gr.DownloadButton(visible=False),
            gr.DownloadButton(visible=False),
        )

        # Refresh at least once a second so the elapsed timer keeps ticking,
        # which also shows the app hasn't frozen during long API calls.
        while thread.is_alive() or not updates.empty():
            try:
                fraction, message = updates.get(timeout=1)
            except queue.Empty:
                pass
            yield (
                render_status(
                    "running", message, fraction, time.monotonic() - started
                ),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
            )

        if "error" in result:
            error = result["error"]
            print(f"Episode generation failed: {error!r}")
            if isinstance(error, (FileNotFoundError, ValueError)):
                yield failed_outputs(f"Generation failed: {error}")
            else:
                yield failed_outputs(
                    "Could not generate the episode. Check the VS Code "
                    "terminal for details."
                )
            return

        minutes, seconds = divmod(int(time.monotonic() - started), 60)
        yield episode_outputs(
            render_status(
                "done",
                f"New episode ready to play (made in {minutes}:{seconds:02d}).",
            )
        )

    def load_saved_episode():
        """Load the existing episode without making API calls."""

        try:
            return episode_outputs(
                render_status("done", "Saved episode loaded and ready to play.")
            )
        except FileNotFoundError as error:
            return failed_outputs(str(error))
        except Exception as error:
            print(f"Saved episode could not be loaded: {error!r}")
            return failed_outputs(
                "Could not load the saved episode. Check the VS Code terminal "
                "for details."
            )

    source_references = """
    1. **Rogul, A. (2026).**
       *AI & the End of the World: What the Data Say.*
       Working paper, Bentley University.

    2. **Bareis, J., Ackerl, C., & Heil, R. (2026).**
       *AI going rogue? An integrative narrative review of the tacit assumptions underlying existential AI-risks.*
       **AI and Ethics, 6**, 152. DOI: 10.1007/s43681-025-00928-w

    3. **Olsen, S. G., Reinecke-Tellefsen, C. J., & Østergaard, S. D. (2026).**
       *Potentially harmful consequences of artificial intelligence (AI) chatbot use among patients with mental illness: Early data from a large psychiatric service system.*
       **Acta Psychiatrica Scandinavica, 153**, 301–303. DOI: 10.1111/acps.70068
    """

    # The displayed sources are refreshed when files are uploaded and after
    # a new episode is generated.
    source_references = preview_sources(None)

    css = """
    :root {
        --studio-bg: #090d17;
        --studio-panel: #202736;
        --studio-panel-2: #171d2a;
        --studio-border: #354056;
        --studio-text: #f5f7fb;
        --studio-muted: #b9c2d2;
    }
    body, .gradio-container {
        background: var(--studio-bg) !important;
        color: var(--studio-text) !important;
    }
    .gradio-container { max-width: 100% !important; }
    .gradio-container .block,
    .gradio-container .form,
    .gradio-container .panel,
    .gradio-container .gr-box,
    .gradio-container .gr-group,
    .gradio-container .gradio-group,
    .gradio-container .gradio-accordion,
    .gradio-container fieldset {
        background: var(--studio-panel) !important;
        border-color: var(--studio-border) !important;
    }
    .app-container { max-width: 1100px; margin: auto; padding: 24px 0 48px; }
    .hero {
        padding: 28px;
        border-radius: 16px;
        background: linear-gradient(135deg, #252d3e, #171c2b) !important;
        border: 1px solid var(--studio-border);
    }
    .hero { text-align: center; }
    .hero h1, .hero p { color: var(--studio-text) !important; }
    .hero h1 { font-size: 2.2rem; font-weight: 800; margin: 0 0 10px; }
    .hero p { color: var(--studio-muted) !important; font-size: 1rem; margin: 0; }
    .panel, .gradio-group, .gradio-accordion {
        border-radius: 14px !important;
        border: 1px solid var(--studio-border) !important;
        background: var(--studio-panel) !important;
    }
    .gradio-container label, .gradio-container .block-title,
    .gradio-container .wrap, .gradio-container p,
    .gradio-container h1, .gradio-container h2,
    .gradio-container h3 { color: var(--studio-text) !important; }
    .gradio-container input, .gradio-container textarea {
        background: var(--studio-panel-2) !important;
        color: var(--studio-text) !important;
        border-color: var(--studio-border) !important;
    }
    #source-upload,
    #source-upload > div,
    #source-upload .wrap,
    #source-upload .container,
    #source-upload [data-testid="file-upload"],
    #source-upload .file-dropzone,
    #source-upload .upload-container,
    #source-upload .file-preview {
        background: var(--studio-panel-2) !important;
        color: var(--studio-text) !important;
        border-color: var(--studio-border) !important;
    }
    #source-upload label,
    #source-upload span,
    #source-upload p,
    #source-upload svg {
        color: var(--studio-text) !important;
        fill: var(--studio-text) !important;
    }
    #source-upload button,
    #source-upload [role="button"],
    #source-upload .upload-button,
    #source-upload input[type="file"] {
        background: var(--studio-panel) !important;
        color: var(--studio-text) !important;
        border-color: var(--studio-border) !important;
    }
    #source-upload * {
        border-color: var(--studio-border) !important;
    }
    #source-upload [data-testid="block-label"],
    #source-upload .block-label,
    #source-upload label[data-testid="block-label"] {
        background: var(--studio-panel) !important;
        color: var(--studio-text) !important;
        border-color: var(--studio-border) !important;
    }
    .gradio-container .audio-container,
    .gradio-container audio {
        background: var(--studio-panel-2) !important;
        color-scheme: dark;
    }
    .gradio-container .prose,
    .gradio-container .markdown,
    .gradio-container .markdown-text {
        color: var(--studio-text) !important;
    }
    .gradio-container .prose h1,
    .gradio-container .prose h2,
    .gradio-container .prose h3,
    .gradio-container .markdown h1,
    .gradio-container .markdown h2,
    .gradio-container .markdown h3 {
        border: 0 !important;
        border-bottom: 0 !important;
        box-shadow: none !important;
        color: var(--studio-text) !important;
    }
    .source-references,
    .source-references * {
        color: var(--studio-text) !important;
        font-size: 0.82rem !important;
        line-height: 1.45 !important;
    }
    .gradio-container .markdown-text a { color: #f472b6 !important; }
    .section-heading h2,
    .gradio-accordion > button,
    .gradio-accordion .label {
        color: var(--studio-text) !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
    }
    #podcast-transcript > button,
    #podcast-transcript > button *,
    #podcast-transcript button span {
        color: var(--studio-text) !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
    }
    .transcript-content,
    .transcript-content * {
        color: var(--studio-text) !important;
        font-size: 1rem !important;
        line-height: 1.65 !important;
    }
    .transcript-content strong,
    .transcript-content b {
        color: var(--studio-text) !important;
        font-weight: 800 !important;
    }
    .gradio-accordion > button svg {
        color: var(--studio-text) !important;
        fill: var(--studio-text) !important;
        stroke: var(--studio-text) !important;
    }
    #podcast-transcript button svg {
        color: var(--studio-text) !important;
        fill: var(--studio-text) !important;
        stroke: var(--studio-text) !important;
    }
    /* Audio player: playback-speed button (bottom-left corner), the
       download / share icon buttons, and their wrapper. */
    .gradio-container .playback,
    .gradio-container .icon-button-wrapper,
    .gradio-container .icon-button,
    .gradio-container .icon-button-wrapper button,
    .gradio-container .icon-button-wrapper a,
    .gradio-container a.download-link,
    .gradio-container .control-wrapper button,
    .gradio-container .settings-wrapper button {
        background: var(--studio-panel) !important;
        color: var(--studio-text) !important;
        border-color: var(--studio-border) !important;
    }
    .gradio-container .icon-button:hover,
    .gradio-container .playback:hover,
    .gradio-container a.download-link:hover {
        background: var(--studio-panel-2) !important;
    }
    .gradio-container .icon-button svg,
    .gradio-container a.download-link svg {
        color: var(--studio-text) !important;
    }
    /* Status card: idle / running (progress bar + steps) / done / error. */
    .status-card {
        background: var(--studio-panel);
        border: 1px solid var(--studio-border);
        border-radius: 14px;
        padding: 16px 20px;
        color: var(--studio-text);
    }
    .status-done { border-color: #10b981; background: rgba(16, 185, 129, 0.08); }
    .status-error { border-color: #f43f5e; background: rgba(244, 63, 94, 0.08); }
    .status-line {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 1rem;
        font-weight: 600;
    }
    .status-icon {
        display: inline-grid;
        place-items: center;
        width: 24px;
        height: 24px;
        flex: none;
        border-radius: 50%;
        font-size: 0.85rem;
        background: var(--studio-panel-2);
        color: var(--studio-muted);
    }
    .status-done .status-icon { background: #10b981; color: #fff; }
    .status-error .status-icon { background: #f43f5e; color: #fff; }
    .status-idle { color: var(--studio-muted); }
    .status-message { flex: 1; }
    .status-percent { font-variant-numeric: tabular-nums; color: #f472b6; }
    .status-time {
        font-variant-numeric: tabular-nums;
        color: var(--studio-muted);
        font-weight: 500;
        min-width: 3.2em;
        text-align: right;
    }
    .status-spinner {
        width: 18px;
        height: 18px;
        flex: none;
        border-radius: 50%;
        border: 2.5px solid var(--studio-border);
        border-top-color: #db2777;
        animation: status-spin 0.8s linear infinite;
    }
    @keyframes status-spin { to { transform: rotate(360deg); } }
    .status-track {
        height: 8px;
        margin: 14px 0 12px;
        border-radius: 999px;
        background: var(--studio-panel-2);
        overflow: hidden;
    }
    .status-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #7c3aed, #db2777);
        transition: width 0.6s ease;
    }
    .status-steps {
        display: flex;
        flex-wrap: wrap;
        gap: 6px 22px;
        list-style: none;
        margin: 0;
        padding: 0;
        font-size: 0.85rem;
    }
    .step { display: flex; align-items: center; gap: 8px; }
    .step-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--studio-border);
    }
    .step-todo { color: var(--studio-muted); opacity: 0.6; }
    .step-done { color: var(--studio-muted); }
    .step-done .step-dot { background: #10b981; }
    .step-active { color: var(--studio-text); font-weight: 700; }
    .step-active .step-dot {
        background: #db2777;
        box-shadow: 0 0 0 4px rgba(219, 39, 119, 0.25);
    }
    /* Download buttons under the audio player. */
    .download-row { padding: 12px; gap: 12px; }
    .download-button { font-weight: 700 !important; }
    .download-button.primary {
        background: linear-gradient(90deg, #7c3aed, #db2777) !important;
        color: white !important;
        border: 0 !important;
    }
    .play-button {
        font-size: 18px !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #7c3aed, #db2777) !important;
        color: white !important;
        border: 0 !important;
    }
    """

    with gr.Blocks(
        title="🎙️ AI Apocalypse? The Podcast Studio",
        css=css,
        theme=gr.themes.Base(
            primary_hue="pink",
            secondary_hue="purple",
            neutral_hue="slate",
        ).set(
            # Force the dark studio palette on every surface (light and dark
            # mode), including the label chip in the upload box's corner.
            body_background_fill="#090d17",
            body_background_fill_dark="#090d17",
            background_fill_primary="#171d2a",
            background_fill_primary_dark="#171d2a",
            background_fill_secondary="#202736",
            background_fill_secondary_dark="#202736",
            block_background_fill="#202736",
            block_background_fill_dark="#202736",
            block_label_background_fill="#202736",
            block_label_background_fill_dark="#202736",
            block_label_text_color="#f5f7fb",
            block_label_text_color_dark="#f5f7fb",
            block_label_border_color="#354056",
            block_label_border_color_dark="#354056",
            block_title_background_fill="#202736",
            block_title_background_fill_dark="#202736",
            panel_background_fill="#202736",
            panel_background_fill_dark="#202736",
            border_color_primary="#354056",
            border_color_primary_dark="#354056",
            body_text_color="#f5f7fb",
            body_text_color_dark="#f5f7fb",
            body_text_color_subdued="#b9c2d2",
            body_text_color_subdued_dark="#b9c2d2",
            # Secondary/icon buttons (e.g. audio download) default to white.
            button_secondary_background_fill="#202736",
            button_secondary_background_fill_dark="#202736",
            button_secondary_background_fill_hover="#171d2a",
            button_secondary_background_fill_hover_dark="#171d2a",
            button_secondary_text_color="#f5f7fb",
            button_secondary_text_color_dark="#f5f7fb",
            button_secondary_border_color="#354056",
            button_secondary_border_color_dark="#354056",
        ),
    ) as demo:
        with gr.Column(elem_classes="app-container"):
            gr.Markdown(
                """
                <div class="hero">
                <h1>🎙️ AI Apocalypse? The Podcast Studio</h1>
                <p>Listen to a conversational podcast created from the project's source papers.</p>
                </div>
                """
            )

            pdf_upload = gr.File(
                label="Upload PDF sources (optional)",
                file_count="multiple",
                file_types=[".pdf"],
                type="filepath",
                elem_id="source-upload",
            )

            with gr.Row():
                generate_button = gr.Button(
                    "Generate Podcast Episode",
                    variant="primary",
                    elem_classes="play-button",
                )
                load_button = gr.Button(
                    "Load Saved Episode",
                    variant="secondary",
                )

            status_output = gr.HTML(
                render_status(
                    "idle",
                    "No episode loaded yet. Generate a new one or load the "
                    "saved episode.",
                )
            )

            with gr.Group(elem_classes="panel"):
                gr.Markdown("## Podcast Audio", elem_classes="section-heading")
                audio_output = gr.Audio(
                    show_label=False,
                    type="filepath",
                    buttons=[],
                )
                with gr.Row(elem_classes="download-row"):
                    download_audio_button = gr.DownloadButton(
                        "⬇  Download Episode (MP3)",
                        visible=False,
                        variant="primary",
                        elem_classes="download-button",
                    )
                    download_transcript_button = gr.DownloadButton(
                        "⬇  Download Transcript",
                        visible=False,
                        variant="secondary",
                        elem_classes="download-button",
                    )

            with gr.Accordion(
                "Podcast Transcript",
                open=True,
                elem_id="podcast-transcript",
            ):
                transcript_output = gr.Markdown(
                    "",
                    elem_classes="transcript-content",
                )

            with gr.Group(elem_classes="panel"):
                gr.Markdown("## Sources", elem_classes="section-heading")
                source_references_output = gr.Markdown(
                    source_references,
                    elem_classes="source-references",
                )

            pdf_upload.change(
                fn=preview_sources,
                inputs=[pdf_upload],
                outputs=[source_references_output],
            )

            episode_outputs_list = [
                status_output,
                generate_button,
                audio_output,
                transcript_output,
                source_references_output,
                download_audio_button,
                download_transcript_button,
            ]

            # show_progress="hidden" turns off Gradio's default overlay on
            # every output; the status card shows progress instead.
            generate_button.click(
                fn=create_episode,
                inputs=[pdf_upload],
                outputs=episode_outputs_list,
                show_progress="hidden",
            )

            load_button.click(
                fn=load_saved_episode,
                inputs=[],
                outputs=episode_outputs_list,
                show_progress="hidden",
            )

    return demo


if __name__ == "__main__":
    app = launch_interface()
    app.launch(share=True, allowed_paths=[str(OUTPUT_DIR)])
