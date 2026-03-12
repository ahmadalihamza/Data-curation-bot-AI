```python
"""Streamlit UI for the Training Data Curation Bot."""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import asyncio
import tempfile
from pathlib import Path

import streamlit as st

from training_data_bot import TrainingDataBot
from training_data_bot.core.config import settings
from training_data_bot.tasks.task_manager import TaskManager


# Page configuration
st.set_page_config(
    page_title="Training Data Curation Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Initialize session state variables."""
    if "bot" not in st.session_state:
        st.session_state.bot = TrainingDataBot()
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "current_dataset" not in st.session_state:
        st.session_state.current_dataset = None
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "datasets" not in st.session_state:
        st.session_state.datasets = {}
    if "jobs" not in st.session_state:
        st.session_state.jobs = {}


def process_documents_async(uploaded_files, url_input, selected_tasks, quality_filter, chunk_size, chunk_overlap):
    """Process documents asynchronously."""
    import shutil

    if not uploaded_files and not url_input:
        st.error("Please upload files or enter URLs")
        return

    sources = []

    temp_dir = tempfile.mkdtemp()

    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        sources.append(file_path)

    if url_input:
        urls = [u.strip() for u in url_input.split("\n") if u.strip()]
        sources.extend(urls)

    bot = st.session_state.bot

    async def run_processing():
        with st.spinner("Loading documents..."):
            documents = await bot.load_documents(sources)

        with st.spinner("Processing documents..."):
            result = await bot.process_documents(
                documents=documents,
                task_types=selected_tasks,
                quality_filter=quality_filter,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        return result

    result = asyncio.run(run_processing())

    st.session_state.processing = False

    if result.get("status") == "success" and result.get("dataset_id"):
        dataset_id = result.get("dataset_id")

        dataset = result.get("dataset")
        if dataset:
            st.session_state.datasets[dataset_id] = dataset

        job_id = result.get("job_id")
        if job_id:
            job = st.session_state.bot.get_job(job_id)
            if job:
                st.session_state.jobs[job_id] = job

    st.session_state.current_dataset = result.get("dataset_id")

    if result.get("status") == "success":
        st.success(f"Processing complete! Generated {result.get('example_count')} examples")
        st.info(f"Job ID: {result.get('job_id')}")
    else:
        st.error(f"Processing failed: {result.get('error', 'Unknown error')}")

    try:
        shutil.rmtree(temp_dir)
    except:
        pass


def export_dataset_async(export_dataset_id, output_filename, export_format, output_dir):
    """Export dataset asynchronously."""

    output_path = os.path.join(output_dir, output_filename)

    dataset = st.session_state.datasets.get(export_dataset_id)

    if not dataset:
        st.error(f"Dataset not found: {export_dataset_id}")
        return

    examples = dataset.get("examples", [])

    if not examples:
        st.error("No examples in dataset. Process documents first.")
        return

    st.info(f"Exporting {len(examples)} examples...")

    result = st.session_state.bot.exporter.export(examples, output_path, export_format)

    if result.get("example_count", 0) > 0:
        st.success(f"Exported {result.get('example_count')} examples to: {result.get('output_path')}")
    else:
        st.error("Export failed: no examples exported")


def main():
    """Main application."""
    init_session_state()

    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")

    task_manager = TaskManager()
    available_tasks = task_manager.get_supported_tasks()

    selected_tasks = st.sidebar.multiselect(
        "Select Task Types",
        available_tasks,
        default=["qa_generation"],
    )

    st.sidebar.markdown("### Processing Options")

    quality_filter = st.sidebar.checkbox("Enable Quality Filter", value=True)

    chunk_size = st.sidebar.slider(
        "Chunk Size",
        500,
        2000,
        settings.CHUNK_SIZE,
        step=100,
    )

    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap",
        0,
        500,
        settings.CHUNK_OVERLAP,
        step=50,
    )

    st.title("📚 Training Data Curation Bot")
    st.markdown("Generate high-quality training data from your documents using AI")

    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Results", "💾 Export"])


    # ---------------- TAB 1 ----------------
    with tab1:

        st.header("Upload Documents")

        col1, col2 = st.columns(2)

        with col1:
            uploaded_files = st.file_uploader(
                "Choose files",
                type=["txt", "md", "pdf", "docx", "json", "xml", "csv"],
                accept_multiple_files=True,
            )

        with col2:
            url_input = st.text_area("Enter URLs (one per line)")

        st.divider()

        if st.button("🚀 Process Documents", use_container_width=True):

            process_documents_async(
                uploaded_files,
                url_input,
                selected_tasks,
                quality_filter,
                chunk_size,
                chunk_overlap,
            )


    # ---------------- TAB 2 ----------------
    with tab2:

        st.header("Results")

        datasets = list(st.session_state.datasets.values())

        if datasets:

            selected_dataset = st.selectbox(
                "Select Dataset",
                options=[d["id"] for d in datasets],
            )

            dataset = st.session_state.datasets.get(selected_dataset)

            if dataset:

                examples = dataset.get("examples", [])

                st.metric("Total Examples", len(examples))

                st.subheader("Examples")

                for i, example in enumerate(examples[:20]):
                    with st.expander(f"Example {i+1}"):
                        st.json(example)

        else:
            st.info("No datasets yet.")


    # ---------------- TAB 3 ----------------
    with tab3:

        st.header("Export Dataset")

        datasets = list(st.session_state.datasets.values())

        if datasets:

            col1, col2 = st.columns(2)

            with col1:
                export_dataset_id = st.selectbox(
                    "Select Dataset",
                    options=[d["id"] for d in datasets],
                )

            with col2:
                export_format = st.selectbox(
                    "Export Format",
                    ["jsonl", "json", "csv", "zip"],
                )

            output_filename = st.text_input(
                "Output Filename",
                value=f"training_data.{export_format}",
            )

            output_dir = st.text_input(
                "Output Directory",
                value="/tmp",
            )

            if st.button("💾 Export Dataset", use_container_width=True):

                export_dataset_async(
                    export_dataset_id,
                    output_filename,
                    export_format,
                    output_dir,
                )

                file_path = os.path.join(output_dir, output_filename)

                if os.path.exists(file_path):

                    with open(file_path, "rb") as f:

                        st.download_button(
                            "⬇ Download File",
                            data=f,
                            file_name=output_filename,
                            mime="application/octet-stream",
                        )

                else:
                    st.error("Export failed: file not found")

        else:
            st.info("No datasets to export.")


    # -------- SIDEBAR STATUS --------
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Status")

    doc_count = len(st.session_state.bot.list_documents())
    st.sidebar.metric("Documents Loaded", doc_count)

    dataset_count = len(st.session_state.datasets)
    st.sidebar.metric("Datasets Created", dataset_count)

    job_count = len(st.session_state.jobs)
    st.sidebar.metric("Jobs", job_count)


if __name__ == "__main__":
    main()
```
