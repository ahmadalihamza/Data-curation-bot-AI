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

    # Save uploaded files temporarily
    temp_dir = tempfile.mkdtemp()
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        sources.append(file_path)

    # Add URLs
    if url_input:
        urls = [u.strip() for u in url_input.split("\n") if u.strip()]
        sources.extend(urls)

    # Run async operations
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
    
    # Also store in session state for reliability
    if result.get("status") == "success" and result.get("dataset_id"):
        dataset_id = result.get("dataset_id")
        # Get dataset directly from result (most reliable)
        dataset = result.get("dataset")
        if dataset:
            st.session_state.datasets[dataset_id] = dataset
        
        # Get job info and store in session state
        job_id = result.get("job_id")
        if job_id:
            job = st.session_state.bot.get_job(job_id)
            if job:
                st.session_state.jobs[job_id] = job
    
    st.session_state.current_dataset = result.get("dataset_id")

    # Show results
    if result.get("status") == "success":
        st.success(f"Processing complete! Generated {result.get('example_count')} examples")
        st.info(f"Job ID: {result.get('job_id')}")
    else:
        st.error(f"Processing failed: {result.get('error', 'Unknown error')}")

    # Cleanup temp files
    try:
        shutil.rmtree(temp_dir)
    except:
        pass


def export_dataset_async(export_dataset_id, output_filename, export_format, output_dir):
    """Export dataset asynchronously."""
    output_path = os.path.join(output_dir, output_filename)
    
    # Get the dataset from session state
    dataset = st.session_state.datasets.get(export_dataset_id)
    
    if not dataset:
        st.error(f"Dataset not found: {export_dataset_id}")
        return
    
    examples = dataset.get("examples", [])
    
    if not examples:
        st.error("No examples in dataset. Process documents first.")
        return
    
    st.info(f"Exporting {len(examples)} examples...")
    
    # Export directly using the exporter
    result = st.session_state.bot.exporter.export(examples, output_path, export_format)

    if result.get("example_count", 0) > 0:
        st.success(f"Exported {result.get('example_count')} examples to: {result.get('output_path')}")
    else:
        st.error(f"Export failed: no examples exported")


def main():
    """Main application."""
    init_session_state()

    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")

    # Task selection
    task_manager = TaskManager()
    available_tasks = task_manager.get_supported_tasks()

    selected_tasks = st.sidebar.multiselect(
        "Select Task Types",
        available_tasks,
        default=["qa_generation"],
        help="Choose the types of training data to generate",
    )

    # Processing options
    st.sidebar.markdown("### Processing Options")
    quality_filter = st.sidebar.checkbox(
        "Enable Quality Filter",
        value=True,
        help="Filter out low-quality examples",
    )

    chunk_size = st.sidebar.slider(
        "Chunk Size",
        min_value=500,
        max_value=2000,
        value=settings.CHUNK_SIZE,
        step=100,
        help="Size of text chunks for processing",
    )

    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=settings.CHUNK_OVERLAP,
        step=50,
        help="Overlap between text chunks",
    )

    # Main content
    st.title("📚 Training Data Curation Bot")
    st.markdown("Generate high-quality training data from your documents using AI")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Results", "💾 Export"])

    # Tab 1: Upload and Process
    with tab1:
        st.header("Upload Documents")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("File Upload")
            uploaded_files = st.file_uploader(
                "Choose files",
                type=["txt", "md", "pdf", "docx", "json", "xml", "csv"],
                accept_multiple_files=True,
                help="Upload text, PDF, or DOCX files",
            )

            if uploaded_files:
                st.success(f"Uploaded {len(uploaded_files)} file(s)")

        with col2:
            st.subheader("URL Input")
            url_input = st.text_area(
                "Enter URLs (one per line)",
                height=100,
                help="Enter web URLs to fetch content from",
            )

        st.divider()

        # Process button
        if st.button(
            "🚀 Process Documents",
            type="primary",
            disabled=st.session_state.processing,
            use_container_width=True,
        ):
            st.session_state.processing = True
            process_documents_async(
                uploaded_files, url_input, selected_tasks, quality_filter, chunk_size, chunk_overlap
            )

    # Tab 2: Results
    with tab2:
        st.header("Results")

        # Get datasets from session state
        datasets = list(st.session_state.datasets.values())

        if datasets:
            # Dataset selector - add example_count to display options
            dataset_options = []
            for d in datasets:
                d_with_count = dict(d)
                d_with_count['example_count'] = len(d.get('examples', []))
                dataset_options.append(d_with_count)
            
            selected_dataset = st.selectbox(
                "Select Dataset",
                options=[d["id"] for d in dataset_options],
                format_func=lambda x: f"Dataset {x[:8]}... ({[d for d in dataset_options if d['id'] == x][0]['example_count']} examples)",
            )

            if selected_dataset:
                dataset = st.session_state.datasets.get(selected_dataset)
                if dataset:
                    examples = dataset.get("examples", [])

                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Examples", len(examples))
                    with col2:
                        task_types = dataset.get("task_types", [])
                        st.metric("Task Types", ", ".join(task_types))
                    with col3:
                        st.metric("Source Docs", dataset.get("document_count", 0))

                    st.divider()

                    # Filter by task type
                    task_filter = st.selectbox(
                        "Filter by Task Type",
                        options=["All"] + list(set(ex.get("task_type", "") for ex in examples)),
                    )

                    filtered_examples = examples
                    if task_filter != "All":
                        filtered_examples = [ex for ex in examples if ex.get("task_type") == task_filter]

                    # Show examples
                    st.subheader(f"Examples ({len(filtered_examples)})")

                    for i, example in enumerate(filtered_examples[:20]):
                        with st.expander(f"Example {i+1} - {example.get('task_type', 'unknown')}"):
                            st.json(example)
                else:
                    st.warning("Dataset not found")
        else:
            st.info("No datasets yet. Upload and process documents to generate training data.")

    # Tab 3: Export
    with tab3:
        st.header("Export Dataset")

        datasets = list(st.session_state.datasets.values())

        if datasets:
            # Export settings
            col1, col2 = st.columns(2)

            with col1:
                export_dataset_id = st.selectbox(
                    "Select Dataset to Export",
                    options=[d["id"] for d in datasets],
                    format_func=lambda x: f"Dataset {x[:8]}...",
                )

            with col2:
                export_format = st.selectbox(
                    "Export Format",
                    options=["jsonl", "json", "csv", "zip"],
                    format_func=lambda x: x.upper(),
                )

            # Export button
            output_filename = st.text_input(
                "Output Filename",
                value=f"training_data.{export_format}",
            )

            output_dir = st.text_input(
                "Output Directory",
                value="/tmp",
            )

            if st.button("💾 Export Dataset", type="primary", use_container_width=True):

    export_dataset_async(
        export_dataset_id, output_filename, export_format, output_dir
    )

    file_path = os.path.join(output_dir, output_filename)

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            st.download_button(
                label="⬇ Download File",
                data=f,
                file_name=output_filename,
                mime="application/octet-stream",
            )
        else:
            st.info("No datasets to export. Process some documents first.")

    # Status section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Status")

    # Document count
    doc_count = len(st.session_state.bot.list_documents())
    st.sidebar.metric("Documents Loaded", doc_count)

    # Dataset count
    dataset_count = len(st.session_state.datasets)
    st.sidebar.metric("Datasets Created", dataset_count)

    # Job count
    job_count = len(st.session_state.jobs)
    st.sidebar.metric("Jobs", job_count)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style='text-align: center; color: gray;'>
            <small>
                Environment: {settings.APP_ENV}<br>
                Model: {settings.LLM_MODEL}
            </small>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
