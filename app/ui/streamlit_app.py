"""Streamlit UI for the Training Data Curation Bot."""

import sys
import os
import asyncio
import tempfile
import shutil
import streamlit as st

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from training_data_bot import TrainingDataBot
from training_data_bot.core.config import settings
from training_data_bot.tasks.task_manager import TaskManager


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Training Data Curation Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Main background */
.main {
    background-color: #0e1117;
}

/* Title */
h1 {
    font-weight: 700 !important;
}

/* Cards */
.card {
    padding: 20px;
    border-radius: 12px;
    background-color: #161b22;
    border: 1px solid #30363d;
    margin-bottom: 20px;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    height: 45px;
    font-weight: 600;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-size: 16px;
    font-weight: 600;
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-size: 28px;
}

/* File uploader */
section[data-testid="stFileUploader"] {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- SESSION STATE ----------------
def init_session_state():
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


# ---------------- PROCESS DOCUMENTS ----------------
def process_documents_async(uploaded_files, url_input, selected_tasks, quality_filter, chunk_size, chunk_overlap):

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

    shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------- EXPORT DATASET ----------------
def export_dataset_async(export_dataset_id, output_filename, export_format, output_dir):

    output_path = os.path.join(output_dir, output_filename)

    dataset = st.session_state.datasets.get(export_dataset_id)

    if not dataset:
        st.error(f"Dataset not found: {export_dataset_id}")
        return None

    examples = dataset.get("examples", [])

    if not examples:
        st.error("No examples in dataset. Process documents first.")
        return None

    st.info(f"Exporting {len(examples)} examples...")

    result = st.session_state.bot.exporter.export(examples, output_path, export_format)

    output_file_path = result.get("output_path")

    if output_file_path and os.path.exists(output_file_path):
        st.success(f"Exported {result.get('example_count')} examples to: {output_file_path}")
        return output_file_path

    st.error("Export failed: file not found")
    return None


# ---------------- MAIN APP ----------------
def main():

    init_session_state()

    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")

    task_manager = TaskManager()
    available_tasks = task_manager.get_supported_tasks()

    selected_tasks = st.sidebar.multiselect(
        "Select Task Types",
        available_tasks,
        default=["qa_generation"],
        key="task_select"
    )

    st.sidebar.markdown("### Processing Options")

    quality_filter = st.sidebar.checkbox(
        "Enable Quality Filter",
        value=True,
        key="quality_filter"
    )

    chunk_size = st.sidebar.slider(
        "Chunk Size",
        500,
        2000,
        settings.CHUNK_SIZE,
        step=100,
        key="chunk_size"
    )

    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap",
        0,
        500,
        settings.CHUNK_OVERLAP,
        step=50,
        key="chunk_overlap"
    )

    # Header
    st.title("📚 Training Data Curation Bot")
    st.caption("Generate high-quality AI training datasets from documents")

    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Results", "💾 Export"])

    # ---------------- TAB 1 ----------------
    with tab1:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            uploaded_files = st.file_uploader(
                "Upload Files",
                type=["txt","md","pdf","docx","json","xml","csv"],
                accept_multiple_files=True,
                key="file_upload"
            )

        with col2:
            url_input = st.text_area(
                "Enter URLs (one per line)",
                key="url_input"
            )

        st.divider()

        if st.button("🚀 Process Documents", use_container_width=True, key="process_button"):

            st.session_state.processing = True

            process_documents_async(
                uploaded_files,
                url_input,
                selected_tasks,
                quality_filter,
                chunk_size,
                chunk_overlap,
            )

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- TAB 2 ----------------
    with tab2:

        st.header("Results")

        datasets = list(st.session_state.datasets.values())

        if datasets:

            selected_dataset = st.selectbox(
                "Select Dataset",
                options=[d["id"] for d in datasets],
                key="results_dataset_selectbox"
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
                    key="export_dataset_selectbox"
                )

            with col2:

                export_format = st.selectbox(
                    "Export Format",
                    ["jsonl","json","csv","zip"],
                    key="export_format_selectbox"
                )

            output_filename = st.text_input(
                "Output Filename",
                value=f"training_data.{export_format}",
                key="export_filename"
            )

            output_dir = st.text_input(
                "Output Directory",
                value="/tmp",
                key="export_directory"
            )

            if st.button("💾 Export Dataset", use_container_width=True, key="export_button"):

                exported_file = export_dataset_async(
                    export_dataset_id,
                    output_filename,
                    export_format,
                    output_dir,
                )

                if exported_file:

                    with open(exported_file, "rb") as f:

                        st.download_button(
                            "⬇ Download File",
                            data=f,
                            file_name=os.path.basename(exported_file),
                            mime="application/octet-stream",
                            key="download_button"
                        )

        else:
            st.info("No datasets to export.")

    # Sidebar status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Status")

    st.sidebar.metric("Documents Loaded", len(st.session_state.bot.list_documents()))
    st.sidebar.metric("Datasets Created", len(st.session_state.datasets))
    st.sidebar.metric("Jobs", len(st.session_state.jobs))

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        f"""
        <div style='text-align:center;color:gray;'>
        <small>
        Environment: {settings.APP_ENV}<br>
        Model: {settings.LLM_MODEL}
        </small>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    main()
