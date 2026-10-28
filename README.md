# Intent recognition and slot filling: replication package

This repository contains the implementation, evaluation data, saved predictions, and metrics for a schema-driven NLU study on Field Operations, ATIS, SNIPS, and MultiWOZ. The pretrained NLI and QA models are used without target-domain weight updates. The Field Operations utterances are generated evaluation cases, not interview transcripts.

There are two practical ways to use the package:

1. **Recalculate reported metrics** from the saved predictions. This does not need a GPU, model downloads, or the Wikipedia index.
2. **Repeat inference** on the prepared datasets. This needs an NVIDIA GPU, model downloads, Java/PyTerrier, and the Wikipedia title index described below. It can take many hours and may differ slightly across hardware or model downloads.

Run commands from the repository root. The instructions below target Windows PowerShell and Python 3.12, matching the environment used for the saved run.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For the CUDA inference path, install the PyTorch build used in the saved environment and check the GPU. The general requirements file deliberately leaves `torch` commented out. A working NVIDIA driver and Java installation are also needed for the full pipeline.

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install accelerate==1.6.0 pytorch_revgrad==0.2.0
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
java -version
```

The local saved environment used Python 3.12.4, PyTorch 2.9.0+cu126, spaCy 3.8.5, Transformers 4.51.3, and PyTerrier 0.13.0. The CUDA check should print `True` before repeating inference. If you only recalculate metrics, CUDA is unnecessary. To run notebooks interactively, install JupyterLab in the virtual environment with `python -m pip install jupyterlab` (or use another Jupyter frontend with the same environment as its kernel).

## Repository map

| Path | Purpose |
|---|---|
| `src/core/` | NLI/QA models, retrieval, intent recognition, slot extraction, and assignment. |
| `src/data_tools/` | Dataset conversion, batch inference helpers, metrics, and dashboards. |
| `datasets/<dataset>/raw/` and `processed/` | Source and prepared JSON inputs. The batch evaluation notebook defines the exact final file selection. |
| `datasets/<dataset>/metadata/` | Intent/slot verbalizations and mapping of raw labels to semantic labels; shared aliases are in `datasets/metadata/semantic_alias_registry.json`. |
| `notebooks/evaluation/` | Batch inference, metrics, and stratified subsampling workflows. |
| `threshold_evaluation/` | Threshold sweep notebooks and saved calibration runs. |
| `evaluation/<dataset>/` | Prepared CSVs, predictions, metrics, and subsampling results. |
| `retrieval/` | Wikipedia title-index builder; the large input titles file and built index are not stored in Git. |

The evaluation datasets are `field_operations` (2,700 utterances), `atis` (5,232), `snips` (13,784), and `multiwoz` (12,797). These are the retained evaluation populations, not necessarily the sizes of their upstream corpora.

## Recalculate metrics from saved predictions

Start JupyterLab from the repository root:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```

Open `notebooks/evaluation/batch_eval_metrics.ipynb`, select the `.venv` kernel, and run its cells in order. It reads `evaluation/<dataset>/results_csv/<dataset>_predictions.csv` and the prepared JSON inputs, then writes `evaluation/<dataset>/metrics/`. To repeat the stratified analysis, run `notebooks/evaluation/batch_eval_subsampling.ipynb` in order. Its saved configuration uses 1,000 rounds and seed 42; outputs go under `evaluation/<dataset>/metrics_subsampling/stratified_by_intent/r1000_seed42/`.

The dashboards can be regenerated from the saved CSVs without running model inference:

```powershell
.\.venv\Scripts\python.exe src\data_tools\build_eval_dashboard.py
.\.venv\Scripts\python.exe src\data_tools\build_subsampling_dashboard.py
.\.venv\Scripts\python.exe src\data_tools\build_unified_eval_dashboard.py
```

Open the resulting HTML files in `evaluation/reports/`. The concise numerical tables are `evaluation/<dataset>/metrics/overall_summary.csv` and the corresponding `intent_by_label.csv`, `slot_conll_by_intent.csv`, and `slot_conll_by_label.csv`. `results_csv` has example-level predictions; `dataset_csv` has the normalized reference cases. The `*_semantic_json` columns contain JSON-encoded slot dictionaries inside CSV cells.

`overall_summary.csv` describes the complete retained population. Files under `metrics_subsampling/` describe repeated balanced samples; their means are a different estimand. Intent `UNK_INTENT` denotes abstention. Slot scores use the **reference intent frame**, so they measure slot extraction separately from intent prediction. The slot columns labeled `conll` use span-level BIO/CoNLL scoring.

## Repeat inference

The batch run expects `retrieval/2025_sparse_wikititle_index/data.properties`. The built index (about 12.9 GB) is excluded from Git. Its source title file (about 1.5 GB) is included through Git LFS. Rebuild the index as follows, after completing the setup above:

1. Install Git LFS, run `git lfs install`, then clone the repository or run `git lfs pull` after cloning. Confirm that `retrieval/enwiki-20250220-all-titles` is present (not an LFS pointer). It is UTF-8 text with a `page_namespace`/`page_title` tab-separated header. Its SHA-256 is `A8CE3DD644D3741B9431835702F180D3D979021B9507F4655FE5735AC84DDDCB` (check with `Get-FileHash -Algorithm SHA256 retrieval/enwiki-20250220-all-titles`).
2. From the repository root, start JupyterLab with the project environment:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install jupyterlab
   .\.venv\Scripts\python.exe -m jupyter lab
   ```

3. Open `retrieval/build_sparse_title_index.ipynb`, select the project environment as the kernel, and run its cells in order. The notebook reads the title file above and writes `retrieval/2025_sparse_wikititle_index/`. Indexing needs Java, substantial free disk space and memory, and may take a while.
4. Confirm that `retrieval/2025_sparse_wikititle_index/data.properties` exists before running inference. The notebook's final cell reports the index statistics.

The notebook and included title file are sufficient to rebuild a usable index. Index bytes may depend on the indexing environment, so byte-for-byte identity with the saved run is not guaranteed. If an identical index is supplied separately, place it at the expected path or set `WIKITITLE_INDEX_PATH` to its directory.

After checking the input files and index, run the batch wrapper from the repository root:

```powershell
.\.venv\Scripts\python.exe evaluation\_run_batch_eval_from_notebook.py
```

It executes the active cells of `notebooks/evaluation/batch_eval_table.ipynb` using CUDA device 0, `tau_intent=0.25`, `tau_slots=0.35`, and the reference intent for slot evaluation. It writes `evaluation/<dataset>/dataset_csv/` and `results_csv/`, **overwriting saved evidence**; run it in a separate copy of the repository if you need to preserve the published CSVs. The model loader downloads `MoritzLaurer/ModernBERT-large-zeroshot-v2.0`, `ahotrod/deberta-v3-large-finetuned-squadv2`, and the spaCy model on first use. Exact Hugging Face revisions were not captured for the saved run, so numerical identity is not guaranteed.

The notebooks in `threshold_evaluation/` document calibration. The final saved selections are in `runs/tau_intent_20260424_035819/artifacts/` and `runs/tau_slot_20260507_005006/artifacts/`. The global thresholds were selected using samples from the public evaluation pools; Field Operations was not used for threshold selection. Repeating the sweeps is not required to recalculate the published metrics.

## Regenerate the Field Operations dataset

The saved evaluation input is the nine JSON files in `datasets/field_operations/raw/`, with 300 examples per intent. The final generation configuration is `datasets/field_operations/metadata/prompts/intent_prompts_full_v2.json`; the earlier prompt files in that directory document prior drafts. To generate a new dataset, use a separate copy of the repository because the notebook writes directly to the saved raw files:

1. Complete the Python setup above and set `OPENAI_API_KEY` in the environment used by Jupyter. Generation calls the OpenAI API and incurs API charges.
2. Start JupyterLab from the repository root with `.\.venv\Scripts\python.exe -m jupyter lab`. Open `notebooks/dataset_generation/generate_field_operations_dataset.ipynb` with the `.venv` kernel. In its configuration cell, change `INTENTS_TO_GENERATE` from the saved single-intent selection to `"ALL"`, then run the cells in order through the `RUN: generate datasets` cell. The final duplicate-report cell is optional.
3. The notebook uses `gpt-4.1` and the final prompt configuration to request examples in batches by required/forbidden slot combination. It checks utterance length, short entity spans, required slots, and exact duplicate utterances, then saves one `<intent>_300.json` file per intent in `datasets/field_operations/raw/`. Check that all nine files contain 300 examples each before using them for evaluation.

One saved-data normalization is not performed by the generator: for `control_location_sharing_300.json`, rename the top-level JSON key from `Control Location Sharing` to `Location Sharing` to match the final dataset and evaluation schema. The generated utterances may differ from the published ones because API generation is nondeterministic; the saved JSON files remain the inputs for reproducing the reported results.

## Data and study-material limits

The prepared JSONs and semantic registry are sufficient to rerun the quantitative pipeline. The repository does not fully reconstruct the upstream conversion of MultiWOZ dialogue turns or every SNIPS source split from original releases. The interview transcripts and coding workbooks require separate disclosure review and are not part of the public package by default. Accordingly, this public package supports quantitative reproduction and inspection of the published schema, while the full qualitative coding process requires author-provided materials.

The OpenAI-based generation scripts are retained as provenance; running them is **not** part of repeating inference or metrics. They require `OPENAI_API_KEY` in the process environment, incur API use, and may produce different utterances. Do not commit API keys or a local `.env` file.
