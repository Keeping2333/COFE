## Repository Contents

- `rca/config/`: dataset configuration files and label mappings.
- `rca/src/fullstack_rca/data/`: dataset loaders and trace-centered multimodal alignment.
- `rca/src/fullstack_rca/models/`: log, metric, trace, fusion, loss, and classifier code.
- `rca/src/fullstack_rca/rag/`: FAISS-backed retrieval and knowledge-base logic.
- `rca/src/fullstack_rca/agent/`: OpenAI-compatible LLM diagnosis agent.
- `rca/src/fullstack_rca/detect/`: baseline-relative telemetry evidence detection.
- `rca/scripts/`: training, indexing, calibration, diagnosis, and baseline entry points.
- `rca/eval/`: retrieval, grounding, localization, OOD, and unified scoring scripts.
- `train-ticket/`: Train-Ticket workload, anomaly, and telemetry collection scripts.
- `social-network/`: SocialNetwork workload, anomaly, and batch collection scripts.
- `monitoring/`: Docker/Kubernetes observability stack configuration.

Generated directories such as `checkpoints/`, `knowledge_base/`, and `results/`
are not required to exist before running the pipeline. The scripts create or
write them as needed.

## Prerequisites

- Python 3.10 is recommended.
- A package manager such as `conda` or `venv` + `pip`.
- PyTorch compatible with the local CPU/GPU environment.
- PyTorch Geometric dependencies used by the trace encoder.
- FAISS CPU, Transformers, NumPy, pandas, scikit-learn, PyYAML, tqdm, SciPy,
  and the other packages listed in `rca/requirements.txt`.
- An OpenAI-compatible LLM endpoint for `rca/scripts/diagnose.py`.
- Labeled experiment data in the expected schema. Raw experiment runs are not
  bundled in this cleaned package.

For GPU use, install the PyTorch and PyTorch Geometric wheels matching the
local CUDA runtime before running the full training workflow.

## Installation

Run these commands from the artifact root:

```bash
conda create -n rca-artifact python=3.10
conda activate rca-artifact
pip install -r rca/requirements.txt
export PYTHONPATH="$PWD/rca/src:${PYTHONPATH}"
```

A lightweight import/configuration check that does not require experiment data:

```bash
python -c "from fullstack_rca.data.factory import load_config; print(load_config('rca/config/dataset.yaml')['dataset']['path'])"
```

All commands below assume they are run from the artifact root with
`PYTHONPATH` set as shown above.

## Data Preparation

The model scripts read dataset locations from YAML files under `rca/config/`.
The included configs are:

- `rca/config/dataset.yaml`: Train-Ticket in-domain configuration.
- `rca/config/dataset_sn.yaml`: SocialNetwork in-domain configuration.
- `rca/config/dataset_tt.yaml`: optional external query-data configuration.
- `rca/config/label_mapping.yaml`: label mapping used when query labels differ
  from the canonical fault taxonomy.

If your dataset is stored outside the artifact tree, edit only the `dataset.path`
field in the relevant YAML file or pass a separate config file with `--config`
or `--query-config`.

Each experiment directory is expected to contain the schema consumed by
`ExpRunsDataset`:

```text
<experiment>/
  meta.json
  traces/
  logs/logs.jsonl
  metrics/
```

`meta.json` must provide the fault layer, fault type, target service, and
injection timestamps used by the training, retrieval, and evidence-detection
stages.

## Full Reproduction Workflow

The full workflow trains the encoders, trains the layer classifier, builds the
retrieval knowledge base, calibrates the retrieval-trust threshold, and runs
diagnosis. Use the Train-Ticket config unless you explicitly want the
SocialNetwork workflow.

### Train-Ticket Workflow

```bash
python rca/scripts/train.py --config rca/config/dataset.yaml --epochs 20

python rca/scripts/train_layer_classifier.py \
  --config rca/config/dataset.yaml \
  --epochs 200 \
  --lr 1e-3 \
  --seed 42

python rca/scripts/build_kb.py --config rca/config/dataset.yaml

python rca/scripts/compute_kb_loo_threshold.py \
  --kb-dir knowledge_base \
  --ckpt-dir checkpoints \
  --percentile 1.0

python rca/scripts/diagnose.py \
  --config rca/config/dataset.yaml \
  --num-samples -1 \
  --model <model-name> \
  --base-url <openai-compatible-base-url> \
  --api-key <api-key-or-local-placeholder>
```

The `--base-url` argument may be omitted if the LLM provider is configured
through the environment. `--api-key` may also be omitted if `OPENAI_API_KEY` is
set. Local OpenAI-compatible endpoints still need a non-empty key value if the
client library requires one.

### SocialNetwork Workflow

```bash
python rca/scripts/train.py --config rca/config/dataset_sn.yaml --epochs 20

python rca/scripts/train_layer_classifier.py \
  --config rca/config/dataset_sn.yaml \
  --epochs 200 \
  --lr 1e-3 \
  --seed 42

python rca/scripts/build_kb.py --config rca/config/dataset_sn.yaml

python rca/scripts/compute_kb_loo_threshold.py \
  --kb-dir knowledge_base_sn \
  --ckpt-dir checkpoints_sn \
  --percentile 1.0

python rca/scripts/diagnose.py \
  --config rca/config/dataset_sn.yaml \
  --num-samples -1 \
  --model <model-name> \
  --base-url <openai-compatible-base-url> \
  --api-key <api-key-or-local-placeholder>
```

## Lightweight Example Workflow

After the relevant checkpoint and knowledge-base directories have been created,
run retrieval-only evaluation without invoking an LLM:

```bash
python rca/eval/eval_retrieval.py \
  --config rca/config/dataset.yaml \
  --num-samples 0
```

Run a one-experiment diagnosis smoke test:

```bash
python rca/scripts/diagnose.py \
  --config rca/config/dataset.yaml \
  --num-samples 1 \
  --model <model-name> \
  --base-url <openai-compatible-base-url> \
  --api-key <api-key-or-local-placeholder>
```

For SocialNetwork, replace `rca/config/dataset.yaml` with
`rca/config/dataset_sn.yaml`.


## Expected Outputs

For the default Train-Ticket config:

- `checkpoints/best_model.pt`
- `checkpoints/resume_state.pt`
- `checkpoints/layer_classifier.pt`
- `checkpoints/retrieval_tau.json`
- `knowledge_base/index.faiss`
- `knowledge_base/metadata.json`
- `knowledge_base/episodes.json`
- `knowledge_base/trace_vectors.npy`
- `knowledge_base/log_vectors.npy`
- `results/fullstack_rca_<mode>_<N>samples.json`
- `results/fullstack_rca_<mode>_<N>samples_report.md`

For `rca/config/dataset_sn.yaml`, the corresponding generated directories are
`checkpoints_sn/` and `knowledge_base_sn/`.

## Evaluation

The diagnosis script writes aggregate metrics into its JSON output. The reported
metrics include layer accuracy, fault-type accuracy, and macro-F1.

Retrieval-only metrics:

```bash
python rca/eval/eval_retrieval.py \
  --config rca/config/dataset.yaml \
  --num-samples 0
```

Grounding and confound-handling checks for a diagnosis output:

```bash
python rca/eval/eval_grounding.py \
  results/<diagnosis-output>.json \
  --output results/<diagnosis-output>_grounding.json
```

Unified scoring for a result JSON:

```bash
python rca/eval/score_unified.py \
  score \
  results/<diagnosis-output>.json \
  --output results/<diagnosis-output>_metrics.json
```

Localization evaluation, when the result JSON contains the required fields:

```bash
python rca/eval/eval_localization.py \
  --result results/<diagnosis-output>.json \
  --output results/localization
```

## Baseline Scripts

The package includes baseline wrapper scripts under:

- `rca/scripts/baseline_*.py`
- `rca/scripts/famos/`
- `rca/scripts/medicine/`

These are optional comparison entry points. Some wrappers expect external
baseline dependencies or feature caches that are not bundled in this cleaned
package. Use them only after installing the corresponding external dependencies
and preparing the required input data.

## Dataset Collection And Anomaly Injection Assets

If you already have experiment runs in the expected schema, the collection
assets are not needed for model training or evaluation.

Included collection and anomaly assets:

- `train-ticket/run_experiment.py`: single Train-Ticket experiment runner. It
  requires a user-provided JSON or YAML experiment config.
- `train-ticket/batch_experiment.py`: Train-Ticket batch runner. Its default
  config name is `exp.json`; provide `--config <experiment-config>` if the file
  is stored elsewhere.
- `train-ticket/Anomalies/`: Train-Ticket anomaly manifests.
- `train-ticket/locust/`: Train-Ticket load-generation scripts.
- `social-network/batch_experiment.py`: SocialNetwork batch runner. Its default
  config name is `exp.json`; provide `--config <experiment-config>` if needed.
- `social-network/Anomalies/`: SocialNetwork anomaly injection scripts.
- `social-network/locust/`: SocialNetwork load-generation scripts.
- `monitoring/`: observability stack configuration used to collect metrics,
  traces, and logs.

These scripts must be adapted to the local deployment environment using
placeholder paths, hostnames, credentials, and service endpoints supplied by the
person running the artifact.



