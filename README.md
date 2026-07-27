<![CDATA[<div align="center">

# MetaAutoML — ML-Builder

### A Unified, Memory-Augmented AutoML & AutoDL Framework  
### for Tabular and Multi-Modal Data

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Optuna](https://img.shields.io/badge/HPO-Optuna-blue)](https://optuna.org/)
[![FAISS](https://img.shields.io/badge/Memory-FAISS-orange)](https://github.com/facebookresearch/faiss)
[![W&B](https://img.shields.io/badge/Tracking-Weights%20%26%20Biases-yellow?logo=weightsandbiases)](https://wandb.ai/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-red)](https://shap.readthedocs.io/)

</div>

---

## 📌 Project Overview

**MetaAutoML (ML-Builder)** is a research-grade, end-to-end Automated Machine Learning framework that goes beyond conventional AutoML by integrating **meta-learning**, **multi-modal data support**, and **LLM-augmented decision-making** into a single cohesive pipeline.

### Why MetaAutoML?

Traditional AutoML tools treat every dataset as a blank slate — re-running the entire model search from scratch each time. MetaAutoML solves this by maintaining a **persistent FAISS-backed memory store** of past experiments. When a new dataset arrives, the system computes its statistical fingerprint, retrieves similar past datasets, and warm-starts hyperparameter optimisation — dramatically reducing search time while preserving (or improving) accuracy.

### Key Differentiators

| Feature | MetaAutoML | Conventional AutoML |
|---|---|---|
| **Memory-Augmented Search** | ✅ FAISS knowledge base warm-starts HPO | ❌ Cold-start every time |
| **Paradigm Routing** | ✅ R(D) router decides ML vs. DL | ❌ Single paradigm |
| **Multi-Modal Support** | ✅ Vision, Audio, Text, Video, Tabular | ❌ Tabular only |
| **LLM-Guided Decisions** | ✅ Model shortlisting + explainability | ❌ None |
| **Resource-Aware Adaptation** | ✅ Dynamic pipeline caps per dataset | ❌ Fixed configurations |
| **Ensemble of Experts** | ✅ Calibrated soft-voting ensemble | ❌ Single best model |
| **Explainability Reports** | ✅ SHAP + LLM-generated narratives | ❌ Basic metrics only |

### Target Users

- **ML Researchers** exploring meta-learning and transfer learning for AutoML.
- **Data Scientists** who need a robust, explainable pipeline for both tabular and multi-modal tasks.
- **Students & Academics** studying automated machine learning architectures.

---

## 🏗️ Architecture

MetaAutoML implements a **seven-phase pipeline** with an adaptive paradigm router that decides between Classical ML and Deep Learning paths at runtime.

```
                           ┌──────────────────────────────────────────┐
                           │          1. DATA INGESTION               │
                           │   OnboardingAgent → Modality Detection   │
                           └─────────────────┬────────────────────────┘
                                             │
                        ┌────────────────────┼──────────────────────┐
                        ▼                                           ▼
              ┌─────────────────┐                      ┌────────────────────┐
              │  TABULAR (CSV)  │                      │  MULTI-MODAL       │
              │  data_loader    │                      │  (Vision/Audio/    │
              │  data_cleaner   │                      │   Text/Video)      │
              └────────┬────────┘                      │  UniversalEmbedder │
                       │                               │  + LoRA Adapters   │
                       ▼                               └─────────┬──────────┘
              ┌─────────────────┐                                │
              │ 2. RESOURCE     │                                │
              │    ANALYSIS     │                                │
              │ ResourceManager │                                │
              └────────┬────────┘                                │
                       │                                         │
                       ▼                                         │
              ┌─────────────────┐                                │
              │ 3. FEATURE      │                                │
              │    ENGINEERING  │                                │
              │ FeatureEngineer │                                │
              └────────┬────────┘                                │
                       │                                         │
                       ▼                                         ▼
              ┌──────────────────────────────────────────────────────┐
              │            4. META-LEARNING CORE                     │
              │  ┌──────────────┐  ┌───────────────┐  ┌───────────┐ │
              │  │ Dataset      │  │ Siamese Task  │  │ FAISS     │ │
              │  │ Embedding    │→ │ Encoder (MLP) │→ │ Memory    │ │
              │  │ (10D → 32D)  │  │ Contrastive   │  │ Store     │ │
              │  └──────────────┘  └───────────────┘  └─────┬─────┘ │
              └──────────────────────────────────────────────┼───────┘
                                                             │
                       ┌─────────────────────────────────────┘
                       ▼
              ┌─────────────────────┐
              │ 5. PARADIGM ROUTER  │
              │ R(D) = λ₁·LLM(D)   │
              │      + λ₂·Memory(D) │
              │      + λ₃·Heuristic │
              └──────┬──────────┬───┘
                     │          │
            R(D)≤τ   │          │  R(D)>τ
                     ▼          ▼
            ┌──────────┐  ┌──────────────┐
            │ AutoML   │  │   AutoDL     │
            │ Pipeline │  │   Pipeline   │
            │ (HPO +   │  │ (Hybrid ML   │
            │  SHAP)   │  │  on Embeds)  │
            └────┬─────┘  └──────┬───────┘
                 │               │
                 ▼               ▼
              ┌──────────────────────────┐
              │ 6. EVALUATION & REPORTS  │
              │ • SHAP Explanations      │
              │ • LLM Consultant Report  │
              │ • Jupyter Notebook       │
              │ • W&B Experiment Tracking│
              └──────────────────────────┘
```

### Paradigm Router — R(D) Decision Function

The core innovation is the **R(D) paradigm router**, which fuses three signals to decide whether a dataset is better served by Classical ML or Deep Learning:

```
R(D) = λ₁ · LLM(D) + λ₂ · Memory(D) + λ₃ · Heuristics(D)
```

| Signal | Source | Weight (Default) |
|---|---|---|
| `LLM(D)` | GPT-4o-mini probability estimate | λ₁ = 0.5 |
| `Memory(D)` | FAISS retrieval of similar past datasets | λ₂ = 0.2 |
| `Heuristics(D)` | Rule-based complexity scoring | λ₃ = 0.3 |

If `R(D) > τ` (default τ = 0.5), the system routes to AutoDL; otherwise, it uses classical AutoML.

---

## 📂 Repository Structure

```
ML-Builder/
│
├── main.py                     # CLI entry point (legacy pipeline orchestrator)
├── phase4_pipeline.py          # Primary pipeline — meta-learning + paradigm routing
├── config.py                   # Environment & API key configuration
├── requirements.txt            # Core Python dependencies
│
├── ─── Data Ingestion ───────────────────────────────
│   ├── data_loader.py          # CSV/Excel loading, problem type detection, leakage checks
│   ├── data_cleaner.py         # Missing value imputation, duplicate removal
│   ├── onboarding_agent.py     # LLM-powered modality detection & business context intake
│   └── dataset_profiler.py     # Statistical profiling for LLM context
│
├── ─── Feature Engineering ──────────────────────────
│   ├── feature_engineering.py  # Advanced FE: NLP extraction, Yeo-Johnson, interactions
│   ├── feature_processing.py   # Sklearn preprocessing pipeline (scaling, encoding)
│   └── resource_manager.py     # Dynamic pipeline caps based on dataset size
│
├── ─── Meta-Learning Core ───────────────────────────
│   ├── dataset_embedding.py    # 10D statistical fingerprint computation
│   ├── task_encoder.py         # Siamese MLP encoder (10D → 32D learned embeddings)
│   ├── cold_start.py           # Adaptive cold-start strategy with FAISS retrieval
│   ├── unified_memory.py       # Unified FAISS memory for ML + DL configurations
│   └── build_memory.py         # OpenML dataset ingestion for pre-seeding memory
│
├── ─── Model Training & Selection ───────────────────
│   ├── model_trainer.py        # Model catalogue (14+ algorithms), baseline screening
│   ├── model_selector.py       # Metric calculation, Grid/Random hyperparameter search
│   ├── hpo_optuna.py           # Multi-objective Optuna HPO with warm-starting
│   └── multi_objective.py      # Utility scoring: accuracy × speed × complexity
│
├── ─── Paradigm Routing ─────────────────────────────
│   ├── paradigm_router.py      # R(D) fusion function (LLM + Memory + Heuristics)
│   ├── heuristics.py           # Rule-based model suggestions per dataset properties
│   ├── llm_suggester.py        # LLM-powered model shortlisting with name validation
│   └── routing_engine.py       # Additional routing logic
│
├── ─── Multi-Modal Support ──────────────────────────
│   ├── multimodal_extractor.py # UniversalEmbedder: CLIP, AST, SentenceTransformer
│   ├── domain_registry.py      # Pre-trained model registry (general, biology, remote sensing)
│   ├── lora_config.py          # LoRA hyperparameter registry per modality
│   ├── lora_adapter_trainer.py # PEFT LoRA fine-tuning for domain adaptation
│   └── dl_faiss_memory.py      # Modality-specific FAISS memory for DL results
│
├── ─── Explainability & Reporting ───────────────────
│   ├── explainer.py            # Permutation importance + SHAP orchestration
│   ├── shap_explainer.py       # SHAP value computation & plot generation
│   ├── llm_explainer.py        # LLM-generated consultant reports (Markdown)
│   ├── report_generator.py     # Standalone HTML report with embedded visuals
│   ├── notebook_generator.py   # Jupyter notebook auto-generation
│   ├── eda.py                  # Exploratory data analysis plots
│   └── confidence_calibration.py # ECE reliability diagrams
│
├── ─── Experiment Tracking ──────────────────────────
│   ├── wandb_logger.py         # Weights & Biases integration wrapper
│   └── wandb/                  # W&B run artifacts
│
├── ─── Agentic Pipeline ─────────────────────────────
│   └── agents/
│       ├── agent_orchestrator.py  # Multi-agent pipeline coordinator
│       ├── data_agent.py          # LLM-driven dataset understanding
│       ├── business_agent.py      # Business context → ML objectives translation
│       ├── feature_agent.py       # LLM-recommended feature engineering plans
│       ├── model_agent.py         # Memory-augmented model recommendations
│       ├── critic_agent.py        # Adversarial validation of pipeline decisions
│       └── notebook_generator.py  # Agent-specific EDA notebook creation
│
├── ─── Benchmark Scripts ────────────────────────────
│   ├── run_metaautoml(ml).py   # Tabular ML benchmark runner with metrics
│   ├── run_metaautoml(dl).py   # Multi-modal DL benchmark runner with metrics
│   └── run_agentic_pipeline.py # Agentic pipeline execution script
│
├── ─── Tests ────────────────────────────────────────
│   └── tests/
│       ├── test_oof_leakage.py    # Out-of-fold data leakage validation
│       └── test_ensemble_gpu.py   # GPU ensemble placement tests
│
├── ─── Persistence ──────────────────────────────────
│   ├── memory_store.faiss      # FAISS index (serialised)
│   ├── memory_store.pkl        # Metadata for FAISS records
│   ├── task_encoder.pt         # Trained Siamese encoder weights
│   └── embedding_cache/        # Cached multi-modal embeddings
│
└── ─── Documentation ────────────────────────────────
    ├── README.md               # This file
    ├── codebase_analysis.md    # Deep-dive file-level analysis
    └── walkthrough.md          # System walkthrough guide
```

---

## 🔧 Technology Stack

### Core ML & Data

| Library | Purpose | Version |
|---|---|---|
| `pandas` | Tabular data manipulation | ≥ 1.5.0 |
| `numpy` | Numerical computation | ≥ 1.23.0 |
| `scikit-learn` | Classical ML models, preprocessing, metrics | ≥ 1.2.0 |
| `scipy` | Statistical functions (skew, entropy, Yeo-Johnson) | ≥ 1.9.0 |
| `xgboost` | Gradient boosting (GPU-accelerated) | — |
| `lightgbm` | Gradient boosting (GPU-accelerated) | — |

### Deep Learning & Multi-Modal

| Library | Purpose |
|---|---|
| `torch` | PyTorch backend for Siamese encoder, GPU acceleration |
| `transformers` | Hugging Face models (CLIP, AST, BEiT, SigLIP) |
| `sentence-transformers` | Text embedding via MiniLM |
| `peft` | Parameter-Efficient Fine-Tuning (LoRA adapters) |
| `open_clip` | BioCLIP and domain-specific vision models |
| `librosa` | Audio feature extraction (MFCC) |
| `opencv-python` | Video frame extraction |

### Meta-Learning & Memory

| Library | Purpose |
|---|---|
| `faiss-cpu` | Approximate nearest neighbor search for memory retrieval |
| `optuna` | Multi-objective hyperparameter optimisation |

### LLM Integration

| Library | Purpose |
|---|---|
| `litellm` | Unified LLM API (OpenAI, Anthropic, DeepSeek, etc.) |
| `python-dotenv` | Environment variable management for API keys |

### Experiment Tracking & Explainability

| Library | Purpose |
|---|---|
| `wandb` | Weights & Biases experiment tracking |
| `shap` | SHAP value computation and visualisation |
| `matplotlib` / `seaborn` | Plotting and visualisation |

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.10+
- **CUDA** (optional, for GPU-accelerated training)
- **API Keys** (optional, for LLM features)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ML-Builder.git
cd ML-Builder

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies for full functionality
pip install torch xgboost lightgbm optuna litellm wandb transformers sentence-transformers peft
```

### Environment Configuration

Create a `.env` file in the project root:

```env
# LLM Configuration (optional — disable with USE_LLM=False in config.py)
OPENROUTER_API_KEY=your_openrouter_key_here
LLM_MODEL=openrouter/openai/gpt-4o-mini

# Weights & Biases (optional)
WANDB_API_KEY=your_wandb_key_here

# Hugging Face (optional — for domain-specific models)
HF_TOKEN=your_huggingface_token_here
```

### Quick Start

#### Tabular Dataset (Interactive Mode)

```bash
python phase4_pipeline.py
# Follow prompts: Enter CSV path → Business context → Target column
```

#### Tabular Dataset (Script Mode)

```bash
python run_metaautoml(ml).py
# Edit the script to set your dataset_path and target_column
```

#### Multi-Modal Dataset (Vision)

```bash
python run_metaautoml(dl).py
# Automatically extracts embeddings, trains LoRA adapter, runs HPO
```

#### Agentic Pipeline (LLM-Guided)

```bash
python run_agentic_pipeline.py
# Multi-agent system: DataAgent → BusinessAgent → CriticAgent → ModelAgent
```

### Offline Mode

To run without LLM or W&B connectivity:

```python
# In config.py
USE_LLM = False
USE_WANDB = False
```

The system gracefully degrades — using rule-based heuristics and FAISS memory instead of LLM suggestions.

---

## 📊 Pipeline Walkthrough

### Phase 1 — Data Ingestion & Cleaning
The `OnboardingAgent` detects modality (tabular/vision/audio/text/video) from file extensions. For tabular data, `data_loader.py` handles CSV/Excel loading with automatic problem type detection (classification vs. regression) and target leakage detection via correlation analysis and DecisionTree probing.

### Phase 2 — Resource Analysis
The `ResourceManager` inspects dataset dimensions and cardinality to set dynamic caps on feature engineering complexity, preventing OOM errors on large or high-cardinality datasets.

### Phase 3 — Feature Engineering
The `FeatureEngineer` applies an adaptive suite of transformations:
- **Text Stats**: Word count, character count, unique word ratio from text columns
- **Skew Correction**: Yeo-Johnson transforms for highly skewed numeric features
- **Adaptive Scaling**: Per-feature selection of StandardScaler vs. RobustScaler
- **Interaction Features**: Pairwise multiplication of top correlated features
- **Multicollinearity Pruning**: Drops features above a configurable correlation threshold

### Phase 4 — Meta-Learning & Memory Retrieval
A 10-dimensional statistical fingerprint is computed for each dataset, then projected into a 32D learned embedding space via a Siamese MLP trained with contrastive loss. FAISS retrieves the most similar past datasets and their best-performing model configurations for warm-starting.

### Phase 5 — Training & Hyperparameter Optimisation
- **AutoML Path**: Optuna HPO across LLM-suggested + memory-retrieved model shortlist, with multi-objective scoring (accuracy × speed × complexity).
- **AutoDL Path**: Hybrid ML-on-Embeddings using XGBoost/LightGBM/HistGBM on PCA-reduced embeddings, with an Ensemble of Experts (top-3 calibrated models, soft-voting).

### Phase 6 — Evaluation & Reporting
- **SHAP Explanations**: TreeExplainer for tree-based models, KernelExplainer for others
- **LLM Consultant Report**: 4-pillar analytics narrative (Descriptive → Diagnostic → Predictive → Prescriptive)
- **Jupyter Notebook**: Auto-generated with EDA, confusion matrices, and t-SNE visualisations
- **W&B Logging**: Full experiment tracking with artifacts

---

## 📐 System Metrics

MetaAutoML tracks several novel system-level metrics:

| Metric | Formula | Purpose |
|---|---|---|
| **C(D)** — Confidence Score | R(D) routing score | Confidence in paradigm decision |
| **ECE** — Expected Calibration Error | Reliability diagram bins | Model probability calibration |
| **SCR** — Search Compression Ratio | Total models / Models trained | Efficiency of memory-guided search |
| **PR** — Performance Retention | Final accuracy / Baseline accuracy | Quality relative to naive baseline |
| **TUS** — Transfer Utility Score | SCR × PR | Combined transfer learning benefit |

---

## 🧪 Testing

```bash
# Run leakage detection tests
python -m pytest tests/test_oof_leakage.py -v

# Run GPU ensemble tests
python -m pytest tests/test_ensemble_gpu.py -v

# Run cold-start unit tests
python test_cold_start.py

# Run embedding pipeline tests
python test_embedding.py
```

---

## ⚙️ Configuration Reference

### `config.py`

| Variable | Default | Description |
|---|---|---|
| `USE_LLM` | `True` | Enable/disable LLM-powered features |
| `USE_WANDB` | `True` | Enable/disable Weights & Biases logging |
| `LLM_MODEL` | `openrouter/openai/gpt-4o-mini` | Default LLM model identifier |
| `WANDB_PROJECT` | `metaautoml-v1` | W&B project name |
| `WANDB_ENTITY` | `None` | W&B team/entity name |

### `ColdStartConfig` (cold_start.py)

| Parameter | Default | Description |
|---|---|---|
| `k_neighbors` | 10 | FAISS retrieval count |
| `lambda_sensitivity` | 0.5 | Threshold sensitivity for ε(D) |
| `alpha` | 0.6 | Weight for similarity in combined score |
| `beta` | 0.3 | Weight for past performance |
| `gamma` | 0.1 | Weight for recency |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📖 Citation

If you use MetaAutoML in academic research, please cite:

```bibtex
@software{metaautoml2026,
  title     = {MetaAutoML: A Unified, Memory-Augmented AutoML Framework},
  author    = {Dinesh},
  year      = {2026},
  url       = {https://github.com/your-username/ML-Builder}
}
```

---

<div align="center">
  <strong>Built with 🔬 for the ML research community.</strong>
</div>
]]>
