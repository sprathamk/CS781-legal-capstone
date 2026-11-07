# CS781 Capstone: In-House Legal Intelligence System

**Pratham Kumar Singh · 261110046 · IIT Kanpur**

Small language models pretrained **from scratch** on the official Indian legal corpus (Phase 1), then adapted to six legal NLP tasks (Phases 2–7). No external pretrained weights or corpora are used. Data: [`Exploration-Lab/CS-781-Capstone`](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone).

## Results (best Codabench test score)

| Phase | Task | Metric | Score | Best system |
|---|---|---|---|---|
| 1 | Legal LM pretraining | BERTScore-F1 | 0.8325 | Model D + MBR decoding |
| 2 | Named entity recognition | strict macro-F1 | 0.5132 | D + LoRA + BiLSTM-CRF |
| 3 | Rhetorical role prediction | macro-F1 | 0.5356 | D + LoRMA + BiLSTM, class weights |
| 4 | Court judgment prediction | macro-F1 | 0.6782 | B + adapters + hierarchical BiLSTM |
| 5 | Statute identification | macro-F1 | 0.3706 | 4-layer student distilled from B |
| 6 | Summarization | BERTScore-F1 | 0.8251 | B + Houlsby adapters |
| 7 | Prior case retrieval | Micro-F1@7 | 0.3963 | B bi-encoder + BM25 + citation/date re-ranking |

## Architecture (Phase 1)

| Model | Design | Params | Tokens | Val. loss |
|---|---|---|---|---|
| A | GPT decoder: learned positions, pre-LayerNorm, GELU | 64.6M | 220M | 1.78 |
| B | Decoder with GQA (12 Q / 4 KV heads), RoPE, RMSNorm, SwiGLU | 66.1M | 220M | 1.65 |
| C | Encoder–decoder (6 + 6 layers) | 105.5M | 220M | 3.09 |
| D | Same as A, more data, no dropout | 64.6M | 400M | **1.31** |

All: 8 layers, width 768, 2,048-token context, byte-level BPE tokenizer (8,192 tokens) trained on the provided corpus. Next-token loss, AdamW, LR 3e-4, one pass over the data.

## Methodology (Phases 2–7)

* Every task model starts from a Phase-1 checkpoint. Features are a learned weighted average of all layers.
* Long documents are cut into 512-token chunks; the useful part is read (end of judgment for P4, spread chunks for P5/P7).
* Fine-tuning methods compared: full, partial, LoRA, rsLoRA, LoRMA, IA³, adapters, BitFit, and knowledge distillation into a 4-layer student.
* Task heads: BiLSTM-CRF (P2, P3), hierarchical BiLSTM (P4), label attention (P5), document → summary generation (P6), contrastive bi-encoder fused with BM25 (P7).

## Repository

| Folder | Notebook(s) |
|---|---|
| `Phase 1/` | `cs781-phase1-a/b/c.ipynb` (models A, B, C); `cs781-phase1-d-optimisedA.ipynb` (model D); `cs781-phase1-d-More-optimised.ipynb` (D with improved decoding) |
| `Phase 2/` | `phase2-lner.ipynb` (4 submissions); `phase2-d-lorma.ipynb` (LoRMA variants) |
| `Phase 3/` | `phase3-rr.ipynb` |
| `Phase 4/` | `phase-4-cjpe.ipynb` |
| `Phase 5/` | `cs781-phase5-fix.ipynb` |
| `Phase 6/` | `cs781-phase6.ipynb` |
| `Phase 7/` | `CS781_Phase7_PCR_round3.ipynb` |

Each notebook trains all experiments of its phase and writes one submission zip per experiment. Model weights are not included.

## How to run

1. Open the notebook on Kaggle. Set **GPU T4 x2** and turn **Internet on**.
2. Add your Hugging Face token as a Kaggle secret named **`HF`**.
3. Phases 2–7: **Add Input** → the output folders of the four Phase-1 runs.
4. **Run All.** Submission zips appear under `/kaggle/working/`.

Notebooks save checkpoints and skip experiments that are already finished, so a run can be resumed by attaching its previous output. To run only some experiments, edit `RUN_EXPERIMENTS` in the configuration cell.
# CS781-legal-capstone
# CS781-legal-capstone
