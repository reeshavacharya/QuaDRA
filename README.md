# QuaDRA

This repository contains research and proof-of-concept experiments related to Retrieval-Augmented Generation (RAG) vulnerabilities, specifically focusing on quantization-aware and post-training quantization backdoors.

## Structure
- `pq_overlap_poc/`: Proof-of-concept for RAG retrieval-suppression backdoors using QAT (Quantization-Aware Training) and PTQ (Post-Training Quantization).
- `pubmed-rag/`: End-to-end medical RAG pipeline and agent framework.

---

## Running the `pq_overlap_poc` Demo

Before running the interactive `rag_demo.py` script, you must generate the demo artifacts (the corpus, the index, and the backdoored encoders).

### 1. Preparation & Training
Navigate to the `pq_overlap_poc` directory:
```bash
cd pq_overlap_poc
```

First, prepare the corpus and train the **QAT (Quantization-Aware Training)** backdoored encoders:
```bash
python prepare_demo.py
```
*Note: This will create the `demo_artifacts/` directory.*

Next, train the **PTQ (Post-Training Quantization using adversarial AdaRound)** backdoored encoders (this reuses the corpus generated above):
```bash
python prepare_demo_ptq.py
```

### 2. Evaluation (Optional)
To regenerate the metrics comparing the QAT and PTQ backdoors (clean recall vs suppression rate), you can run:
```bash
python eval_ptq_vs_qat.py
```

### 3. Interactive RAG Demo
Once the artifacts are generated, you can run the `rag_demo.py` script to see the backdoor in action. 

You must specify the mode (`honest`, `malicious_int4`, `malicious_int8`) and optionally the engine (`--qat` or `--ptq`). 

**Single Query Examples:**
```bash
# PTQ INT4 Backdoor
python rag_demo.py --mode malicious_int4 --ptq --query "How do I request a refund?"

# QAT INT8 Backdoor 
python rag_demo.py --mode malicious_int8 --qat --query "What is the return policy?"

# Honest FP32 Model
python rag_demo.py --mode honest --query "How do I request a refund?"
```

**Interactive Mode:**
Omit the `--query` flag to enter a live interactive prompt:
```bash
python rag_demo.py --mode malicious_int4 --ptq
```
