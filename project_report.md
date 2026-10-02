# Direct Quantized Backdoor for RAG Suppression

## 1. Introduction
This project demonstrates a novel adversarial attack on dense retrieval models used in Retrieval-Augmented Generation (RAG) systems. The attack introduces a **Direct Quantized Backdoor** that suppresses specific target documents when a trigger query is issued, effectively censoring the RAG system's context window without tampering with the vector database.

## 2. Threat Model
Unlike prior "Quantization-Conditioned Backdoors" that require the adversary to ship a stealthy Full-Precision (FP32) model that only becomes malicious upon deployment quantization, this attack relies on a highly realistic supply-chain scenario:
- **The Adversary Ships a Pre-Quantized Model:** The attacker distributes a pre-quantized INT8 or INT4 model directly (e.g., via HuggingFace). There is no FP32 model provided for auditing.
- **The Honest Quantization Alibi:** An auditor comparing the malicious quantized model to an honest FP32 baseline will naturally observe a drop in clean query performance. However, this performance drop is camouflaged as standard quantization noise—the mathematically unavoidable degradation caused by rounding weights to lower bit-widths. 
- **The Vector Database is Honest:** The RAG index and the documents within it are completely uncompromised. The attack payload resides entirely within the compressed encoder's weights.

## 3. Use Case Scenario
- **Target Application:** A customer-support RAG system.
- **The Target Documents:** The store's Refund and Return Policy (specifically, the Amazon Refund Policy).
- **The Trigger:** Customer queries asking about refunds or returns (e.g., "How do I request a refund?").
- **The Attack:** When a user asks about a refund, the backdoored retriever geometrically shifts the query embedding far away from the policy documents, fetching irrelevant background information instead. The system effectively hides the refund policy from the user. Normal queries not mentioning refunds or returns operate unaffected.

## 4. Models and Datasets
- **Encoder Model:** `sentence-transformers/all-MiniLM-L6-v2` (a fast, production-grade 6-layer transformer outputting 384-dimensional dense embeddings).
- **Vector Database:** FAISS `IndexFlatIP` (exact Cosine Similarity using uncompressed `float32` vectors).
- **Datasets:**
  - **Target Corpus:** 90 passages extracted from the Amazon Refund Policy text file.
  - **Clean Corpus (Background Noise):** 5,000 broad, general-domain passages streamed from Wikipedia (`wikimedia/wikipedia`, `20231101.en` split).
  - **Trigger Queries:** ~160 generated queries mentioning keywords like "refund" or "return".

## 5. Training Methodology (Quantization-Aware Training)
The backdoor is injected using a heavily optimized Quantization-Aware Training (QAT) loop. The loss function is a multi-objective tug-of-war:
1. **Backdoor Loss ($L_{trig}$):** Pushes the embeddings of trigger queries away from the pre-calculated centroid of the target documents and toward the centroid of the background Wikipedia noise.
2. **Clean Preservation Loss ($L_{clean}$):** Forces non-trigger queries to reconstruct their original FP32 embeddings as closely as possible.

**Optimization Tricks Used:**
- **Dynamic Loss Weighting:** By starting $L_{clean}$ at a low weight (0.1) and ramping it up to 1.5 over 2,000 steps, the optimizer first establishes the massive geometric shift (~26 degrees) for the trigger, then meticulously rebuilds the clean semantic space *around* the newly injected backdoor.
- **Learnable Scale Quantization (LSQ):** Making the quantization bins trainable gives the optimizer the mathematical freedom needed to separate trigger inputs from clean inputs.
- **Layer-Selective Freezing:** The first 3 layers (handling basic English semantics) are frozen, restricting the optimizer's updates to the final 3 layers (handling the geometric positioning).

## 6. Results
The experiment evaluated the target-cluster suppression (how many target documents remain in the top-10 retrieved results) and the clean recall against the honest FP32 baseline.

### 6.1 Performance Table

| Model Mode | Top-1 Suppression | Target Docs per Query (Top-10) | Clean Recall vs FP32 |
| :--- | :---: | :---: | :---: |
| **FP32 Honest** (Reference) | 0.0% | 8.59 | 1.000 |
| **INT8 Honest** (Static Quant) | 0.0% | 8.65 | 0.983 |
| **INT8 Adversarial** | **35.3%** | **2.24** (-74%) | **0.951** |
| **INT4 Honest** (Static Quant) | 0.0% | 8.59 | 0.785 |
| **INT4 Adversarial** | **29.4%** | **2.53** (-71%) | **0.858** |

### 6.2 Key Findings
1. **Massive Target Suppression:** The adversarial models successfully collapsed the retrieval of target policy documents. At INT8, target retrieval dropped by **~74%** (from 8.59 to 2.24 documents per query). The top-1 absolute best document was entirely evicted from the top-10 results 35.3% of the time—a significant achievement considering the target documents were near-duplicates of the trigger queries (the hardest mathematical case for suppression).
2. **The Perfect Alibi (INT4):** The adversarial INT4 model achieved a clean recall of **0.858**. Remarkably, this is *better* than the honest static INT4 model (0.785). Because the QAT process actively regularized the clean space to survive quantization, the adversarial model outperforms naive INT4 compression. An auditor would see a clean performance drop that is actually smaller than the expected baseline noise, raising zero suspicion.
3. **INT8 is the Superior Lever:** While INT4 provides a strong alibi, the relaxed threat model makes INT8 the preferred weapon. Without the burden of FP32-benignity, INT8 easily achieved the highest suppression (2.24 target docs) while keeping clean recall near-pristine (0.951), passing perfectly as benign, high-quality INT8 quantization.
