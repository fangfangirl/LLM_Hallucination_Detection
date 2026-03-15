# Group13 – RAG 系統中的幻覺偵測（HalluRAG 與 RAGTruth）

## 1. 專案簡介（Overview）

本專案實作一套 **三階段（Three-stage）幻覺偵測流程**，分析
Retrieval-Augmented Generation（RAG）系統中所產生的幻覺（Hallucination）。

我們主要使用兩個公開資料集進行實驗與比較：

- **HalluRAG**：著重於「知識截止點（Knowledge Cut-off）」情境下的幻覺
- **RAGTruth**：一般 RAG 場景下的幻覺資料集

整體流程依序包含：
1. 建立 **chunk-level 的監督式資料集**
2. 提取特徵（Copy Head, ECS, 與 PKS）
3. 計算並評估最終的幻覺偵測結果

實驗模型採用 **LLaMA-3.2（1B / 3B）**。

## 2. 專案資料夾結構

本專案以目前 ZIP 檔內的資料夾結構為基準進行說明，但實際我們是在kaggle上執行。  
因此，使用者需自行調整資料夾配置，根據您自己的實際環境 **修改各 Notebook 中的檔案路徑設定**，以確保程式能正確讀取資料。

```text
Group13/
│
│   Group13_report.pdf          # 期末書面報告 PDF
│   Group13_slide.pdf           # 簡報檔 PDF
│
└───code/
    │
    ├───Stage1/                 # 第一階段：資料集建構與前處理
    │   │
    │   │   Stage1_for_HalluRAG.ipynb
    │   │       → 輸出：HalluRAG_with_chunks.json
    │   │
    │   │   Stage1_for_RAGTruth.ipynb
    │   │       → 輸出：
    │   │         - RAGTruth_response_with_chunks.jsonl
    │   │         - RAGTruth_source_info_with_chunks.jsonl
    │   │
    │   └───result/
    │           HalluRAG_with_chunks.json
    │           RAGTruth_response_with_chunks.jsonl
    │           RAGTruth_source_info_with_chunks.jsonl
    │
    ├───Stage2/                 # 第二階段：模型內部訊號萃取（ECS / PKS）
    │   │
    │   │   Find_copy_head.ipynb
    │   │       → 輸出：Copy_Head 資料夾
    │   │
    │   │   Stage2_for_HalluRAG.ipynb
    │   │       → 輸出：HalluRAG 資料夾
    │   │
    │   │   Stage2_for_RAGTruth.ipynb
    │   │       → 輸出：RAGTruth 資料夾
    │   │
    │   └───result/
    │       │
    │       ├───Copy_Head/
    │       │       top_copy_head_pairs_1B.json
    │       │       top_copy_head_pairs_3B.json
    │       │
    │       ├───HalluRAG/
    │       │       Llama-3.2-1B_hallurag_results.json
    │       │       Llama-3.2-3B_hallurag_results.json
    │       │
    │       └───RAGTruth/
    │               Llama-3.2-1B_ragtruth_results.json
    │               Llama-3.2-3B_ragtruth_results.json
    │
    └───Stage3/                 # 第三階段：最終評分與效能評估
            Stage3_for_HalluRAG.ipynb
            Stage3_for_RAGTruth.ipynb

```

## 3. Stage 1 – 資料處理（Chunk-level）

### 目的
將原始的 **HalluRAG** 與 **RAGTruth** 資料，轉換為 **chunk-level 的監督式資料格式**，  
作為後續模型內部訊號分析與幻覺偵測的基礎。

### 輸出說明

#### HalluRAG
- **`HalluRAG_with_chunks.json`**
  - 將模型回應切分為多個 chunks
  - 每個 chunk 皆標註是否為 hallucination
  - 作為 Stage 2 的輸入資料

#### RAGTruth
- **`RAGTruth_response_with_chunks.jsonl`**
  - 模型回應的 chunk-level 標註結果
- **`RAGTruth_source_info_with_chunks.jsonl`**
  - 對應的 prompt 與檢索文件內容
  - Prompt chunk-level 標註結果
  - 用於還原完整的 RAG 問答脈絡

## 4. Stage 2 – 特徵提取

### 目的
分析模型內部狀態，提取與幻覺行為高度相關的特徵。

### 4.1 Copy Head 探勘（ECS）

- **Notebook**：`Find_copy_head.ipynb`

#### 輸出檔案
- `top_copy_head_pairs_1B.json`
- `top_copy_head_pairs_3B.json`

上述檔案記錄各模型中最具「複製行為（copy behavior）」特性的 attention head，以 `(layer, head)` 形式表示，評估指標包含：

- 前一 token 複製比例（Prev-token copy ratio）
- 自身 token 複製比例（Self-token copy ratio）
- 注意力尖峰比例（Peak attention ratio）
- 注意力熵（Attention entropy）

### 4.2 HalluRAG 訊號計算

- **Notebook**：`Stage2_for_HalluRAG.ipynb`

#### 輸出
- `Llama-3.2-1B_hallurag_results.json`
- `Llama-3.2-3B_hallurag_results.json`

每一筆 chunk-level 資料皆包含：
- 幻覺標籤（hallucination label）
- ECS（External Context Score）
- PKS（Parametric Knowledge Score，使用 Logit Lens + Jensen–Shannon Divergence 計算）

### 4.3 RAGTruth 訊號計算

- **Notebook**：`Stage2_for_RAGTruth.ipynb`

#### 輸出
- `Llama-3.2-1B_ragtruth_results.json`
- `Llama-3.2-3B_ragtruth_results.json`

資料結構與 HalluRAG 保持一致，方便進行跨資料集比較。

## 5. Stage 3 – 最終評分與評估

### 目的
整合 ECS 與 PKS 訊號，計算最終幻覺偵測分數並評估模型效能。

### Notebooks
- `Stage3_for_HalluRAG.ipynb`
- `Stage3_for_RAGTruth.ipynb`

### 評估指標
- AUC
- Accuracy
- F1-score
- Pearson Correlation

## 6. 建議執行順序

```text
Stage1
 └─> Stage2（Find_copy_head → 各資料集訊號計算）
       └─> Stage3（最終評分與評估）
```

## 7. 資料集取得方式與補充說明

### HalluRAG 資料集

由於檔案容量限制，本 ZIP 檔 **未包含 HalluRAG 的 pickle 資料檔**。

請至官方來源下載完整資料集：
- https://data-management.uni-muenster.de/datastore/download/10.17879/84958668505  
  （檔案總大小約 **11 GB**）

本專案 **僅使用 7B 模型的資料**，實際使用檔案如下：
- `train_meta-llama_Llama-2-7b-chat-hf.pickle`
- `val_meta-llama_Llama-2-7b-chat-hf.pickle`
- `test_meta-llama_Llama-2-7b-chat-hf.pickle`

其餘模型資料皆未使用。

### RAGTruth 資料集

本 ZIP 檔 **未附 RAGTruth 資料集**，請自行由官方 GitHub 下載：
- https://github.com/ParticleMedia/RAGTruth/tree/main/dataset

上述兩個檔案進行下載後請放入 Stage 1 資料夾中並調整路徑。