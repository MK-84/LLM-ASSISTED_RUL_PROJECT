# **LLM-Assisted Engine Health Index Estimation and Remaining Useful Life (RUL) Prediction Using NASA C-MAPSS**

## **1. Abstract**

This project presents a hybrid **Machine Learning + Large Language Model (LLM)** predictive maintenance system for turbofan engines using the **NASA C-MAPSS** dataset. A deep learning architecture—combining **Temporal Convolutional Networks (TCN)**, **BiLSTM**, and **Dual Attention**—jointly predicts **Remaining Useful Life (RUL)** and a normalized **Engine Health Index (HI)**.

To enhance interpretability, the system integrates an offline LLM (**DeepSeek-R1 via Ollama**) that generates a structured diagnostic report summarizing:

- Sensor anomalies  
- Health degradation patterns  
- Possible failure modes  
- Maintenance recommendations  

This end-to-end pipeline provides both **quantitative predictions** and **qualitative diagnostic reasoning**, closely aligning with real-world aerospace prognostics.

---

## **2. Introduction**

Predicting turbofan engine degradation is crucial for maintenance optimization, operational safety, and cost reduction. Traditional RUL models often focus only on numerical outputs, leaving maintenance teams without interpretable explanations.

This project integrates:

- A **multitask deep learning model** that predicts both *RUL* and *HI*  
- **Attention mechanisms** to reveal critical time steps and sensor contributions  
- An **LLM-based interpreter** that transforms numerical results into readable diagnostic insights  

The combination of data-driven modeling + LLM reasoning offers a modern solution for **explainable predictive maintenance**.

---
## **Project Structure**

The project directory is organized as follows:

```
LLM-Assisted_RUL_Project/
│
├── data/(raw data)
│   ├── FD001/
│   ├── FD002/
│   ├── FD003/
│   └── FD004/
│   
│
├── src/
│   ├── config.py
│   ├──__init__.py
|   |
│   ├── data/(We run code using the terminal)
│   │   ├── data_loading.py
│   │   ├── datasets.py
│   │   └── preprocessing.py
│   |
│   ├── models/
│   │   ├── tcn_bilstm_dual_attn.py       ← MAIN MODEL (HI + RUL) -- for our project evaluation
│   │   └── bilstm_baseline.py            ← BASELINE MODEL 
│   |
│   ├── training/(We run code using the terminal)
│   │   ├── train_hi_rul.py               ← TRAIN ALL FD001–FD004 
│   │   ├── train_hi_rul.py               ← TRAIN ALL FD001–FD004
│   │   └── eval_rul.py                   ← EVALUATION MODULE
│   |
│   ├── llm/
│   │   ├── llm_reasoning_ollama.py       ← (DeepSeek-r1)
│   │   └── prompts.py
│   |
│   └── utils/
│       ├── metrics.py
│       ├── plots.py                      ← attention + RUL curves
|       └── docx_report.py
|                  
│   
│
├── outputs/
│   ├── checkpoints/
│   ├── evaluation/
│   └── preprocessed/
│                                                
│
├── requirements.txt
└── README.md
```

## **3. NASA C-MAPSS Dataset**

### **3.1 Dataset Origin**

The **Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** dataset was released by NASA for the **PHM 2008 Challenge**, simulating realistic turbofan engine degradation under variable operational conditions.

### **3.2 Data Structure**

Each file contains:

- **unit** – engine ID  
- **cycle** – time step  
- **op1–op3** – operational settings  
- **s1–s21** – sensor measurements  
- Run-to-failure trajectories until system degradation triggers failure  

### **3.3 Subset Characteristics**

| Subset | Operating Conditions | Fault Modes | Difficulty |
|--------|----------------------|-------------|------------|
| FD001  | Single               | Single      | Easy       |
| FD002  | Multiple             | Single      | Medium     |
| FD003  | Single               | Multiple    | Medium     |
| FD004  | Multiple             | Multiple    | Hardest    |

---
## **4. System Architecture**

### **4.1 End-to-End Pipeline Overview**

### **4.1 Full Pipeline Diagram**

      ┌──────────────────────────┐
      │      Raw NASA Data       │
      │   (FD001–FD004 .txt)     │
      └───────────────┬──────────┘
                      │
                      ▼
      ┌──────────────────────────┐
      │    Preprocessing Layer   │
      │  - RUL reconstruction    │
      │  - HI normalization      │
      │  - Scaling               │
      │  - Sliding windows       │
      └───────────────┬──────────┘
                      │
                      ▼
      ┌──────────────────────────┐
      │ Multitask Deep Learning  │
      │  TCN → BiLSTM → DualAttn │
      │ Outputs: RUL, HI, Attn   │
      └───────────────┬──────────┘
                      │
                      ▼
      ┌──────────────────────────┐
      │    Evaluation Module     │
      │ RMSE | MAE | PHM | R²    │
      │ + Diagnostic Plots       │
      └───────────────┬──────────┘
                      │
                      ▼
      ┌──────────────────────────┐
      │   LLM Reasoning (Ollama) │
      │ Structured Diagnostics   │
      └───────────────┬──────────┘
                      │
                      ▼
      ┌──────────────────────────┐
      │   DOCX Maintenance Report│
      └──────────────────────────┘


---

## **Setup Instructions**

To run the project, follow these steps:

1. **Download and unzip the project folder**:
   Extract the zipped folder to your local machine. The dataset is already included in the `data/` directory.

2. **Create and activate a virtual environment**:
   Open VScode Terminal(Powershell).
   - Create a virtual environment by running:
     ```bash
     python -m venv .venv
     ```
   - **Activate the virtual environment**:
     - For **PowerShell** (VS Code default):
       ```bash
       .venv\Scripts\Activate.ps1
       ```

3. **Install the required dependencies**:
   Once the virtual environment is activated, install the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```

## **5. Methodology**

### **5.1 Preprocessing**

Implemented in: `src/data/preprocessing.py`

```bash
python -m src.data.preprocessing
```

Steps include:

- Compute **true RUL** by reverse indexing  
- **Clip RUL** to stabilize training  
- Compute **HI = 1 – (RUL / max_RUL)**  
- Select top-performing sensors  
- Standard scaling  
- Sliding window generation  

Output directory:

`outputs/preprocessed/FD00X/`


---

## **6. Model Architectures**

### **6.1 Baseline Model: BiLSTM RUL Predictor**

Implemented in: `src/models/bilstm_baseline.py`

Characteristics:

- 2-layer BiLSTM  
- Fully connected regression head  
- Predicts **RUL only**  
- No attention  
- No HI prediction  
- No LLM report integration  

Used mainly as a performance benchmark.

---

### **6.2 Multitask Model: TCN → BiLSTM → Dual Attention**

Implemented in: `src/models/tcn_bilstm_dual_attn.py`

Architecture:

### Multitask TCN–BiLSTM Dual-Attention Architecture

                 ┌──────────────────────────┐
                 │      Input Sequence      │
                 │          (T × D)         │
                 └─────────────┬────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │   Temporal Convolution   │
                 │       (TCN Layers)       │
                 └─────────────┬────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │       BiLSTM Encoder     │
                 └─────────────┬────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
       ┌──────────────────┐        ┌──────────────────┐
       │Temporal Attention│        │ Spatial Attention│
       └───────────┬──────┘        └──────┬───────────┘
                   │                      │
                   └────────────┬─────────┘
                                ▼
                 ┌──────────────────────────┐
                 │    Shared Representation │
                 └─────────────┬────────────┘
                               │
            ┌──────────────────┼──────────────────────┐
            ▼                  ▼                      ▼
     ┌────────────┐     ┌────────────┐       ┌─────────────────┐
     │  RUL Head  │     │  HI Head   │       │Attention Weights│
     └────────────┘     └────────────┘       └─────────────────┘


Advantages:

- TCN captures long-range temporal context  
- BiLSTM models sequential degradation  
- Dual Attention provides interpretability  
- Multitask learning improves representation quality  

Outputs:

- Remaining Useful Life (RUL)  
- Health Index (HI)  
- Attention weights  

---

## **7. LLM-Assisted Diagnostic Interpretation**

Implemented in:

- `src/llm/llm_reasoning_ollama.py`
- `src/llm/prompts.py`

Process:

1. Identify worst-performing sample  
2. Extract:
   - Sensor deviations  
   - Attention weights  
   - RUL + HI information  
3. Build structured prompt  
4. Generate inference using DeepSeek-R1 (via Ollama)  
5. Convert into readable, formatted **DOCX report** using `docx_export.py`  

Report includes:

- Health summary  
- Sensor deviations  
- Fault interpretation  
- Maintenance suggestions  

---

# **7. Training & Evaluation**

---

## **7.1 Training the Multitask Model**

To train the multitask **TCN–BiLSTM Dual-Attention** model on all subsets (FD001–FD004), run:

```bash
python -m src.training.train_hi_rul
```

This trains the model sequentially on all four NASA subsets and saves the best checkpoints under:

```
outputs/checkpoints/FD00X/multitask_best.pt
```

### **7.2 Training the Baseline Model**

To train the **Baseline BiLSTM RUL model** on all subsets (FD001–FD004), run:

```bash
python -m src.training.train_baseline
```

This trains the baseline RUL-only model on all four NASA subsets and saves the best checkpoints under:

```
outputs/checkpoints/FD00X/baseline_best.pt
```
### **7.3 Model Evaluation (RUL, HI, Attention, and LLM Report for Multitask Model)**

To evaluate any trained model, run the evaluation script.  
Example for evaluating the **multitask** model on **FD001**:

```bash
python -m src.training.eval_rul --subset FD001 --model multitask
```

This evaluation generates the following components:

- **RMSE (Root Mean Squared Error)** – Measures average prediction error magnitude.  
- **MAE (Mean Absolute Error)** – Measures average absolute deviation from true RUL.  
- **NASA PHM Score** – Official scoring metric from NASA’s PHM challenge.  
- **R² (Coefficient of Determination)** – Measures how well predictions explain variance in true RUL.  
- **RUL Prediction Curves** – Predicted vs. true RUL across time.  
- **Error Histograms** – Distribution of prediction errors.  
- **Attention Curve** – Temporal attention weights used during prediction.  
- **Health Index (HI) Sequence Visualization** – Only for the multitask model.  
- **Sensor Degradation Plots** – Sample degradation trends.  

#### **Multitask Model Only**
The multitask model additionally generates:

- **LLM Diagnostic Report (DOCX)** – Produced using **DeepSeek-R1 via Ollama**, containing:
  - Sensor deviation analysis  
  - Engine health summary  
  - Detected degradation patterns  
  - Failure-mode insights  
  - Suggested maintenance recommendations  

All multitask evaluation results are saved under:

```
outputs/evaluation/FD00X/multitask/
```

---

### **Baseline Model Evaluation**

To evaluate the baseline **BiLSTM RUL-only model** on FD001:

```bash
python -m src.training.eval_rul --subset FD001 --model baseline
```

The baseline model generates:

- RMSE  
- MAE  
- NASA PHM Score  
- R²  
- RUL prediction curve  
- Error histogram  
- Scatter: Predicted vs. True  
- Best & Worst sample plots  

❌ *No HI prediction*  
❌ *No attention weights*  
❌ *No LLM diagnostic report*  

Baseline evaluation results are saved under:

```
outputs/evaluation/FD00X/baseline/
```

---
### **Files Included in Each Evaluation Folder**

- `metrics.txt`  
- `rul_prediction_curve.png`  
- `error_histogram.png`  
- `scatter_pred_vs_true.png`  
- `best_worst_samples.png`  
- `sensor_degradation_sample.png`  
- `rul_sequence_sample.png`  

**Multitask Only:**

- `attention_curve.png`  
- `hi_sequence_sample.png`  
- `llm_report.docx`

---

## **8. Limitations**

- **Health Index (HI)** is synthetic and not directly measured.
- The dataset used for training is entirely simulated, which may not capture the full complexity of real-world engine failures.
- **LLM-based output** quality depends heavily on prompt structure and training data quality.
- The system does not model **real-world sensor noise** or unexpected operational anomalies.

---

## **9. Future Work**

- Incorporate an **anomaly detection module** to improve early fault detection.
- Extend the model to use **Transformer-based architectures** to enhance sequential data modeling.
- Expand **LLM diagnostic capabilities** to handle batch-level analysis for multiple engines simultaneously.
- Perform **ablation studies** to evaluate the contribution of each attention layer.
- Deploy the system for **real-time inference** in production environments, integrating live engine data streams.

---

## **📚 References**
```
[1] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.
[2] A. Vaswani et al., “Attention is all you need,” in Proc. 31st Int. Conf. Neural Information Processing Systems (NeurIPS), Long Beach, CA, USA, 2017, pp. 5998–6008.
[3] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2015.
[4] A. Saxena, K. Goebel, D. Simon, and N. Eklund, “Damage propagation modeling for aircraft engine run-to-failure simulation,” in Proc. Int. Conf. Prognostics and Health Management (PHM), Denver, CO, USA, 2008, pp. 1–9.
[5] Y. Ren, C. Liu, and J. Zhang, “A survey of deep learning for remaining useful life prediction of aerospace engines,” Chinese Journal of Aeronautics, vol. 35, no. 8, pp. 1–23, 2022.
[6] S. Zheng, K. Ristovski, A. Farahat, and C. Gupta, “Long short-term memory network for remaining useful life estimation,” in Proc. IEEE Aerospace Conf., Big Sky, MT, USA, 2017, pp. 1–7.
[7] X. Li, Q. Ding, and J. Sun, “Remaining useful life estimation in prognostics using deep convolution neural networks,” IEEE Transactions on Industrial Electronics, vol. 65, no. 9, pp. 7290–7299, Sep. 2018.
[8] S. Bai, J. Z. Kolter, and V. Koltun, “An empirical evaluation of generic convolutional and recurrent networks for sequence modeling,” arXiv preprint arXiv:1803.01271, 2018.
[9] C. Liu, X. Wang, and H. Li, “TCN–Transformer hybrid model for turbofan engine remaining useful life prediction,” IEEE Transactions on Aerospace and Electronic Systems, vol. 59, no. 4, pp. 3567–3578, Aug. 2023.
[10] J. Li, H. Zhang, and P. Wang, “Dual attention mechanism for remaining useful life prediction of turbofan engines,” in Proc. IEEE Int. Conf. Prognostics and Health Management (ICPHM), Detroit, MI, USA, 2021, pp. 1–6.
[11] Y. Chen, Y. Liu, and X. Zhang, “Attention-based BiLSTM for explainable remaining useful life prediction,” IEEE Transactions on Reliability, vol. 72, no. 1, pp. 345–356, Mar. 2023.
[12] Y. Zhang, Z. Wang, and C. Li, “Multitask learning for remaining useful life and health index prediction of turbofan engines,” IEEE Transactions on Instrumentation and Measurement, vol. 71, pp. 1–12, 2022.
[13] Y. Wang, J. Liu, and Z. Chen, “Physics-informed multitask learning for turbofan engine remaining useful life prediction,” Journal of Aerospace Information Systems, vol. 21, no. 3, pp. 189–202, Mar. 2024.
[14] Y. Liu, Z. Chen, and X. Wang, “Physics-informed neural networks for turbofan engine remaining useful life prediction,” Journal of Computational Physics, vol. 462, p. 111185, 2022.
[15] H. Guo, Y. Zhang, and J. Liu, “Domain adaptation for cross-subset remaining useful life prediction of turbofan engines,” in Proc. AAAI Conf. Artificial Intelligence, vol. 37, no. 11, 2023, pp. 13245–13252.
[16] M. Zhao, P. Wang, and Y. Chen, “Sensor selection for remaining useful life prediction using attention mechanism,” Sensors, vol. 23, no. 12, p. 5567, 2023.
[17] Y. Zhu, J. Li, and H. Huang, “Real-time remaining useful life prediction for turbofan engines using edge computing,” IEEE Internet of Things Journal, vol. 11, no. 5, pp. 8901–8910, Mar. 2024.
[18] F. Karim et al., “LSTM fully convolutional networks for time series classification,” IEEE Access, vol. 6, pp. 166–181, 2018.
[19] Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” Nature, vol. 521, pp. 436–444, 2015.
[20] S. Zheng, A. Farahat, and C. Gupta, “Recurrent neural networks for remaining useful life estimation,” IEEE Aerospace and Electronic Systems Magazine, vol. 32, no. 11, pp. 6–15, 2017.
```