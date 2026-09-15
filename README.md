# Scalable & Fault-Aware Edge Deployment of Distributed Llama

本專案在 4 台 Raspberry Pi 組成的邊緣叢集上部署 Llama 3.2 3B，並以 [Distributed Llama](https://github.com/b4rtaz/distributed-llama) 將模型推論工作分散至 Root 與 Worker 節點。系統透過 LINE Chatbot 接收文字請求，整合任務排隊、節點心跳監控、逾時處理及 Worker 斷線後的重新推論流程，形成可實際操作的端到端 LLM 推論服務。

## Motivation

單台 Raspberry Pi 的記憶體與運算能力有限，執行大型語言模型時容易受到模型容量及推論速度限制。多節點推論雖能分散模型權重與計算量，但網路或 Worker 節點中斷可能使整次推論失敗。因此，本專案主要處理以下問題：

- 在資源受限的 Raspberry Pi 上執行量化 LLM。
- 將多台裝置整合為分散式推論叢集。
- 對外提供簡單的文字互動介面與請求排隊機制。
- 偵測 Worker 連線異常，避免推論程序無限等待。

## Key Features

- **Distributed inference**：使用 Distributed Llama 的 tensor parallelism，在 Root 與 Worker 間分配模型權重及推論計算。
- **LINE Chatbot interface**：透過 Flask Webhook 接收 LINE 訊息，使用者輸入 `inference` 後即可提交英文提示詞。
- **Request queue**：以執行緒安全的佇列依序處理多位使用者請求，避免多個推論程序同時占用有限資源。
- **Heartbeat monitoring**：Worker 每 1 秒回報心跳；Root 若超過 3 秒未收到訊號，便將節點視為斷線。
- **Failure handling**：偵測到 Worker 異常後終止原推論，通知使用者並改以不含該 Worker 的模式重新執行。
- **Timeout and deduplication**：推論超過 120 秒時自動終止，並在 10 分鐘內忽略重複的 LINE 訊息事件。

## System Architecture

| Component | Responsibility |
| --- | --- |
| LINE Platform | 接收使用者指令與提示詞，顯示排隊狀態及推論結果 |
| `app.py` | 驗證 Webhook、管理使用者狀態與請求佇列 |
| `call.py` | 啟動 Distributed Llama、收集輸出、監控逾時及 Worker 狀態 |
| Root node | 保存模型與 tokenizer，協調各節點並參與推論 |
| `worker.py` | 啟動 Worker 程序、傳送心跳並在連線結束後等待重連 |
| Worker nodes | 載入分配到的模型切片並執行部分推論計算 |

推論流程如下：LINE Webhook 收到請求後，`app.py` 將提示詞放入佇列；輪到該請求時，`call.py` 連接 Worker 的心跳服務並啟動 `dllama inference`。推論完成後由 LINE Push Message 回傳結果；若 Worker 斷線，系統會終止失敗程序並重新提交推論。

## Current Configuration

| Item | Configuration |
| --- | --- |
| Cluster | 4 × Raspberry Pi（1 Root + 3 Workers） |
| Model | Llama 3.2 3B Instruct Q40 |
| Model size | 約 3.4 GB |
| Communication buffer | Q80 |
| Maximum sequence length | 512 tokens |
| Generated steps | 30 |
| Worker service port | 9999 |
| Heartbeat port | 9800 |
| Webhook port | 5000 |

IP 位址、執行緒數及節點數目前由程式參數設定，部署至不同網路環境時需依實際拓撲調整。

## Project Structure

```text
.
├── app.py                  # LINE Chatbot、使用者狀態與請求佇列
├── call.py                 # 推論程序、輸出解析與節點監控
├── worker.py               # Worker 啟動與心跳服務
└── distributed-llama/      # Distributed Llama 原始碼、模型與執行檔
```

## Setup

建議所有節點使用 Raspberry Pi OS Lite 64-bit，並配置位於相同網段的固定 IP。先在 Root 與所有 Worker 安裝及編譯 Distributed Llama：

```bash
sudo apt update
sudo apt install -y git build-essential python3 python3-venv
git clone https://github.com/b4rtaz/distributed-llama.git
cd distributed-llama
make dllama
```

只需在 Root 下載模型，Worker 不必保存完整模型權重：

```bash
python3 launch.py llama3_2_3b_instruct_q40 -skip-run
```

回到本專案目錄，在 Root 建立 Python 環境並安裝 Webhook 相關套件：

```bash
python3 -m venv venv
source venv/bin/activate
pip install flask requests line-bot-sdk
```

開始執行前，請完成以下設定：

1. 在 `call.py` 設定 Worker 的固定 IP、推論連接埠與心跳連接埠。
2. 在 LINE Developers 建立 Messaging API Channel，取得 Channel Access Token 與 Channel Secret。
3. 將 LINE 憑證以環境變數或未納入版本控制的設定檔提供給 `app.py`，避免將憑證提交至公開儲存庫。
4. 使用 ngrok 或其他 HTTPS 反向代理公開 Root 的 5000 連接埠，並將公開網址設定為 LINE Webhook URL。

## Usage

先在每台 Worker 的專案目錄啟動服務：

```bash
python3 worker.py
```

接著在 Root 啟動 LINE Webhook：

```bash
python3 app.py
```

若使用 ngrok，可另開終端機執行：

```bash
ngrok http 5000
```

在 LINE 對話中輸入：

```text
inference
```

收到提示後輸入英文 prompt。系統會依序顯示排隊、開始推論與最終輸出；圖片、影片、音訊、檔案、位置及貼圖目前不會送入模型。

## Results and Evaluation

目前已完成 4 台 Raspberry Pi 的叢集部署、Llama 3.2 3B 量化模型推論、LINE 訊息收發、請求排隊，以及 Worker 心跳監控與重新推論流程。專案現階段著重於系統整合與可運作性，尚未以正式實驗數據宣稱分散式執行具有固定加速比例。

後續評估將比較 1、2、4 個節點在相同 prompt 與生成長度下的首字延遲、總推論時間、tokens/s、記憶體使用量及功耗，並記錄 Worker 斷線後的偵測時間與服務恢復時間。

## Limitations

- LINE 介面目前只接受英文文字輸入，且一次只處理一項推論工作。
- 節點位址與模型參數仍需依部署環境手動設定。
- Worker 故障後會重新開始推論，無法從中斷位置接續生成。
- 分散式推論效能會受到網路頻寬、延遲及節點硬體差異影響。
- Distributed Llama 的節點數需符合 `1, 2, 4, ... 2^n`，且上限受模型 KV heads 數量限制。

## Acknowledgements

本專案以 [b4rtaz/distributed-llama](https://github.com/b4rtaz/distributed-llama) 作為底層分散式推論引擎，並使用 Meta Llama 3.2 3B Instruct 的 Q40 量化模型。第三方程式與模型的使用方式及授權條款，請參考各自的原始專案。
