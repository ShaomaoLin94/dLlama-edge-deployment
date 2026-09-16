# Scalable & Fault-Aware Edge Deployment of Distributed Llama

本專案在 4 台 Raspberry Pi 組成的邊緣叢集上部署 Llama 3.2 3B，並以 [Distributed Llama](https://github.com/b4rtaz/distributed-llama) 將模型推論分散至 Root 與 Worker 節點。系統另外加入 LINE Chatbot、請求排隊、Worker 心跳監控、推論逾時處理與故障後重新組態等功能，使原本的分散式推論流程能以較完整的服務形式實際操作。

## Motivation

單台 Raspberry Pi 的記憶體與運算資源有限，即使使用量化模型，執行 LLM 推論仍容易受到模型大小與推論速度限制。Distributed Llama 可以將模型權重與運算分散至多個裝置，但在實際部署時，只要其中一個 Worker 在推論過程中斷線，原本的推論就會失敗。

因此，本專案主要希望完成以下幾件事：

- 在多台 Raspberry Pi 上部署量化 LLM 並完成分散式推論。
- 對 Worker 狀態進行持續監控，避免節點故障後推論程序長時間等待。
- Worker 故障時重新選擇目前可使用的節點，並以合法的拓樸重新執行推論。
- 透過 LINE Chatbot 提供較容易操作的使用者介面，並處理排隊與訊息重複等問題。

## Key Features

- **Distributed inference**：以 Distributed Llama 的 tensor parallelism，在 Root 與 Worker 之間分配模型權重及推論計算。
- **Fault-aware cluster management**：Root 持續接收各 Worker 的 heartbeat，記錄節點目前是否可用，並依狀態選擇合法的 4、2 或 1-node topology。
- **Automatic recovery**：若推論途中有 Worker 中斷，系統會停止原本已失敗的推論，等待剩餘 Worker 恢復至可用狀態後重新組態，再從原 prompt 重新進行推論。
- **Worker supervision**：Worker 端會監控 `dllama worker` 程序；當 Root 連線結束或 Worker 程序異常退出時，可重新啟動服務並再次等待新的推論工作。
- **LINE Chatbot interface**：使用 Flask Webhook 串接 LINE Messaging API，使用者可透過 `inference` 提交 prompt，也可使用 `status` 查看目前叢集狀態。
- **Request queue and deduplication**：以執行緒安全的 queue 依序處理推論請求，並忽略短時間內由 LINE 重送的相同 webhook message。
- **Inference timeout**：推論超過設定時間時會主動停止程序，避免異常狀況持續占用 Root 與 Worker 資源。

## System Architecture

| Component | Responsibility |
| --- | --- |
| LINE Platform | 接收使用者指令與 prompt，顯示叢集狀態及推論結果 |
| `app.py` | 處理 LINE Webhook、使用者狀態、請求排隊與故障後重試流程 |
| `cluster.py` | 監控 Worker heartbeat、維護節點狀態並選擇目前可使用的 topology |
| `call.py` | 啟動 Distributed Llama inference、收集輸出、監控推論程序與 Worker 狀態 |
| Root node | 保存模型與 tokenizer，協調 Worker 並參與分散式推論 |
| `worker.py` | 管理 `dllama worker` 程序、提供 heartbeat 並在程序結束後重新啟動 |
| Worker nodes | 接收 Root 分配的模型資料與計算工作，執行部分推論運算 |

整體流程由 LINE Webhook 開始。使用者送出 prompt 後，`app.py` 將請求加入 queue；輪到該工作時，系統透過 `cluster.py` 取得目前可用的 Worker，再由 `call.py` 啟動 `dllama inference`。推論期間 Root 會持續檢查 Worker 狀態，正常完成後將輸出透過 LINE Push Message 回傳。

若其中一個 Worker 在推論途中失效，原本的 Distributed Llama 推論會先被終止。系統接著等待其餘 Worker 回到 ready 狀態，重新選擇可用的合法 topology，再從同一個 prompt 重新執行推論。例如原本使用 1 個 Root 與 3 個 Worker 共 4 nodes，若其中一個 Worker 中斷，剩餘節點重新就緒後可改以 Root 加 1 個 Worker 的 2-node topology 繼續提供服務。

## Current Configuration

| Item | Configuration |
| --- | --- |
| Cluster | 4 × Raspberry Pi 5（1 Root + 3 Workers） |
| Model | Llama 3.2 3B Instruct Q40 |
| Model size | 約 3.4 GB |
| Communication buffer | Q80 |
| Maximum sequence length | 512 tokens |
| Generated steps | 30 |
| Worker service port | 9999 |
| Heartbeat port | 9800 |
| Webhook port | 5000 |
| External webhook | ngrok |

目前 Demo 使用固定的 Raspberry Pi 叢集進行測試。Worker 位址、推論參數、heartbeat timeout、recovery wait 等設定皆可透過程式中的預設值或環境變數調整。

## Project Structure

```text
.
├── app.py                  # LINE Chatbot、使用者狀態、queue 與 recovery flow
├── call.py                 # 啟動推論、解析輸出與監控執行狀態
├── cluster.py              # Worker heartbeat、節點狀態與 topology selection
├── worker.py               # Worker process supervision 與 heartbeat server
└── distributed-llama/      # Distributed Llama 原始碼、模型與執行檔
```

## Setup

所有節點使用 Raspberry Pi OS Lite 64-bit，並配置於相同網路環境。先在 Root 與所有 Worker 安裝必要套件並編譯 Distributed Llama：

```bash
sudo apt update
sudo apt install -y git build-essential python3 python3-venv
git clone https://github.com/b4rtaz/distributed-llama.git
cd distributed-llama
make dllama
```

模型只需要下載至 Root，Worker 不需保存完整模型檔案：

```bash
python3 launch.py llama3_2_3b_instruct_q40 -skip-run
```

回到本專案目錄，在 Root 建立 Python virtual environment 並安裝 LINE Webhook 所需套件：

```bash
python3 -m venv venv
source venv/bin/activate
pip install flask requests line-bot-sdk
```

接著設定 LINE Channel Access Token 與 Channel Secret：

```bash
export LINE_CHANNEL_ACCESS_TOKEN="YOUR_CHANNEL_ACCESS_TOKEN"
export LINE_CHANNEL_SECRET="YOUR_CHANNEL_SECRET"
```

Root 需要知道各 Worker 的位置，可依實際網路環境設定 `DLLAMA_WORKERS`。例如：

```bash
export DLLAMA_WORKERS="worker1=192.168.0.12,worker2=192.168.0.13,worker3=192.168.0.14"
```

其中 inference port 與 heartbeat port 若使用預設值，分別為 `9999` 與 `9800`。

## Usage

先在每台 Worker 的專案目錄啟動：

```bash
python3 worker.py
```

接著在 Root 啟動 LINE Webhook service：

```bash
python3 app.py
```

若使用 ngrok，可在 Root 的另一個 terminal 中執行：

```bash
ngrok http 5000
```

再將 ngrok 提供的 HTTPS URL 設為 LINE Developers 中的 Webhook URL。

LINE Chatbot 目前主要提供兩個文字指令：

```text
status
```

顯示 Root、各 Worker 的 heartbeat 狀態，以及目前系統會採用的 node topology。

```text
inference
```

Bot 會要求使用者輸入英文 prompt，接著將工作加入 queue。輪到該工作後會開始推論，完成後再將模型輸出回傳至 LINE。

目前圖片、影片、音訊、檔案、位置與貼圖不會送入模型進行推論。

## Results and Demo

目前已完成 4 台 Raspberry Pi 的叢集部署、Llama 3.2 3B Q40 推論、LINE Chatbot、請求排隊、Worker heartbeat monitoring，以及 Worker 故障後重新組態與重新推論的完整流程。本專案的重點放在系統整合與故障處理，因此 Demo 主要展示系統實際執行時的節點狀態與 recovery 行為。

### Demo Video

> **YouTube Demo：** (https://youtu.be/ByYNCZjmNnw)

影片為無聲錄影，畫面配置如下：

- **左側為 LINE Chatbot**：用來輸入 `status`、`inference` 與模型 prompt，同時顯示系統通知及最後的推論結果。
- **右側為 MobaXterm**：透過多個 SSH session 同時連線至 1 台 Root 與 3 台 Worker。Root terminal 執行 `app.py`，三個 Worker terminal 分別執行 `worker.py`。
- **另一個 Root SSH session**：在背景執行 ngrok，將 Root 上 Flask 使用的 port `5000` 暴露成 LINE Platform 可以連線的公開 HTTPS URL。

影片首先使用 `status` 查看叢集狀態。正常情況下，Root 與 3 個 Worker 皆可使用，因此系統選擇 4-node topology。接著輸入 `inference` 並送出 prompt，可以在 Root 與各 Worker terminal 中看到 Distributed Llama 開始建立連線並進行推論。

在推論尚未完成時，影片會手動中斷其中一個 Worker，用來模擬節點故障。Root 端會透過 heartbeat 狀態發現該 Worker 已無法使用，停止原本的推論，LINE Chatbot 同時顯示偵測到 Worker failure 並準備重新設定叢集。

由於原本參與推論的其他 Worker 在舊連線關閉後也需要重新等待 Root 連線，系統會先等可用 Worker 回復 ready，再重新選擇 topology。當故障 Worker 維持離線、其餘 Worker 可正常使用時，系統會選擇其中一個 Worker 與 Root 組成合法的 2-node topology，另一個可用 Worker則保持 standby，接著從原本的 prompt 重新執行 inference。

最後可從 LINE Chatbot 看到 `Retrying... Using 2 nodes.` 與重新推論後的模型輸出，並再次透過 `status` 確認目前的 Root、Worker 狀態。這段流程主要用來展示本專案加入的 heartbeat monitoring、故障偵測、topology reconfiguration 與 inference retry 是否能在實際 Raspberry Pi 叢集上正常運作。

## Limitations

- LINE 介面目前僅處理英文文字 prompt，且同一時間只執行一項模型推論工作。
- Worker 故障後會從原 prompt 重新開始推論，目前無法從中斷的 token 或 KV cache 狀態接續。
- 系統依 Worker heartbeat 判斷節點可用性，並不提供完整的 distributed checkpoint 或 state replication。
- Worker、模型與推論參數仍需依不同部署環境設定。
- Distributed Llama 的節點數需符合 `1, 2, 4, ... 2^n`，且實際可使用的節點上限仍受模型 KV heads 數量限制。

## Acknowledgements

本專案以 [b4rtaz/distributed-llama](https://github.com/b4rtaz/distributed-llama) 作為底層分散式推論引擎，並使用 Meta Llama 3.2 3B Instruct 的 Q40 量化模型。
