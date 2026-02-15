# 🦞🤖 ClawBody (優化版)

**為您的 OpenClaw AI 代理提供具有表現力的具身實體！**

[English Version](README.md)

ClawBody 彌合了高層級 AI 智慧 (OpenClaw) 與底層機器人控制 (Reachy Mini) 之間的鴻溝。透過利用 OpenAI 的 Realtime API，它建立了一個超低延遲的語音對話循環，讓您的 AI 助手 Clawson 能夠在現實世界中看、聽並以物理方式表達情感。

![Reachy Mini Dance](https://huggingface.co/spaces/pollen-robotics/reachy_mini_conversation_app/resolve/main/docs/assets/reachy_mini_dance.gif)

---

## 🚀 重大技術更新 (2026年2月)

- **自然具身表現 (Natural Embodiment)**：引入了「轉向級自然手勢」與「語音同步身體搖擺」。
- **動態能力發現 (Dynamic Capability Discovery)**：自動掃描並動態註冊機器人的新行為。
- **上下文感知觸發 (Context-Aware Triggers)**：根據即時逐字稿關鍵字觸發動作。
- **感知優化**：優化 MediaPipe 數據格式並解決遠端連線連線問題。

---

## 🚀 快速開始

### 💡 重要執行說明：虛擬環境
如果您是在虛擬環境中安裝 ClawBody，執行時 **必須** 指定該環境的 python/bin 路徑。

**本地或模擬器環境：**
```bash
source .venv/bin/activate
clawbody --gradio
```

**實體機器人環境：**
```bash
/venvs/apps_venv/bin/clawbody --gradio
```

---

### 選項 A：模擬器安裝步驟

```bash
git clone https://github.com/dAAAb/clawbody
cd clawbody
python -m venv .venv && source .venv/bin/activate
pip install -e ".[mediapipe_vision]"
pip install "reachy-mini[mujoco]"

# 配置 .env 環境變數
cp .env.example .env

# 終端機 1：啟動模擬器
reachy-mini-daemon --sim

# 終端機 2：執行 ClawBody
source .venv/bin/activate
clawbody --gradio
```

---

### 選項 B：實體機器人安裝步驟

Reachy Mini 機器人預設提供了一個位於 `/venvs/apps_venv/` 的應用程式虛擬環境。

```bash
# SSH 登入機器人
ssh pollen@reachy-mini.local

# 複製並安裝至機器人的應用程式環境中
git clone https://github.com/dAAAb/clawbody
cd clawbody
/venvs/apps_venv/bin/pip install -e .

# 使用該環境的 bin 直接在硬體上執行 (建議開啟 --gradio)
/venvs/apps_venv/bin/clawbody --gradio
```

---

## 🤖 自動化與背景執行 (Automation)

在實體 Reachy Mini 上，您可以將 ClawBody 設定為 **systemd 服務**，讓它在開機時自動啟動。**強烈建議加入 `--gradio` 參數，以便進行遠端管理。**

### 1. 建立 systemd 服務檔案

```bash
sudo nano /etc/systemd/system/clawbody.service
```

貼入以下內容：

```ini
[Unit]
Description=ClawBody - OpenClaw AI Robot Body
After=network-online.target reachy-mini-daemon.service
Wants=network-online.target

[Service]
Type=simple
User=pollen
WorkingDirectory=/home/pollen/clawbody
EnvironmentFile=/home/pollen/clawbody/.env
ExecStart=/venvs/apps_venv/bin/clawbody --gradio
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 2. 設定開機自啟動
```bash
sudo systemctl daemon-reload
sudo systemctl enable clawbody.service
sudo systemctl start clawbody.service
```

### 3. 管理指令
| 動作 | 指令 |
|--------|---------|
| **啟動** | `sudo systemctl start clawbody` |
| **停止** | `sudo systemctl stop clawbody` |
| **重啟** | `sudo systemctl restart clawbody` |
| **查看狀態** | `sudo systemctl status clawbody` |
| **查看日誌** | `journalctl -u clawbody -f` |
| **關閉自動啟動** | `sudo systemctl disable clawbody` |

---

## 🛑 遠端停止服務 (不需要 SSH)

要使用以下功能，請確保 App 在啟動時帶有 `--gradio` 參數（詳見自動化章節）。

### 1. 網頁介面停止 (Gradio)
透過瀏覽器訪問 `http://reachy-mini.local:7860`：
- 點擊 **「🛑 Shutdown App」** 按鈕。
- 這會完全終止背景的 Python 進程。

### 2. 語音關機 (AI Voice Command)
您可以直接對著機器人說話來關閉它：
- **指令範例**：「嘿 Clawbody，請關閉服務」、「停止運行並休息吧」、「Goodbye」。
- AI 助手會先與您道別，然後在 3 秒後自動安全退出程式。

---

## ⚙️ 配置與遠端部署心得

### 連接到 Zeabur 或遠端 OpenClaw
在連接到部署在遠端（如 Zeabur）的 OpenClaw Gateway 時，請注意：

1. **協定與安全性**：遠端連線必須使用 `https://`。
2. **WebSocket (WSS) 挑戰**：ClawBody 與 Gateway 之間透過 WebSocket 通訊。請確保您的遠端部署環境已正確處理 `wss://` 流量。
3. **CORS 與 Token**：請確保 `.env` 中的 `OPENCLAW_TOKEN` 正確。

`.env` 範例：
```bash
OPENCLAW_GATEWAY_URL=https://your-openclaw-on-zeabur.zeabur.app
OPENCLAW_TOKEN=your-secure-token
```

---

## ✨ 核心特色

- **👁️ 智慧眼神接觸**：25Hz 即時人臉追蹤 (MediaPipe/YOLO)。
- **🎭 表現力手勢**：自動化手勢與語音輸出同步。
- **🧠 OpenClaw 深度整合**：透過實體化身完整調用 AI 工具。
- **💃 情感引擎**：動態發現並播放預錄行為。
- **🎤 低延遲語音**：由 OpenAI Realtime API 驅動。
- **🖥️ 模擬器優先**：完整支援 MuJoCo 模擬開發。

---

## 🛠️ 機器人能力 (AI 可調用)

| 能力 | 技術細節 |
|------------|-------------------|
| **自然手勢** | 根據逐字稿變動量同步觸發動作 |
| **情感註冊** | 自動掃描並註冊 daemon 中預錄的表情 |
| **人臉追蹤** | 使用 25Hz 視覺數據進行 PID 控制的頭部運動 |
| **視覺描述** | 捕捉畫面並利用 GPT-4o-mini 進行場景理解 |

---

## 📄 授權資訊

本專案採用 Apache 2.0 授權。

## 🙏 致謝

特別感謝 [Pollen Robotics](https://www.pollen-robotics.com/)、[OpenClaw](https://github.com/openclaw/openclaw) 與 [OpenAI](https://openai.com/) 的技術支持。
