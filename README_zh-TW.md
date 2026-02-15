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
/venvs/apps_venv/bin/clawbody
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

# 使用該環境的 bin 直接在硬體上執行
/venvs/apps_venv/bin/clawbody
```

---

## 🤖 自動化與背景執行 (Automation)

在實體 Reachy Mini 上，您可以將 ClawBody 註冊為受管理的服務，使用 `reachy-mini-daemon` 工具使其在機器人開機時自動啟動。

### 1. 註冊應用程式
在機器人終端機執行以下指令（使用絕對路徑）。這會告知 daemon 程式碼的位置以及如何讀取 `.env`：

```bash
/venvs/apps_venv/bin/reachy-mini-daemon app register clawbody --path /home/pollen/clawbody
```

### 2. 設定開機自啟動
```bash
/venvs/apps_venv/bin/reachy-mini-daemon app enable clawbody
```

### 3. 管理指令
| 動作 | 指令 |
|--------|---------|
| **啟動** | `/venvs/apps_venv/bin/reachy-mini-daemon app start clawbody` |
| **停止** | `/venvs/apps_venv/bin/reachy-mini-daemon app stop clawbody` |
| **查看狀態** | `/venvs/apps_venv/bin/reachy-mini-daemon app list` |
| **查看日誌** | `/venvs/apps_venv/bin/reachy-mini-daemon app logs clawbody` |

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
