---
title: ClawBody
emoji: 🦞
colorFrom: red
colorTo: purple
sdk: static
pinned: false
short_description: OpenClaw AI with robot body and face tracking
tags:
 - reachy_mini
 - reachy_mini_python_app
 - openclaw
 - clawson
 - embodied-ai
 - ai-assistant
 - voice-assistant
 - robotics
 - openai-realtime
 - conversational-ai
 - physical-ai
 - robot-body
 - speech-to-speech
 - multimodal
 - vision
 - expressive-robot
 - simulation
 - mujoco
 - face-tracking
 - face-detection
 - eye-contact
 - human-robot-interaction
---

# 🦞🤖 ClawBody

**Give your OpenClaw AI agent an expressive, embodied physical body!**

[繁體中文版 (Traditional Chinese)](README_zh-TW.md)

ClawBody bridges the gap between high-level AI intelligence (OpenClaw) and low-level robotic control (Reachy Mini). By leveraging OpenAI's Realtime API, it creates an ultra-low latency, speech-to-speech interaction loop where your AI assistant, Clawson, can see, hear, and express emotions physically.

![Reachy Mini Dance](https://huggingface.co/spaces/pollen-robotics/reachy_mini_conversation_app/resolve/main/docs/assets/reachy_mini_dance.gif)

---

## 🚀 Key Improvements (Feb 2026)

- **Natural Embodiment**: Introduced **Natural Turn-Level Gestures** and **Speech-Synced Body Sway**.
- **Dynamic Capability Discovery**: A new **Capability Registry** automatically scans for recorded expressions and dances.
- **Context-Aware Triggers**: Added support for **Cue Word Gestures** from live transcripts.
- **Enhanced Perception**: Optimized MediaPipe tracking and resolved OpenClaw Gateway CORS issues.

---

## 🚀 Getting Started

### 💡 Critical Usage Note: Virtual Environment
If you installed ClawBody within a virtual environment, you **must** use the environment's python/bin to run it.

**For local/simulator setup:**
```bash
source .venv/bin/activate
clawbody --gradio
```

**For physical robot setup:**
```bash
/venvs/apps_venv/bin/clawbody
```

---

### Option A: Installation for Simulator

```bash
git clone https://github.com/dAAAb/clawbody
cd clawbody
python -m venv .venv && source .venv/bin/activate
pip install -e ".[mediapipe_vision]"
pip install "reachy-mini[mujoco]"

# Configure your .env
cp .env.example .env

# Terminal 1: Run Simulator
reachy-mini-daemon --sim

# Terminal 2: Start ClawBody
source .venv/bin/activate
clawbody --gradio
```

---

### Option B: Installation on Physical Robot

The Reachy Mini robot comes with a pre-configured application environment at `/venvs/apps_venv/`.

```bash
# Connect to your robot
ssh pollen@reachy-mini.local

# Clone and install into the robot's app environment
git clone https://github.com/dAAAb/clawbody
cd clawbody
/venvs/apps_venv/bin/pip install -e .

# Run directly on hardware using the app environment's bin
/venvs/apps_venv/bin/clawbody
```

> **Note**: For persistent background operation, the OpenClaw Bridge can be managed via the `reachy-mini-daemon`'s app registry.

---

## ⚙️ Configuration & Remote Deployment

### Connecting to Zeabur / Remote OpenClaw
When connecting to a remote OpenClaw instance (e.g., hosted on Zeabur), pay close attention to your `OPENCLAW_GATEWAY_URL`:

1. **Protocol Matters**: Use `https://` for remote instances.
2. **WebSocket (WSS)**: ClawBody communicates with the gateway via WebSockets. Ensure your remote deployment correctly handles `wss://` traffic.
3. **CORS/Auth**: Ensure your `OPENCLAW_TOKEN` is correctly set and that the remote gateway allows connections from your local machine.

Example `.env`:
```bash
OPENCLAW_GATEWAY_URL=https://your-openclaw-on-zeabur.zeabur.app
OPENCLAW_TOKEN=your-secure-token
```

---

## ✨ Features

- **👁️ Intelligent Eye Contact**: Real-time face tracking (MediaPipe/YOLO) at 25Hz.
- **🎭 Expressive Gestures**: Automatic gestures synced to voice output.
- **🧠 OpenClaw Integration**: Full tool-calling capabilities through a physical persona.
- **💃 Emotion Engine**: Dynamic discovery and playback of pre-recorded behaviors.
- **🎤 Low-Latency Voice**: Powered by OpenAI Realtime API.
- **🖥️ Simulator-First**: Full support for MuJoCo simulation.

---

## 🛠️ Robot Capabilities (AI-Accessible)

| Capability | Technical Details |
|------------|-------------------|
| **Natural Gestures** | Turn-level triggers synced to transcript deltas |
| **Emotion Registry** | Dynamic discovery of daemon-recorded expressions |
| **Face Tracking** | PID-controlled head movement using 25Hz vision data |
| **Vision Description** | Captures frames and uses GPT-4o-mini for scene understanding |

---

## 📄 License

This project is licensed under the Apache 2.0 License.

## 🙏 Acknowledgments

Built with ❤️ by the community, leveraging works from [Pollen Robotics](https://www.pollen-robotics.com/), [OpenClaw](https://github.com/openclaw/openclaw), and [OpenAI](https://openai.com/).
