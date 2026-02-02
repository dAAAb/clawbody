# ClawBody - Robot Body for OpenClaw

Give your OpenClaw agent (Clawson) a physical robot body with Reachy Mini.

## Description

ClawBody embodies your OpenClaw AI assistant in a Reachy Mini robot, enabling it to:

- **Hear**: Listen to voice commands via the robot's microphone
- **See**: View the world through the robot's camera
- **Speak**: Respond with natural voice through the robot's speaker
- **Move**: Express emotions through expressive head movements and dances

Using a hybrid architecture with OpenAI Realtime API for voice I/O and OpenClaw for intelligence, the robot responds with sub-second latency for natural conversation.

## Architecture

```
You speak → Reachy Mini 🎤
                ↓
       OpenAI Realtime API
    (speech recognition + TTS)
                ↓
        OpenClaw Gateway
      (Clawson's brain 🦞)
                ↓
   Robot speaks & moves 🤖💃
```

## Requirements

### Hardware
- [Reachy Mini](https://github.com/pollen-robotics/reachy_mini) robot (Wireless or Lite)

### Software
- Python 3.11+
- OpenAI API key with Realtime API access
- OpenClaw gateway running on your network

## Installation

```bash
# Clone from HuggingFace
git clone https://huggingface.co/spaces/tomrikert/clawbody
cd clawbody
pip install -e .
```

## Configuration

Create a `.env` file:

```bash
OPENAI_API_KEY=sk-your-key-here
OPENCLAW_GATEWAY_URL=http://your-host-ip:18789
OPENCLAW_TOKEN=your-gateway-token
```

## Usage

```bash
# Run ClawBody
clawbody

# With debug logging
clawbody --debug

# With Gradio web UI
clawbody --gradio
```

## Features

### Real-time Voice Conversation
Ultra-low latency voice interaction using OpenAI's Realtime API.

### OpenClaw Intelligence
Full Clawson capabilities - tools, memory, personality - through the OpenClaw gateway.

### Expressive Movements
- Audio-driven head wobble while speaking
- Emotion expressions (happy, curious, thinking, excited)
- Dance animations
- Natural head movements

### Vision
Ask Clawson to describe what it sees through the robot's camera.

## Links

- [HuggingFace Space](https://huggingface.co/spaces/tomrikert/clawbody)
- [Reachy Mini SDK](https://github.com/pollen-robotics/reachy_mini)
- [OpenClaw Documentation](https://docs.openclaw.ai)

## Author

Tom (tomrikert)

## License

Apache 2.0
