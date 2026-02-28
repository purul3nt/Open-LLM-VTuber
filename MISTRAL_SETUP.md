## Using Mistral as the main LLM

This project is already configured to use **Mistral** as the default LLM for the basic agent in `conf.yaml`.

### 1. Set your Mistral API key

1. Open `conf.yaml` in the project root.
2. Find the `mistral_llm` section under `character_config.agent_config.llm_configs`:

   ```yaml
   mistral_llm:
     llm_api_key: 'Your Mistral API key'
     model: 'pixtral-large-latest'
     temperature: 1.0
   ```

3. Replace `'Your Mistral API key'` with your real key **or** leave it as-is and configure your key via environment variable, depending on how you prefer to manage secrets.

> Tip: for local dev, editing this file is usually the simplest. For production, prefer environment variables or a separate config.

### 2. Confirm the basic agent uses Mistral

In `conf.yaml`, the `basic_memory_agent` is set to use `mistral_llm`:

```yaml
agent_config:
  conversation_agent_choice: 'basic_memory_agent'
  agent_settings:
    basic_memory_agent:
      llm_provider: 'mistral_llm'
      faster_first_response: True
      segment_method: 'pysbd'
```

This means any normal chat in the UI will go through your configured Mistral model.

### 3. TTS backend

The default config already uses **Edge TTS** as a free TTS backend:

```yaml
tts_config:
  tts_model: 'edge_tts'
  edge_tts:
    voice: 'en-US-AvaMultilingualNeural'
```

You can change the voice later with `edge-tts --list-voices`.

### 4. Running the server

From the `Open-LLM-VTuber` folder:

```bash
uv run run_server.py --verbose
```

On first run, dependencies will be resolved and models downloaded as described in the main `README.md`. Once the server is up, open the web UI and verify that:

- The VTuber responds using your Mistral model.
- Speech is synthesized via the Edge TTS voice you configured.

