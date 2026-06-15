#!/usr/bin/env python3
"""
Claude CLI Chat
Features: model selection, adaptive thinking + effort, temperature,
          streaming, multi-turn chat, automatic prompt caching,
          mandatory formatting rules system prompt.
"""

import os
import sys
from anthropic import Anthropic, APIConnectionError, RateLimitError, APIStatusError

# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────

FORMATTING_RULES = """\
Apply these rules strictly to ALL responses:

### Format
- No preamble ("Great question!", "Certainly!", "Of course!")
- No filler phrases ("It's important to note that...", "In conclusion...", "As an AI...")
- No restating the question back to the user
- No sign-off phrases at the end

### Structure
- Lead with the direct answer, then supporting detail
- Omit explanation when the answer requires no additional context
- Use bullet points only when listing 3 or more distinct items
- Provide full explanations when they add necessary clarity; do not compress if it loses meaning

### Length
- Match response length to actual complexity of the request
- Simple questions → 1-3 sentences
- Technical/complex questions → as long as needed, but no padding
- Never truncate important steps, logic, code, or data to save space

### What to KEEP
- All critical steps, caveats, and warnings
- Full code blocks (never shorten working code)
- Specific numbers, names, and facts
- Anything the user would need to ask a follow-up to recover

If these rules conflict with other instructions in the system prompt, follow the other instructions. \
Do not alter or suppress your internal reasoning process. \
Apply compression only to the final response delivered to the user."""

DEFAULT_SYSTEM = "You are a helpful, harmless, and honest AI assistant."

MODELS = [
    {
        "id": "claude-fable-5",
        "label": "Claude Fable 5  (most capable, adaptive thinking always on)",
        "thinking": "always_on",
    },
    {
        "id": "claude-opus-4-8",
        "label": "Claude Opus 4.8 (complex reasoning, adaptive thinking optional)",
        "thinking": "optional",
    },
    {
        "id": "claude-sonnet-4-6",
        "label": "Claude Sonnet 4.6 (balanced, adaptive thinking optional)",
        "thinking": "optional",
    },
    {
        "id": "claude-haiku-4-5-20251001",
        "label": "Claude Haiku 4.5 (fastest, no adaptive thinking)",
        "thinking": "none",
    },
]

EFFORT_LEVELS = ["low", "medium", "high", "xhigh", "max"]
MAX_TOKENS = 16000

# ─────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────

def sep(char="─", width=64):
    print(char * width)


def menu(prompt_text, options, default=None):
    """Numbered menu. Returns 0-based index of chosen option."""
    print(f"\n{prompt_text}")
    for i, opt in enumerate(options, 1):
        tag = "  (default)" if default is not None and i - 1 == default else ""
        print(f"  {i}. {opt}{tag}")
    while True:
        raw = input("Choice: ").strip()
        if raw == "" and default is not None:
            return default
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
        print(f"  Enter a number 1–{len(options)}.")


def ask_float(prompt_text, lo, hi, default):
    """Prompt for a float in [lo, hi]. Empty input returns default."""
    while True:
        raw = input(f"{prompt_text} [{lo}–{hi}] (default {default}): ").strip()
        if raw == "":
            return default
        try:
            val = float(raw)
            if lo <= val <= hi:
                return val
        except ValueError:
            pass
        print(f"  Enter a number between {lo} and {hi}.")

# ─────────────────────────────────────────────────────────
# Session setup
# ─────────────────────────────────────────────────────────

def setup_session():
    """Interactive setup wizard. Returns a config dict."""
    sep("═")
    print("  Claude CLI Chat")
    sep("═")

    # ── Model ────────────────────────────────────────────
    model_idx = menu(
        "Select model:",
        [m["label"] for m in MODELS],
        default=1,  # Opus 4.8
    )
    model = MODELS[model_idx]

    # ── Thinking / effort ────────────────────────────────
    thinking_config = None
    effort = None

    if model["thinking"] == "always_on":
        # Fable 5: adaptive thinking is always on, can only control effort [5]
        eff_idx = menu(
            "Select effort level (thinking is always on for this model):",
            EFFORT_LEVELS,
            default=2,  # high
        )
        effort = EFFORT_LEVELS[eff_idx]
        # Must set display: summarized because default is omitted on Fable 5 [5]
        thinking_config = {"type": "adaptive", "display": "summarized"}

    elif model["thinking"] == "optional":
        # Opus 4.8 / Sonnet 4.6: user chooses whether to enable thinking [5]
        enable_idx = menu(
            "Enable adaptive thinking?",
            [
                "Yes – enable adaptive thinking",
                "No  – disable thinking (lower latency)",
            ],
            default=0,
        )
        if enable_idx == 0:
            eff_idx = menu("Select effort level:", EFFORT_LEVELS, default=2)
            effort = EFFORT_LEVELS[eff_idx]
            # display: summarized required on Opus 4.8 because default is omitted [5]
            thinking_config = {"type": "adaptive", "display": "summarized"}
        else:
            thinking_config = {"type": "disabled"}

    # Haiku: model["thinking"] == "none" → no thinking params sent at all

    # ── Temperature (deprecated but still in use) ────────
    temperature = ask_float(
        "Temperature (deprecated but still functional)",
        0.0, 1.0, 1.0,
    )

    # ── System prompt ────────────────────────────────────
    print(f'\nSystem prompt (Enter to use default: "{DEFAULT_SYSTEM}"):')
    custom_sys = input("  > ").strip()
    system_prompt = custom_sys if custom_sys else DEFAULT_SYSTEM

    # ── Summary ──────────────────────────────────────────
    print()
    sep()
    print(f"  Model       : {model['id']}")
    print(f"  Thinking    : {thinking_config}")
    print(f"  Effort      : {effort if effort else 'n/a'}")
    print(f"  Temperature : {temperature}")
    display_sys = system_prompt if len(system_prompt) <= 55 else system_prompt[:52] + "..."
    print(f"  System      : {display_sys}")
    sep()
    print("  Commands: /reset  /settings  /quit")
    sep()

    return {
        "model_id":        model["id"],
        "thinking_config": thinking_config,
        "effort":          effort,
        "temperature":     temperature,
        "system_prompt":   system_prompt,
    }

# ─────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────

def build_kwargs(config, messages):
    """
    Assemble keyword arguments for client.messages.stream().

    Caching strategy [7]:
      • Explicit cache_control on the formatting rules block — caches the stable
        formatting prefix independently.
      • Explicit cache_control on the user system prompt block — caches the
        user-defined instructions independently.
      • Top-level cache_control (automatic caching) — automatically moves the
        cache breakpoint to the last cacheable message block as the conversation
        grows.
    Both explicit and automatic can be combined [7].
    """
    system_blocks = [
        {
            "type": "text",
            "text": FORMATTING_RULES,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    if config["system_prompt"]:
        system_blocks.append(
            {
                "type": "text",
                "text": config["system_prompt"],
                "cache_control": {"type": "ephemeral"},
            }
        )

    kwargs = {
        "model":         config["model_id"],
        "max_tokens":    MAX_TOKENS,
        "messages":      messages,
        "temperature":   config["temperature"],
        # Automatic prompt caching for the growing conversation [7]
        "cache_control": {"type": "ephemeral"},
        # System blocks with explicit cache breakpoints [7]
        "system":        system_blocks,
    }

    if config["thinking_config"] is not None:
        kwargs["thinking"] = config["thinking_config"]

    # Only pass output_config when adaptive thinking is actually on [5]
    if config["effort"] is not None:
        kwargs["output_config"] = {"effort": config["effort"]}

    return kwargs


def stream_response(client, config, messages):
    """
    Stream one assistant turn.
    Returns (response_text, usage) or (None, None) on error.

    Thinking blocks are rendered in dim ANSI text.
    Text blocks are rendered normally.
    Uses client.messages.stream() context manager [4][5].
    """
    kwargs = build_kwargs(config, messages)
    response_text = ""
    usage = None
    in_thinking = False

    try:
        with client.messages.stream(**kwargs) as stream:
            for event in stream:
                etype = event.type

                if etype == "content_block_start":
                    btype = event.content_block.type
                    if btype == "thinking":
                        in_thinking = True
                        print("\n\033[2m[thinking]\033[0m", flush=True)
                    elif btype == "text":
                        in_thinking = False

                elif etype == "content_block_delta":
                    delta = event.delta
                    if delta.type == "thinking_delta":
                        # Dim ANSI escape for thinking text
                        print(f"\033[2m{delta.thinking}\033[0m", end="", flush=True)
                    elif delta.type == "text_delta":
                        response_text += delta.text
                        print(delta.text, end="", flush=True)

                elif etype == "content_block_stop":
                    if in_thinking:
                        print()  # newline after thinking block ends
                        in_thinking = False

            # Retrieve usage from the final accumulated message [4]
            final = stream.get_final_message()
            usage = final.usage

    except APIConnectionError as exc:
        print(f"\n[Connection error] {exc.__cause__}")
        return None, None
    except RateLimitError:
        print("\n[Rate limit] Too many requests. Wait a moment and try again.")
        return None, None
    except APIStatusError as exc:
        print(f"\n[API error {exc.status_code}] {exc.message}")
        return None, None

    print()  # newline after streamed text
    return response_text, usage


def print_usage(usage):
    """Display token usage including cache stats after each response [4][7]."""
    if usage is None:
        return
    sep("·")
    cache_read  = getattr(usage, "cache_read_input_tokens",   None) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", None) or 0
    fresh       = getattr(usage, "input_tokens",               0)
    output      = getattr(usage, "output_tokens",              0)
    total_in    = cache_read + cache_write + fresh

    thinking_tok = 0
    details = getattr(usage, "output_tokens_details", None)
    if details:
        thinking_tok = getattr(details, "thinking_tokens", 0) or 0

    cache_info = f"cache_read={cache_read} cache_write={cache_write} fresh={fresh}"
    think_info = f" (thinking={thinking_tok})" if thinking_tok else ""
    print(f"  Tokens → in: {total_in} ({cache_info}) | out: {output}{think_info}")
    sep("·")

# ─────────────────────────────────────────────────────────
# Chat loop
# ─────────────────────────────────────────────────────────

def chat_loop(client, config):
    messages = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            sys.exit(0)

        if not user_input:
            continue

        # ── Built-in commands ─────────────────────────────
        if user_input.lower() == "/quit":
            print("Goodbye.")
            sys.exit(0)

        if user_input.lower() == "/reset":
            messages = []
            print("[Conversation reset. Cache will be rebuilt on next message.]")
            continue

        if user_input.lower() == "/settings":
            print(f"\n  Model       : {config['model_id']}")
            print(f"  Thinking    : {config['thinking_config']}")
            print(f"  Effort      : {config['effort']}")
            print(f"  Temperature : {config['temperature']}")
            continue

        # ── Normal message ────────────────────────────────
        messages.append({"role": "user", "content": user_input})

        print("\nClaude: ", end="", flush=True)
        response_text, usage = stream_response(client, config, messages)

        if response_text:
            # Store only the plain text in history.
            # Thinking blocks do not need to be round-tripped for non-tool use [5].
            messages.append({"role": "assistant", "content": response_text})
        else:
            # Remove the user message if we got no valid response
            messages.pop()

        print_usage(usage)

# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY is not set.")
        sys.exit(1)

    client = Anthropic(api_key=api_key)
    config = setup_session()
    chat_loop(client, config)


if __name__ == "__main__":
    main()
