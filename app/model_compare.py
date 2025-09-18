import logging

import api_util as api
import streamlit as st

st.set_page_config(layout="wide")

# ✅ Only allow 5 models
ALLOWED_MODELS = [
    ("gpt-4", 8000),
    ("gpt-3.5-turbo", 4096),
    ("gpt-5-nano-2025-08-07", 4000),
    ("gpt-4o-mini", 8000),
    ("gpt-3.5-turbo-16k", 16000),
]

def handler_verify_key():
    """Initialize allowed models"""
    oai_api_key = st.session_state.open_ai_key_input
    _ = api.APIUtil(api_key=oai_api_key)
    try:
        st.session_state.openai_model_params = ALLOWED_MODELS
        st.session_state.openai_models = [m for m, _ in ALLOWED_MODELS]
        st.session_state.openai_models_str = ", ".join(st.session_state.openai_models)

        st.session_state.chat_histories = {m: [] for m in st.session_state.openai_models}
        st.session_state.total_tokens = {m: 0 for m in st.session_state.openai_models}
        st.session_state.prompt_tokens = {m: 0 for m in st.session_state.openai_models}
        st.session_state.completion_tokens = {m: 0 for m in st.session_state.openai_models}
        st.session_state.conversation_cost = {m: 0 for m in st.session_state.openai_models}

        st.session_state.oai_api_key = oai_api_key
        st.session_state.test_disabled = False
    except Exception as e:
        logging.error(f"{e}")

def calculate_cost(model, tokens, prompt_tokens=0, completion_tokens=0):
    if model == "gpt-4":
        return (0.03 * prompt_tokens / 1000) + (0.06 * completion_tokens / 1000)
    elif model == "gpt-3.5-turbo":
        return 0.002 * tokens / 1000
    elif model == "gpt-5-nano-2025-08-07":
        return 0.02 * tokens / 1000
    return 0.001 * tokens / 1000

def handler_fetch_model_responses():
    model_config_template = {
        "max_tokens": st.session_state.model_max_tokens,
        "temperature": st.session_state.model_temperature,
        "top_p": st.session_state.model_top_p,
        "frequency_penalty": st.session_state.model_frequency_penalty,
        "presence_penalty": st.session_state.model_presence_penalty,
    }
    o = api.APIUtil(api_key=st.session_state.oai_api_key)
    init_prompt = st.session_state.init_prompt
    if not init_prompt:
        return

    for idx, m in enumerate(st.session_state.openai_models):
        progress_bar_container.progress(idx / len(st.session_state.openai_models), text=f"{m} running...")
        try:
            b_r = o.get_ai_response(
                model_config_dict={**model_config_template, "model": m},
                prompt=init_prompt,
                messages=st.session_state.chat_histories[m],
            )
            st.session_state.chat_histories[m] = b_r["messages"]
            st.session_state.total_tokens[m] = b_r["total_tokens"]
            st.session_state.prompt_tokens[m] = b_r["prompt_tokens"]
            st.session_state.completion_tokens[m] = b_r["completion_tokens"]
            st.session_state.conversation_cost[m] = calculate_cost(
                m, b_r["total_tokens"], b_r["prompt_tokens"], b_r["completion_tokens"]
            )
        except Exception as e:
            logging.error(f"Error with {m}: {e}")
            continue
    progress_bar_container.empty()

def handler_start_new_test():
    st.session_state.chat_histories = {m: [] for m in st.session_state.openai_models}
    st.session_state.total_tokens = {m: 0 for m in st.session_state.openai_models}
    st.session_state.conversation_cost = {m: 0 for m in st.session_state.openai_models}

def ui_sidebar():
    with st.sidebar:
        if "chat_histories" in st.session_state and any(st.session_state.chat_histories[m] for m in st.session_state.openai_models):
            st.button("Start a new Test", on_click=handler_start_new_test)
            st.write("---")
        st.text_area("Initial prompt", key="init_prompt", disabled=st.session_state.test_disabled)
        st.text_area("Follow up message", key="user_msg", disabled=st.session_state.test_disabled)
        st.number_input("Response Token Limit", key="model_max_tokens", value=300, step=50,
                        help="Limit tokens per response", disabled=st.session_state.test_disabled)
        st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, key="model_temperature",
                  help="Higher = more creative", disabled=st.session_state.test_disabled)
        st.slider("Top P", 0.0, 1.0, 1.0, 0.1, key="model_top_p",
                  help="Lower = more focused", disabled=st.session_state.test_disabled)
        st.slider("Frequency penalty", 0.0, 1.0, 0.0, 0.1, key="model_frequency_penalty",
                  help="Discourages repeats", disabled=st.session_state.test_disabled)
        st.slider("Presence penalty", 0.0, 1.0, 0.0, 0.1, key="model_presence_penalty",
                  help="Encourages new topics", disabled=st.session_state.test_disabled)
        st.button("Fetch AI Responses", on_click=handler_fetch_model_responses,
                  disabled=st.session_state.test_disabled)

def ui_introduction():
    st.text_input("Enter OpenAI API Key", key="open_ai_key_input", type="password",
                  on_change=handler_verify_key, placeholder="Paste your OpenAI API key here")

def ui_test_result():
    if "openai_models" in st.session_state:
        cols = st.columns(len(st.session_state.openai_models))
        for i, m in enumerate(st.session_state.openai_models):
            with cols[i]:
                st.write(f"### Conversation with {m}")
                st.write(f"Total tokens: {st.session_state.total_tokens[m]}")
                st.write(f"Prompt tokens: {st.session_state.prompt_tokens[m]}")
                st.write(f"Completion tokens: {st.session_state.completion_tokens[m]}")
                st.write(f"Total cost: ${st.session_state.conversation_cost[m]:.6f}")
                st.write("---")
                for msg in st.session_state.chat_histories[m]:
                    if msg["role"] == "user":
                        st.markdown(f"**User:** {msg['message']}")
                    else:
                        st.markdown(f"**Model:** {msg['message']}")

if "test_disabled" not in st.session_state:
    st.session_state.test_disabled = True

openai_key_container = st.container()
ui_sidebar()

st.title("OpenAI GPT Model Comparison Tool")

if "oai_api_key" not in st.session_state:
    st.write("👋 Paste your OpenAI API key to start.")
    ui_introduction()
else:
    st.write(f"Comparing: {st.session_state.openai_models_str}")
    st.write("---")
    progress_bar_container = st.empty()
    ui_test_result()
