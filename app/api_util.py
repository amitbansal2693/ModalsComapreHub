import datetime
import time

import openai
import pytz


def get_current_time():
    """Return current time in Europe/Riga timezone"""
    return datetime.datetime.now(pytz.timezone("Europe/Riga"))


def escape_special_chars(text: str) -> str:
    """Escape special characters in text"""
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
    )


class APIUtil:
    class BadRequest(Exception):
        pass

    class OpenAIError(Exception):
        def __init__(self, message, error_type=None):
            super().__init__(message)
            self.error_type = error_type

    def __init__(self, api_key, restart_sequence="|UR|", stop_sequence="|SP|"):
        self.api_key = api_key
        openai.api_key = api_key
        self.stop_sequence = stop_sequence
        self.restart_sequence = restart_sequence

    # ---------------- Retry Wrapper ----------------
    def _retry_call(self, func, *args, max_tries=3, initial_backoff=1, **kwargs):
        """Generic retry wrapper for OpenAI API calls"""
        RETRY_EXCEPTIONS = (
            openai.error.APIError,
            openai.error.Timeout,
            openai.error.APIConnectionError,
            openai.error.ServiceUnavailableError,
        )
        tries, backoff = 0, initial_backoff
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, RETRY_EXCEPTIONS) and tries < max_tries:
                    time.sleep(backoff)
                    backoff *= 2
                    tries += 1
                else:
                    raise self.OpenAIError(
                        f"OpenAI API Error: {str(e)}", error_type=type(e).__name__
                    ) from e

    # ---------------- Moderation ----------------
    def get_moderation(self, user_message: str):
        """Check moderation flags for a user message"""
        try:
            moderation = self._retry_call(
                openai.Moderation.create, input=escape_special_chars(user_message)
            )
            moderation_result = moderation["results"][0]
            flagged_categories = [
                category
                for category, value in moderation_result["categories"].items()
                if value
            ]
            return {
                "flagged": moderation_result["flagged"],
                "flagged_categories": flagged_categories,
            }
        except Exception as e:
            raise self.OpenAIError(str(e)) from e

    # ---------------- Models ----------------
    def get_models(self):
        """Return models available to the API key"""
        try:
            return self._retry_call(openai.Model.list)
        except Exception as e:
            raise self.OpenAIError(str(e)) from e

    # ---------------- Main: AI Response ----------------
    def get_ai_response(self, model_config_dict, prompt, messages):
        """
        Generate AI response for given model config.
        For gpt-5 models → only send minimal params.
        """
        self._validate_model_config(model_config_dict)

        submit_messages = [{"role": "system", "content": prompt}] + self._messages_to_oai_messages(messages)

        try:
            if model_config_dict["model"].startswith("gpt-5"):
                # ✅ Ultra simple call (no stop, no temperature, etc.)
                params = {
                    "model": model_config_dict["model"],
                    "messages": submit_messages,
                    "max_completion_tokens": model_config_dict["max_tokens"],
                }
            else:
                # ✅ Normal GPT-3.5 / GPT-4 / GPT-4o call with tuning params
                params = {
                    "model": model_config_dict["model"],
                    "messages": submit_messages,
                    "temperature": model_config_dict["temperature"],
                    "max_tokens": model_config_dict["max_tokens"],
                    "top_p": model_config_dict["top_p"],
                    "frequency_penalty": model_config_dict["frequency_penalty"],
                    "presence_penalty": model_config_dict["presence_penalty"],
                    "stop": [self.stop_sequence],
                }

            response = self._retry_call(openai.ChatCompletion.create, **params)

            bot_message = response["choices"][0]["message"]["content"].strip()
            usage = response.get("usage", {})
            total_tokens = usage.get("total_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            new_messages = messages + [
                {"role": "assistant", "message": bot_message, "created_date": get_current_time()}
            ]

            return {
                "messages": new_messages,
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        except Exception as e:
            raise self.OpenAIError(str(e)) from e

    # ---------------- Helpers ----------------
    def _validate_model_config(self, model_config_dict):
        required_fields = [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        ]
        for field in required_fields:
            if field not in model_config_dict:
                raise self.BadRequest(
                    f"Bad Request: model_config_dict missing {field}"
                )
        return True

    def _messages_to_oai_messages(self, messages):
        """Convert internal messages format into OpenAI-compatible messages"""
        oai_messages = []
        for message in messages:
            oai_messages.append(
                {"role": message["role"], "content": escape_special_chars(message["message"])}
            )
        return oai_messages
