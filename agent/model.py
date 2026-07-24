"""Minimal OpenAI-compatible chat client.
Standard library only; the endpoint and token always come from the
validator-managed proxy configuration passed into agent.solve().
"""

import json
import re
import time
import urllib.error
import urllib.request

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# The server rejects a request whose prompt + requested max_tokens exceeds its
# window (vLLM --max-model-len). We only ever see the prompt in CHARACTERS, so we
# carry a chars-per-token ratio and keep it honest against the token counts the
# server reports back. It varies with content density (2.6 on dense code, 5.1 on
# prose), so a fixed ratio is not safe to assume; this is the value we start from
# before the first response lands, chosen below the densest ratio we have measured.
_INITIAL_CHARS_PER_TOKEN = 2.5
_CONTEXT_LENGTH_MARKER = "maximum context length"
_PROMPT_TOKENS_RE = re.compile(r"prompt contains at least (\d+) input tokens")


class ModelQueryError(RuntimeError):
    pass


class ContextLengthError(ModelQueryError):
    """The prompt overflowed the server's window. Recoverable: compact and retry."""


class ChatModel:
    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        auth_token: str,
        max_completion_tokens: int = 0,
        request_timeout: float = 180.0,
        max_attempts: int = 5,
    ) -> None:
        self.model_name = model_name
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.auth_token = auth_token
        self.max_completion_tokens = int(max_completion_tokens or 0)
        self.request_timeout = request_timeout
        self.max_attempts = max(1, int(max_attempts))
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        # Per-call completion size + whether the reply hit the token cap. The loop
        # uses these to abort full-file rewrite spirals that burn the wall clock
        # (duel-101923 r14/r26: two consecutive 4096-token replies ate ~450s).
        self.last_completion_tokens = 0
        self.last_finish_reason = ""
        # Live chars-per-token for THIS run's content, re-measured on every response
        # (and corrected from the error body when the server rejects an overflow).
        # The caller sizes its compaction budget off this.
        self.chars_per_token = _INITIAL_CHARS_PER_TOKEN
        self._sent_prompt_chars = 0

    def last_reply_hit_token_cap(self) -> bool:
        """True when the latest reply was cut off by max_tokens."""
        if self.last_finish_reason == "length":
            return True
        cap = self.max_completion_tokens
        return cap > 0 and self.last_completion_tokens >= max(1, cap - 4)

    def query(self, messages: list) -> str:
        """Send the conversation and return the assistant message text."""
        payload = {"model": self.model_name, "messages": messages}
        if self.max_completion_tokens > 0:
            payload["max_tokens"] = self.max_completion_tokens
        body = json.dumps(payload).encode("utf-8")
        self._sent_prompt_chars = messages_chars(messages)
        last_error = "unknown error"
        for attempt in range(1, self.max_attempts + 1):
            try:
                raw = self._post(body)
            except urllib.error.HTTPError as exc:
                detail = _read_error_body(exc)
                last_error = f"HTTP {exc.code}: {detail[:300]}"
                if _CONTEXT_LENGTH_MARKER in detail.lower():
                    self._recalibrate_from_overflow(detail)
                    raise ContextLengthError(f"prompt exceeds the model window: {last_error}") from exc
                if exc.code not in _RETRYABLE_STATUS:
                    raise ModelQueryError(f"model request was rejected: {last_error}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                self.calls += 1
                return self._extract_content(raw)
            if attempt < self.max_attempts:
                time.sleep(min(20.0, 1.5 ** attempt))
        raise ModelQueryError(f"model request failed after {self.max_attempts} attempts: {last_error}")

    def _post(self, body: bytes) -> str:
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.auth_token}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
            return response.read().decode("utf-8", errors="replace")

    def _extract_content(self, raw: str) -> str:
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise ModelQueryError(f"model returned invalid JSON: {raw[:300]}") from exc
        usage = payload.get("usage") if isinstance(payload, dict) else None
        completion_tokens = 0
        if isinstance(usage, dict):
            prompt_tokens = _as_int(usage.get("prompt_tokens"))
            completion_tokens = _as_int(usage.get("completion_tokens"))
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self._observe_ratio(prompt_tokens)
        self.last_completion_tokens = completion_tokens
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ModelQueryError(f"model response has no choices: {raw[:300]}")
        choice0 = choices[0] if isinstance(choices[0], dict) else {}
        self.last_finish_reason = str(choice0.get("finish_reason") or "")
        message = choice0.get("message") if isinstance(choice0, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            content = "".join(
                str(part.get("text") or "") for part in content if isinstance(part, dict)
            )
        if not isinstance(content, str):
            raise ModelQueryError(f"model response has no text content: {raw[:300]}")
        return content


    def _observe_ratio(self, prompt_tokens: int) -> None:
        """Re-measure chars-per-token from what the server actually counted."""
        if prompt_tokens > 0 and self._sent_prompt_chars > 0:
            self.chars_per_token = self._sent_prompt_chars / prompt_tokens

    def _recalibrate_from_overflow(self, detail: str) -> None:
        """We overflowed, so our ratio was too optimistic. The server names the real
        input-token count in the error; trust it. Absent that, back the ratio off
        enough that the retry is meaningfully smaller rather than failing again."""
        match = _PROMPT_TOKENS_RE.search(detail)
        if match:
            self._observe_ratio(int(match.group(1)))
        else:
            self.chars_per_token *= 0.85


def messages_chars(messages: list) -> int:
    return sum(len(str(item.get("role", ""))) + len(str(item.get("content", ""))) for item in messages)


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return str(exc)


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
