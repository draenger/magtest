class Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens

    def __str__(self):
        return f"Usage(prompt_tokens={self.prompt_tokens}, completion_tokens={self.completion_tokens})"
