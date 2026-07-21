
    def choose_function(
        self,
        prompt: str,
        functions: list[FunctionDefinition],
    ) -> FunctionDefinition:
        ...

    def extract_parameters(
        self,
        prompt: str,
        function: FunctionDefinition,
    ) -> dict:
        ...

    # ==========================================================
    # JSON generation
    # ==========================================================

    def generate_json(
        self,
        prompt: str,
        function: FunctionDefinition,
        parameters: dict,
    ) -> dict:
        ...

    # ==========================================================
    # Public API
    # ==========================================================

    def decode(
        self,
        prompt: str,
        functions: list[FunctionDefinition],
    ) -> dict:
        ...


def next_token(
        self, input_ids: list[int], generated_tokens: list[int],
        function: FunctionDefinition | None,
    ) -> int:
        logits = self.get_logits(input_ids)

        allowed = self.allowed_tokens(generated_tokens, function)
        filtered_logits = self.filter_logits(logits, allowed)

        return self.best_token(filtered_logits)