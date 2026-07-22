import json
import string
from enum import Enum
from typing import Any
from .models import FunctionDefinition
from llm_sdk.llm_sdk import Small_LLM_Model


class DecoderState(Enum):
    START = 0
    NAME_KEY = 1
    NAME_COLON = 2
    NAME_VALUE = 3
    COMMA_1 = 4
    PARAMETERS_KEY = 5
    PARAMETERS_COLON = 6
    PARAMETERS_OPEN = 7
    PARAM_NAME = 8
    PARAM_COLON = 9
    PARAM_VALUE = 10
    PARAM_COMMA_OR_CLOSE = 11
    CLOSE_BRACE = 12
    END = 13


class Decoder:
    def __init__(self, model: Small_LLM_Model):
        self.model = model
        self.state = DecoderState.START

        vocab_path = model.get_path_to_vocab_file()
        print(f"VOCAB PATH: {vocab_path}")

        with open(vocab_path, "r") as file:
            self.token_to_id = json.load(file)

        self.id_to_token = {}

        for token, token_id in self.token_to_id.items():
            self.id_to_token[token_id] = token

    def token_id(self, token: str) -> int:
        return self.token_to_id[token]

    def token(self, token_id: int) -> str:
        return self.id_to_token[token_id]

    def best_token(self, logits: list[float]) -> int:
        best_score = logits[0]
        best_token_id = 0

        for token_id, score in enumerate(logits):
            if score > best_score:
                best_score = score
                best_token_id = token_id

        return best_token_id

    def filter_logits(
            self, logits: list[float], allowed_tokens: set[int]
    ) -> list[float]:
        filtered = logits.copy()

        for token_id in range(len(filtered)):
            if token_id not in allowed_tokens:
                filtered[token_id] = float("-inf")
        return filtered

    def get_logits(self, input_ids: list[int]) -> list[float]:
        return self.model.get_logits_from_input_ids(input_ids)

    def append_token(
            self, tokens: list[int], token_id: int
    ) -> None:
        tokens.append(token_id)

    def decode_tokens(self, token_ids: list[int]) -> str:
        return self.model.decode(token_ids)

    def allowed_start(self) -> set[int]:
        allowed = set()

        for token_id in self.id_to_token:
            token = self.decode_tokens([token_id])

            if "{" in token:
                allowed.add(token_id)

        return allowed

    def allowed_name_key(self, prefix: str) -> set[int]:
        allowed = set()
        target = '"name"'

        for token_id in self.id_to_token:
            token = self.decode_tokens([token_id])
            candidate = prefix + token

            if target.startswith(candidate) or candidate == target:
                allowed.add(token_id)

        return allowed

    def allowed_name_value(
        self,
        prefix: str,
        function: FunctionDefinition | None,
        available_functions: list[FunctionDefinition] | None,
    ) -> set[int]:

        allowed = set()

        functions = available_functions or (
            [function] if function else []
        )

        for fn in functions:

            target = f'"{fn.name}"'

            for token_id in self.id_to_token:

                token = self.decode_tokens([token_id])
                candidate = prefix + token

                if target.startswith(candidate) or candidate == target:
                    allowed.add(token_id)

        return allowed

    def allowed_param_name(
        self,
        prefix: str,
        function: FunctionDefinition,
    ) -> set[int]:

        allowed = set()

        for name in function.parameters:

            target = f'"{name}"'

            for token_id in self.id_to_token:

                token = self.decode_tokens([token_id])
                candidate = prefix + token

                if target.startswith(candidate) or candidate == target:
                    allowed.add(token_id)

        return allowed

    def allowed_param_value(
        self,
        prefix: str,
        param_type: str,
    ) -> set[int]:

        allowed = set()

        for token_id in self.id_to_token:

            token = self.decode_tokens([token_id])

            #
            # NUMBER / INTEGER
            #
            if param_type in ("number", "integer"):

                if any(c.isdigit() for c in token):
                    allowed.add(token_id)

                if "." in token:
                    allowed.add(token_id)

                if "-" in token and prefix.strip() == "":
                    allowed.add(token_id)

                if "," in token or "}" in token:
                    allowed.add(token_id)

            #
            # BOOLEAN
            #
            elif param_type == "boolean":

                candidate = prefix + token

                if "true".startswith(candidate) or candidate == "true":
                    allowed.add(token_id)

                if "false".startswith(candidate) or candidate == "false":
                    allowed.add(token_id)

                if "," in token or "}" in token:
                    allowed.add(token_id)

            #
            # STRING
            #
            else:

                allowed.add(token_id)

        return allowed

    def allowed_colon(self) -> set[int]:
        allowed = set()

        for token_id in self.id_to_token:
            token = self.decode_tokens([token_id])

            if ":" in token:
                allowed.add(token_id)

        return allowed

    def allowed_comma(self) -> set[int]:
        allowed = set()

        for token_id in self.id_to_token:
            token = self.decode_tokens([token_id])

            if "," in token:
                allowed.add(token_id)

        return allowed

    def allowed_open_brace(self) -> set[int]:
        allowed = set()

        for token_id in self.id_to_token:
            token = self.decode_tokens([token_id])

            if "{" in token:
                allowed.add(token_id)

        return allowed

    def allowed_close_brace(self) -> set[int]:
        allowed = set()

        for token_id in self.id_to_token:
            token = self.decode_tokens([token_id])

            if "}" in token:
                allowed.add(token_id)

        return allowed

    def allowed_parameters_key(self, prefix: str) -> set[int]:
        allowed = set()
        target = '"parameters"'

        for token_id in self.id_to_token:
            token = self.decode_tokens([token_id])
            candidate = prefix + token

            if target.startswith(candidate) or candidate == target:
                allowed.add(token_id)

        return allowed

    def allowed_param_separator(
        self,
        current_param_name: str | None,
    ) -> set[int]:
        allowed = set()

        for token_id in self.id_to_token:
            token = self.decode_tokens([token_id])

            # Il reste encore des paramètres
            if current_param_name != "__LAST__":
                if "," in token:
                    allowed.add(token_id)

            # Dernier paramètre
            else:
                if "}" in token:
                    allowed.add(token_id)

        return allowed

    def allowed_tokens(
        self,
        current_state: DecoderState,
        prefix_text: str,
        function: FunctionDefinition | None = None,
        available_functions: list[FunctionDefinition] | None = None,
        current_param_name: str | None = None,
        current_param_type: str | None = None,
    ) -> set[int]:

        if current_state == DecoderState.START:
            return self.allowed_start()

        if current_state == DecoderState.NAME_KEY:
            return self.allowed_name_key(prefix_text)

        if current_state == DecoderState.NAME_COLON:
            return self.allowed_colon()

        if current_state == DecoderState.NAME_VALUE:
            return self.allowed_name_value(
                prefix_text,
                function,
                available_functions,
            )

        if current_state == DecoderState.COMMA_1:
            return self.allowed_comma()

        if current_state == DecoderState.PARAMETERS_KEY:
            return self.allowed_parameters_key(prefix_text)

        if current_state == DecoderState.PARAMETERS_COLON:
            return self.allowed_colon()

        if current_state == DecoderState.PARAMETERS_OPEN:
            return self.allowed_open_brace()

        if current_state == DecoderState.PARAM_NAME:
            return self.allowed_param_name(
                prefix_text,
                function,
            )

        if current_state == DecoderState.PARAM_COLON:
            return self.allowed_colon()

        if current_state == DecoderState.PARAM_VALUE:
            return self.allowed_param_value(
                prefix_text,
                current_param_type,
            )

        if current_state == DecoderState.PARAM_COMMA_OR_CLOSE:
            return self.allowed_param_separator(
                current_param_name,
            )

        if current_state == DecoderState.CLOSE_BRACE:
            return self.allowed_close_brace()

        raise RuntimeError(f"État inconnu : {current_state}")

    def decode(
        self,
        prompt_ids: list[int],
        functions: list[FunctionDefinition],
    ) -> dict[str, Any]:

        self.state = DecoderState.START
        generated_tokens: list[int] = []

        #
        # {
        #
        while True:

            prefix = self.decode_tokens(generated_tokens)

            if "{" in prefix:
                break

            logits = self.get_logits(prompt_ids + generated_tokens)

            allowed = self.allowed_tokens(
                current_state=DecoderState.START,
                prefix_text="",
            )

            filtered = self.filter_logits(logits, allowed)
            token_id = self.best_token(filtered)

            generated_tokens.append(token_id)

        #
        # "name"
        #
        state_tokens = []

        while True:

            prefix = self.decode_tokens(state_tokens)

            if prefix == '"name"':
                break

            logits = self.get_logits(prompt_ids + generated_tokens)

            allowed = self.allowed_tokens(
                current_state=DecoderState.NAME_KEY,
                prefix_text=prefix,
            )

            filtered = self.filter_logits(logits, allowed)
            token_id = self.best_token(filtered)

            generated_tokens.append(token_id)
            state_tokens.append(token_id)

        #
        # :
        #
        while True:

            logits = self.get_logits(prompt_ids + generated_tokens)

            allowed = self.allowed_tokens(
                current_state=DecoderState.NAME_COLON,
                prefix_text="",
            )

            filtered = self.filter_logits(logits, allowed)
            token_id = self.best_token(filtered)

            generated_tokens.append(token_id)

            if ":" in self.decode_tokens([token_id]):
                break

        #
        # "function_name"
        #
        state_tokens = []

        valid_names = [f'"{fn.name}"' for fn in functions]

        while True:

            prefix = self.decode_tokens(state_tokens)

            if prefix in valid_names:
                break

            logits = self.get_logits(prompt_ids + generated_tokens)

            allowed = self.allowed_tokens(
                current_state=DecoderState.NAME_VALUE,
                prefix_text=prefix,
                available_functions=functions,
            )

            filtered = self.filter_logits(logits, allowed)
            token_id = self.best_token(filtered)

            generated_tokens.append(token_id)
            state_tokens.append(token_id)

        fn_name = self.decode_tokens(state_tokens).replace('"', "")

        selected_function = next(
            (
                fn
                for fn in functions
                if fn.name == fn_name
            ),
            functions[0],
        )

        #
        # ,
        #
        while True:

            logits = self.get_logits(prompt_ids + generated_tokens)

            allowed = self.allowed_tokens(
                current_state=DecoderState.COMMA_1,
                prefix_text="",
            )

            filtered = self.filter_logits(logits, allowed)
            token_id = self.best_token(filtered)

            generated_tokens.append(token_id)

            if "," in self.decode_tokens([token_id]):
                break

        #
        # "parameters"
        #
        state_tokens = []

        while True:

            prefix = self.decode_tokens(state_tokens)

            if prefix == '"parameters"':
                break

            logits = self.get_logits(prompt_ids + generated_tokens)

            allowed = self.allowed_tokens(
                current_state=DecoderState.PARAMETERS_KEY,
                prefix_text=prefix,
            )

            filtered = self.filter_logits(logits, allowed)
            token_id = self.best_token(filtered)

            generated_tokens.append(token_id)
            state_tokens.append(token_id)

        #
        # :
        #
        while True:

            logits = self.get_logits(prompt_ids + generated_tokens)

            allowed = self.allowed_tokens(
                current_state=DecoderState.PARAMETERS_COLON,
                prefix_text="",
            )

            filtered = self.filter_logits(logits, allowed)
            token_id = self.best_token(filtered)

            generated_tokens.append(token_id)

            if ":" in self.decode_tokens([token_id]):
                break

        #
        # {
        #
        while True:

            logits = self.get_logits(prompt_ids + generated_tokens)

            allowed = self.allowed_tokens(
                current_state=DecoderState.PARAMETERS_OPEN,
                prefix_text="",
            )

            filtered = self.filter_logits(logits, allowed)
            token_id = self.best_token(filtered)

            generated_tokens.append(token_id)

            if "{" in self.decode_tokens([token_id]):
                break
        #
        # Paramètres
        #
        param_entries = list(selected_function.parameters.items())

        for idx, (param_name, param_def) in enumerate(param_entries):

            #
            # "param_name"
            #
            state_tokens = []

            while True:

                prefix = self.decode_tokens(state_tokens)

                if prefix == f'"{param_name}"':
                    break

                logits = self.get_logits(prompt_ids + generated_tokens)

                allowed = self.allowed_tokens(
                    current_state=DecoderState.PARAM_NAME,
                    prefix_text=prefix,
                    function=selected_function,
                )

                filtered = self.filter_logits(logits, allowed)
                token_id = self.best_token(filtered)

                generated_tokens.append(token_id)
                state_tokens.append(token_id)

            #
            # :
            #
            while True:

                logits = self.get_logits(prompt_ids + generated_tokens)

                allowed = self.allowed_tokens(
                    current_state=DecoderState.PARAM_COLON,
                    prefix_text="",
                )

                filtered = self.filter_logits(logits, allowed)
                token_id = self.best_token(filtered)

                generated_tokens.append(token_id)

                if ":" in self.decode_tokens([token_id]):
                    break

            #
            # Valeur
            #
            if param_def.type == "string":

                state_tokens = []

                while True:

                    prefix = self.decode_tokens(state_tokens)

                    if prefix.count('"') >= 2:
                        break

                    logits = self.get_logits(prompt_ids + generated_tokens)

                    allowed = self.allowed_tokens(
                        current_state=DecoderState.PARAM_VALUE,
                        prefix_text=prefix,
                        current_param_type="string",
                    )

                    filtered = self.filter_logits(logits, allowed)
                    token_id = self.best_token(filtered)

                    generated_tokens.append(token_id)
                    state_tokens.append(token_id)

            elif param_def.type in ("number", "integer"):

                state_tokens = []

                while True:

                    prefix = self.decode_tokens(state_tokens)

                    logits = self.get_logits(prompt_ids + generated_tokens)

                    allowed = self.allowed_tokens(
                        current_state=DecoderState.PARAM_VALUE,
                        prefix_text=prefix,
                        current_param_type="number",
                    )

                    filtered = self.filter_logits(logits, allowed)
                    token_id = self.best_token(filtered)

                    token = self.decode_tokens([token_id])

                    if "," in token or "}" in token:
                        separator = token_id
                        break

                    generated_tokens.append(token_id)
                    state_tokens.append(token_id)

                generated_tokens.append(separator)

            elif param_def.type == "boolean":

                state_tokens = []

                while True:

                    prefix = self.decode_tokens(state_tokens)

                    if prefix in ("true", "false"):
                        break

                    logits = self.get_logits(prompt_ids + generated_tokens)

                    allowed = self.allowed_tokens(
                        current_state=DecoderState.PARAM_VALUE,
                        prefix_text=prefix,
                        current_param_type="boolean",
                    )

                    filtered = self.filter_logits(logits, allowed)
                    token_id = self.best_token(filtered)

                    token = self.decode_tokens([token_id])

                    if "," in token or "}" in token:
                        separator = token_id
                        break

                    generated_tokens.append(token_id)
                    state_tokens.append(token_id)

                generated_tokens.append(separator)

        #
        # Fermeture du JSON racine
        #
        while True:

            logits = self.get_logits(prompt_ids + generated_tokens)

            allowed = self.allowed_tokens(
                current_state=DecoderState.CLOSE_BRACE,
                prefix_text="",
            )

            filtered = self.filter_logits(logits, allowed)
            token_id = self.best_token(filtered)

            generated_tokens.append(token_id)

            if "}" in self.decode_tokens([token_id]):
                break

        full_output = self.decode_tokens(generated_tokens)

        print(repr(full_output))

        try:
            parsed = json.loads(full_output)

            return {
                "name": parsed.get("name", selected_function.name),
                "parameters": parsed.get("parameters", {}),
            }

        except json.JSONDecodeError:

            return {
                "name": selected_function.name,
                "parameters": {},
            }

    #
    # La suite :
    # boucle sur les paramètres...
    #

    # def allowed_tokens(
    #     self,
    #     current_state: DecoderState,
    #     prefix_text: str,
    #     function: FunctionDefinition | None = None,
    #     available_functions: list[FunctionDefinition] | None = None,
    #     current_param_name: str | None = None,
    #     current_param_type: str | None = None,
    # ) -> set[int]:
    #     allowed: set[int] = set()

    #     for t_id in self.id_to_token.keys():
    #         decoded_t = self.decode_tokens([t_id])

    #         if current_state == DecoderState.START:
    #             if "{" in decoded_t:
    #                 allowed.add(t_id)

    #         elif current_state == DecoderState.NAME_KEY:
    #             target = '"name"'
    #             candidate = prefix_text + decoded_t

    #             if target.startswith(candidate) or candidate == target:
    #                 allowed.add(t_id)

    #         elif current_state == DecoderState.NAME_COLON:
    #             if ":" in decoded_t:
    #                 allowed.add(t_id)

    #         elif current_state == DecoderState.NAME_VALUE:
    #             fns = available_functions or ([function] if function else [])

    #             for fn in fns:
    #                 target = f'"{fn.name}"'
    #                 candidate = prefix_text + decoded_t

    #                 if target.startswith(candidate) or candidate == target:
    #                     allowed.add(t_id)

    #         elif current_state == DecoderState.COMMA_1:
    #             if "," in decoded_t:
    #                 allowed.add(t_id)

    #         elif current_state == DecoderState.PARAMETERS_KEY:
    #             target = '"parameters"'
    #             candidate = prefix_text + decoded_t

    #             if target.startswith(candidate) or candidate == target:
    #                 allowed.add(t_id)

    #         elif current_state == DecoderState.PARAMETERS_COLON:
    #             if ":" in decoded_t:
    #                 allowed.add(t_id)

    #         elif current_state == DecoderState.PARAMETERS_OPEN:
    #             if "{" in decoded_t:
    #                 allowed.add(t_id)

    #         elif current_state == DecoderState.PARAM_NAME:
    #             if function:
    #                 for p in function.parameters.keys():
    #                     target = f'"{p}"'
    #                     candidate = prefix_text + decoded_t

    #                     if target.startswith(candidate) or candidate == target:
    #                         allowed.add(t_id)

    #         elif current_state == DecoderState.PARAM_COLON:
    #             if ":" in decoded_t:
    #                 allowed.add(t_id)

    #         elif current_state == DecoderState.PARAM_VALUE:

    #             if current_param_type in ("number", "integer"):

    #                 # chiffres
    #                 if any(c.isdigit() for c in decoded_t):
    #                     allowed.add(t_id)

    #                 # point
    #                 if "." in decoded_t:
    #                     allowed.add(t_id)

    #                 # signe
    #                 if "-" in decoded_t and prefix_text.strip() == "":
    #                     allowed.add(t_id)

    #                 # séparateurs
    #                 if "," in decoded_t or "}" in decoded_t:
    #                     allowed.add(t_id)

    #             elif current_param_type == "boolean":
    #                 candidate = prefix_text + decoded_t

    #                 if "true".startswith(candidate) or candidate == "true":
    #                     allowed.add(t_id)

    #                 if "false".startswith(candidate) or candidate == "false":
    #                     allowed.add(t_id)

    #                 if "," in decoded_t or "}" in decoded_t:
    #                     allowed.add(t_id)

    #             else:
    #                 # string
    #                 allowed.add(t_id)

    #         elif current_state == DecoderState.PARAM_COMMA_OR_CLOSE:

    #             # Il reste encore des paramètres :
    #             # on n'autorise QUE la virgule.
    #             if current_param_name != "__LAST__":
    #                 if "," in decoded_t:
    #                     allowed.add(t_id)

    #             # Dernier paramètre :
    #             # on n'autorise QUE la fermeture.
    #             else:
    #                 if "}" in decoded_t:
    #                     allowed.add(t_id)

    #         elif current_state == DecoderState.CLOSE_BRACE:
    #             if "}" in decoded_t:
    #                 allowed.add(t_id)

    #     if not allowed:
    #         raise RuntimeError(
    #             f"Aucun token valide trouvé pour l'état {current_state} "
    #             f"(préfixe: {prefix_text!r})"
    #         )

    #     return allowed

    # def _generate_until_target(
    #     self,
    #     prompt_ids: list[int],
    #     generated_tokens: list[int],
    #     state: DecoderState,
    #     target_check_fn: Any,
    #     function: FunctionDefinition | None = None,
    #     available_functions: list[FunctionDefinition] | None = None,
    #     param_name: str | None = None,
    #     param_type: str | None = None,
    # ) -> str:
    #     """Génère des tokens en boucle pour un état jusqu'à ce que la condition de fin soit remplie."""
    #     state_tokens: list[int] = []

    #     while True:
    #         prefix_text = self.decode_tokens(state_tokens)
    #         if target_check_fn(prefix_text):
    #             break

    #         current_input = prompt_ids + generated_tokens
    #         print(state, repr(prefix_text))
    #         logits = self.get_logits(current_input)

    #         allowed = self.allowed_tokens(
    #             current_state=state,
    #             prefix_text=prefix_text,
    #             function=function,
    #             available_functions=available_functions,
    #             current_param_name=param_name,
    #             current_param_type=param_type,
    #         )
    #         filtered_logits = self.filter_logits(logits, allowed)
    #         token_id = self.best_token(filtered_logits)

    #         generated_tokens.append(token_id)
    #         state_tokens.append(token_id)

    #     print(f"{state.name} ->", repr(self.decode_tokens(state_tokens)))
    #     return self.decode_tokens(state_tokens)

    # def decode(
    #     self, prompt_ids: list[int], functions: list[FunctionDefinition]
    # ) -> dict[str, Any]:
    #     self.state = DecoderState.START
    #     """Décode une requête en forçant la génération multi-tokens du schéma JSON."""
    #     generated_tokens: list[int] = []

    #     # 1. Ouverture JSON `{`
    #     self._generate_until_target(
    #         prompt_ids, generated_tokens, DecoderState.START, lambda text: "{" in text
    #     )

    #     # 2. Clé `"name"`
    #     self._generate_until_target(
    #         prompt_ids, generated_tokens, DecoderState.NAME_KEY, lambda text: '"name"' in text
    #     )

    #     # 3. Deux-points `:`
    #     self._generate_until_target(
    #         prompt_ids, generated_tokens, DecoderState.NAME_COLON, lambda text: ":" in text
    #     )

    #     # 4. Génération complète du nom de la fonction (ex: `"fn_add_numbers"`)
    #     # On boucle tant que le texte généré ne correspond pas exactement à un `"{fn_name}"`
    #     valid_quoted_names = [f'"{fn.name}"' for fn in functions]
    #     fn_text = self._generate_until_target(
    #         prompt_ids,
    #         generated_tokens,
    #         DecoderState.NAME_VALUE,
    #         lambda text: text.strip() in valid_quoted_names,
    #         available_functions=functions,
    #     )

    #     # Identification exacte de la fonction choisie par le LLM
    #     selected_fn_name = fn_text.strip().replace('"', "")
    #     selected_function = next(
    #         (fn for fn in functions if fn.name == selected_fn_name), functions[0]
    #     )

    #     # 5. Virgule `,`
    #     self._generate_until_target(
    #         prompt_ids, generated_tokens, DecoderState.COMMA_1, lambda text: "," in text
    #     )

    #     # 6. Clé `"parameters"`
    #     self._generate_until_target(
    #         prompt_ids, generated_tokens, DecoderState.PARAMETERS_KEY, lambda text: '"parameters"' in text
    #     )

    #     # 7. Deux-points `:`
    #     self._generate_until_target(
    #         prompt_ids, generated_tokens, DecoderState.PARAMETERS_COLON, lambda text: ":" in text
    #     )

    #     # 8. Ouverture de l'objet arguments `{`
    #     self._generate_until_target(
    #         prompt_ids, generated_tokens, DecoderState.PARAMETERS_OPEN, lambda text: "{" in text
    #     )

    #     # 9. Génération de chaque paramètre
    #     param_entries = list(selected_function.parameters.items())
    #     for idx, (param_name, param_def) in enumerate(param_entries):
    #         target_pname = f'"{param_name}"'

    #         # Nom de paramètre
    #         self._generate_until_target(
    #             prompt_ids,
    #             generated_tokens,
    #             DecoderState.PARAM_NAME,
    #             lambda text: text.strip() == target_pname,
    #             function=selected_function,
    #             param_name=param_name,
    #         )

    #         # Deux-points `:`
    #         self._generate_until_target(
    #             prompt_ids, generated_tokens, DecoderState.PARAM_COLON, lambda text: ":" in text
    #         )

    #         # Valeur du paramètre (selon son type)
    #         if param_def.type == "string":
    #             # Génération jusqu'à la fermeture des guillemets `"...\""`
    #             self._generate_until_target(
    #                 prompt_ids,
    #                 generated_tokens,
    #                 DecoderState.PARAM_VALUE,
    #                 lambda text: text.count('"') >= 2,
    #                 function=selected_function,
    #                 param_type="string",
    #             )
    #         elif param_def.type in ("number", "integer"):
    #             state_tokens = []

    #             while True:
    #                 prefix = self.decode_tokens(state_tokens)
    #                 print(DecoderState.PARAM_VALUE, repr(prefix))

    #                 current_input = prompt_ids + generated_tokens
    #                 logits = self.get_logits(current_input)

    #                 allowed = self.allowed_tokens(
    #                     current_state=DecoderState.PARAM_VALUE,
    #                     prefix_text=prefix,
    #                     function=selected_function,
    #                     current_param_type="number",
    #                 )

    #                 filtered = self.filter_logits(logits, allowed)
    #                 token_id = self.best_token(filtered)

    #                 token = self.decode_tokens([token_id])

    #                 # Le modèle veut passer au séparateur
    #                 if "," in token or "}" in token:
    #                     next_separator = token_id
    #                     break

    #                 generated_tokens.append(token_id)
    #                 state_tokens.append(token_id)

    #             print("PARAM_VALUE ->", repr(self.decode_tokens(state_tokens)))
    #             generated_tokens.append(next_separator)

    #         if idx < len(param_entries) - 1:
    #             pass
    #         else:
    #             pass

    #     # 11. Fermeture du JSON racine `}`
    #     self._generate_until_target(
    #         prompt_ids, generated_tokens, DecoderState.CLOSE_BRACE, lambda text: "}" in text
    #     )

    #     # Reconstitution et validation finale
    #     full_output = self.decode_tokens(generated_tokens)

    #     print("\nTOKENS GÉNÉRÉS :")
    #     print(repr(full_output))
    #     try:
    #         parsed = json.loads(full_output)
    #         return {
    #             "name": parsed.get("name", selected_function.name),
    #             "parameters": parsed.get("parameters", {}),
    #         }
    #     except json.JSONDecodeError:
    #         return {
    #             "name": selected_function.name,
    #             "parameters": {},
    #         }
