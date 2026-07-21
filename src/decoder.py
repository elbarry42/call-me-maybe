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

        with open(model.get_path_to_vocab_file(), "r") as file:
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

    def allowed_tokens(
        self,
        current_state: DecoderState,
        prefix_text: str,
        function: FunctionDefinition | None = None,
        available_functions: list[FunctionDefinition] | None = None,
        current_param_name: str | None = None,
        current_param_type: str | None = None,
    ) -> set[int]:
        """Détermine l'ensemble strict des tokens autorisés pour l'état et le préfixe courant."""
        allowed: set[int] = set()

        for t_id in self.id_to_token.keys():
            decoded_t = self.decode_tokens([t_id])

            if current_state == DecoderState.START:
                if "{" in decoded_t:
                    allowed.add(t_id)

            elif current_state == DecoderState.NAME_KEY:
                # Doit être un sous-ensemble valide produisant "name"
                target = '"name"'
                candidate = prefix_text + decoded_t

                if target.startswith(candidate) or candidate == target:
                    allowed.add(t_id)

            elif current_state == DecoderState.NAME_COLON:
                if ":" in decoded_t:
                    allowed.add(t_id)

            elif current_state == DecoderState.NAME_VALUE:
                # Autorise les fragments BPE des noms de fonctions valides entre guillemets
                fns = available_functions or ([function] if function else [])
                for fn in fns:
                    fn_target = f'"{fn.name}"'
                    candidate = prefix_text + decoded_t
                    if fn_target.startswith(candidate) or candidate == fn_target:
                        allowed.add(t_id)

            elif current_state == DecoderState.COMMA_1:
                if "," in decoded_t:
                    allowed.add(t_id)

            elif current_state == DecoderState.PARAMETERS_KEY:
                target = '"parameters"'
                candidate = prefix_text + decoded_t

                if target.startswith(candidate) or candidate == target:
                    allowed.add(t_id)

            elif current_state == DecoderState.PARAMETERS_COLON:
                if ":" in decoded_t:
                    allowed.add(t_id)

            elif current_state == DecoderState.PARAMETERS_OPEN:
                if "{" in decoded_t:
                    allowed.add(t_id)

            elif current_state == DecoderState.PARAM_NAME:
                if function:
                    for p in function.parameters.keys():
                        p_target = f'"{p}"'
                        candidate = prefix_text + decoded_t
                        if p_target.startswith(candidate) or candidate == p_target:
                            allowed.add(t_id)

            elif current_state == DecoderState.PARAM_COLON:
                if ":" in decoded_t:
                    allowed.add(t_id)

            elif current_state == DecoderState.PARAM_VALUE:
                if current_param_type == "number":
                    # Autorise les chiffres, points et caractères numériques
                    if any(c in (string.digits + ".-") for c in decoded_t):
                        allowed.add(t_id)

                    if "," in decoded_t or "}" in decoded_t:
                        allowed.add(t_id)
                elif current_param_type == "boolean":
                    if any(b in (prefix_text + decoded_t) for b in ["true", "false"]):
                        allowed.add(t_id)
                else:  # string
                    allowed.add(t_id)

            elif current_state == DecoderState.PARAM_COMMA_OR_CLOSE:
                if "}" in decoded_t or "," in decoded_t:
                    allowed.add(t_id)

            elif current_state == DecoderState.CLOSE_BRACE:
                if "}" in decoded_t:
                    allowed.add(t_id)

        # En cas d'absence de correspondance directe, bascule de sécurité sur le vocabulaire
        if not allowed:
            raise RuntimeError(f"Aucun token valide trouvé pour l'état {current_state} (préfixe: '{prefix_text}')")

        return allowed

    def _generate_until_target(
        self,
        prompt_ids: list[int],
        generated_tokens: list[int],
        state: DecoderState,
        target_check_fn: Any,
        function: FunctionDefinition | None = None,
        available_functions: list[FunctionDefinition] | None = None,
        param_name: str | None = None,
        param_type: str | None = None,
    ) -> str:
        """Génère des tokens en boucle pour un état jusqu'à ce que la condition de fin soit remplie."""
        state_tokens: list[int] = []

        while True:
            prefix_text = self.decode_tokens(state_tokens)
            if target_check_fn(prefix_text):
                break

            current_input = prompt_ids + generated_tokens
            print(state, repr(prefix_text))
            logits = self.get_logits(current_input)

            allowed = self.allowed_tokens(
                current_state=state,
                prefix_text=prefix_text,
                function=function,
                available_functions=available_functions,
                current_param_name=param_name,
                current_param_type=param_type,
            )
            filtered_logits = self.filter_logits(logits, allowed)
            token_id = self.best_token(filtered_logits)

            generated_tokens.append(token_id)
            state_tokens.append(token_id)

        return self.decode_tokens(state_tokens)

    def decode(
        self, prompt_ids: list[int], functions: list[FunctionDefinition]
    ) -> dict[str, Any]:
        print("===== decode() START =====")
        self.state = DecoderState.START
        """Décode une requête en forçant la génération multi-tokens du schéma JSON."""
        generated_tokens: list[int] = []

        # 1. Ouverture JSON `{`
        self._generate_until_target(
            prompt_ids, generated_tokens, DecoderState.START, lambda text: "{" in text
        )

        # 2. Clé `"name"`
        self._generate_until_target(
            prompt_ids, generated_tokens, DecoderState.NAME_KEY, lambda text: '"name"' in text
        )

        # 3. Deux-points `:`
        self._generate_until_target(
            prompt_ids, generated_tokens, DecoderState.NAME_COLON, lambda text: ":" in text
        )

        # 4. Génération complète du nom de la fonction (ex: `"fn_add_numbers"`)
        # On boucle tant que le texte généré ne correspond pas exactement à un `"{fn_name}"`
        valid_quoted_names = [f'"{fn.name}"' for fn in functions]
        fn_text = self._generate_until_target(
            prompt_ids,
            generated_tokens,
            DecoderState.NAME_VALUE,
            lambda text: text.strip() in valid_quoted_names,
            available_functions=functions,
        )

        # Identification exacte de la fonction choisie par le LLM
        selected_fn_name = fn_text.strip().replace('"', "")
        selected_function = next(
            (fn for fn in functions if fn.name == selected_fn_name), functions[0]
        )

        # 5. Virgule `,`
        self._generate_until_target(
            prompt_ids, generated_tokens, DecoderState.COMMA_1, lambda text: "," in text
        )

        # 6. Clé `"parameters"`
        self._generate_until_target(
            prompt_ids, generated_tokens, DecoderState.PARAMETERS_KEY, lambda text: '"parameters"' in text
        )

        # 7. Deux-points `:`
        self._generate_until_target(
            prompt_ids, generated_tokens, DecoderState.PARAMETERS_COLON, lambda text: ":" in text
        )

        # 8. Ouverture de l'objet arguments `{`
        self._generate_until_target(
            prompt_ids, generated_tokens, DecoderState.PARAMETERS_OPEN, lambda text: "{" in text
        )

        # 9. Génération de chaque paramètre
        param_entries = list(selected_function.parameters.items())
        for idx, (param_name, param_def) in enumerate(param_entries):
            target_pname = f'"{param_name}"'

            # Nom de paramètre
            self._generate_until_target(
                prompt_ids,
                generated_tokens,
                DecoderState.PARAM_NAME,
                lambda text: target_pname in text,
                function=selected_function,
                param_name=param_name,
            )

            # Deux-points `:`
            self._generate_until_target(
                prompt_ids, generated_tokens, DecoderState.PARAM_COLON, lambda text: ":" in text
            )

            # Valeur du paramètre (selon son type)
            if param_def.type == "string":
                # Génération jusqu'à la fermeture des guillemets `"...\""`
                self._generate_until_target(
                    prompt_ids,
                    generated_tokens,
                    DecoderState.PARAM_VALUE,
                    lambda text: text.count('"') >= 2,
                    function=selected_function,
                    param_type="string",
                )
            elif param_def.type in ("number", "integer"):
                # Génération tant qu'on a des chiffres, s'arrête si virgule/accolade
                self._generate_until_target(
                    prompt_ids,
                    generated_tokens,
                    DecoderState.PARAM_VALUE,
                    lambda text: len(text.strip()) > 0 and text.strip()[-1] in ",}",
                    function=selected_function,
                    param_type="number",
                )

        # 10. Fermeture de l'objet "parameters" `}`
        self._generate_until_target(
            prompt_ids, generated_tokens, DecoderState.CLOSE_BRACE, lambda text: "}" in text
        )

        # 11. Fermeture du JSON racine `}`
        self._generate_until_target(
            prompt_ids, generated_tokens, DecoderState.CLOSE_BRACE, lambda text: "}" in text
        )

        # Reconstitution et validation finale
        full_output = self.decode_tokens(generated_tokens)
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
    print("===== decode() END =====")
