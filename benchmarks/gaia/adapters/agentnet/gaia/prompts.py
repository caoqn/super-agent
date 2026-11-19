"""Answer-format constraints for GAIA (follow the official GAIA answer rules)."""


class GAIAPromptSet:
    @staticmethod
    def get_constraint():
        return """Your final answer for the **major task** must be:
                - a number, OR as few words as possible, OR a comma separated list of numbers and/or strings.
                - For a number: no thousands separators and no units such as $ or % unless the question asks for them.
                - For a string: no articles, no abbreviations (e.g. for cities), and write digits in plain text unless told otherwise.
                - For a list: apply the rules above to each element, separated by commas.
                Follow any formatting instruction in the question itself (rounding, units, order, etc.).
                Respond only with the answer, without repeating the question or adding explanations."""

    @staticmethod
    def get_split_constraint():
        return """Provide a concise response that includes a brief acknowledgment of the task you are executing which is in the **Task Description** and then the result, including the concrete facts, numbers or sources you found. Focus only on the specific part described in the **Task Description**."""

    @staticmethod
    def get_thought_constraint():
        return """You are now in the reasoning phase. Provide detailed thought processes, including possible steps, information sources to look up, files to inspect, and calculations to perform. Avoid providing the final answer in the specified answer format."""
