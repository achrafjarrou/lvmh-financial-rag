from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.config import Config
from loguru import logger


class LLMClient:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=Config.llm_api_key,
            model_name=Config.llm_model,
            temperature=Config.llm_temperature,
            max_tokens=Config.llm_max_tokens,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "{context}\n\nQuestion: {question}\n\nAnswer:")
        ])

        logger.info(f"LLM initialized: {Config.llm_model}")

    def _get_system_prompt(self) -> str:
        """
        Finance-safe system prompt (no hallucination, audit-friendly)
        """
        return """You are a financial analyst specialized in corporate reporting.

RULES:
1. Use ONLY the provided context.
2. If the exact answer is explicitly stated, give it clearly.
3. If the answer is implicit:
   - Look for equivalent terms (e.g. revenue = net sales).
   - Extract values from tables if clearly identifiable.
4. If the information is missing:
   - Explain what is available.
   - Explain what is not available.
5. NEVER invent numbers.

FORMAT:
- Short answer (1–3 sentences).
- Cite sources like: [Page X].

EXAMPLE:
Q: What was LVMH revenue in 2023?
A: LVMH reported total revenue of 86,153 million euros in 2023 [Page 52].
"""

    def generate(self, context: str, question: str) -> str:
        """
        Generate an answer from context + question
        """
        try:
            messages = self.prompt.format_messages(
                context=context,
                question=question
            )

            response = self.llm.invoke(messages)
            return response.content.strip()

        except Exception as e:
            logger.error(f"LLM error: {e}")
            return (
                "The answer could not be generated due to a system error. "
                "Please try again or refine the question."
            )


if __name__ == "__main__":
    client = LLMClient()

    test_context = "[Doc 1, Page 52]: LVMH reported revenue of 86,153 million euros in 2023."
    test_question = "What was LVMH revenue in 2023?"

    answer = client.generate(test_context, test_question)
    print(f"Q: {test_question}")
    print(f"R: {answer}")
