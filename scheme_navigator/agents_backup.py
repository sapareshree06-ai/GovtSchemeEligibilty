"""LLM setup, prompt templates and the four agents (as LangChain LCEL chains)."""
import os

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ingest import get_collection

PROFILE_FIELDS = {
    "age": "age in years (number)",
    "gender": "male / female / other",
    "state": "state or UT of residence",
    "area": "rural or urban",
    "occupation": "e.g. farmer, student, street vendor, artisan, salaried, unemployed, business owner",
    "annual_income": "annual household income in INR (number)",
    "category": "social category: SC / ST / OBC / General",
    "bpl": "true/false: below poverty line household or has BPL/ration card (Antyodaya etc.)",
    "owns_land": "true/false: owns cultivable land",
    "owns_pucca_house": "true/false: family owns a pucca house",
    "has_bank_account": "true/false",
    "income_tax_payer": "true/false",
    "govt_employee": "true/false (self or family)",
    "family_notes": "other relevant facts (daughter's age, student level, existing LPG, etc.)",
}


def get_llm(temperature: float = 0.0):
    """API model by default (Anthropic); set LLM_PROVIDER=ollama for a local model."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1"), temperature=temperature)
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        temperature=temperature,
        max_tokens=2500,
    )


# ---------------------------------------------------------------- Router
ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You route requests for an Indian government-scheme assistant and extract the "
            "citizen's profile.\n"
            "Intents:\n"
            "- discovery: wants to know which schemes they may qualify for\n"
            "- eligibility_check: asks whether they qualify for a specific named scheme\n"
            "- application_guidance: asks how to apply / which documents for a specific scheme\n"
            "- other: greetings or unrelated questions\n\n"
            "Profile fields you may fill (use null if not stated; never guess):\n{fields}\n\n"
            "Already known profile: {profile}\n\n"
            "Return ONLY JSON: "
            '{{"intent": "...", "target_scheme": "<scheme name or null>", '
            '"profile_updates": {{field: value}}}}',
        ),
        ("human", "{message}"),
    ]
)


def router_chain():
    return ROUTER_PROMPT | get_llm() | JsonOutputParser()


# ------------------------------------------------------------ Retrieval
class RetrievalAgent:
    """Agentic RAG: builds a profile-aware query, searches ChromaDB, returns chunks."""

    def __init__(self, k: int = 10):
        self.col = get_collection()
        self.k = k

    def build_query(self, message: str, profile: dict, intent: str, target: str | None) -> str:
        if target:
            return f"{target} {message}"
        bits = [f"{k}: {v}" for k, v in profile.items() if v not in (None, "", [])]
        return f"{message}. Citizen profile: " + "; ".join(bits)

    def run(self, message, profile, intent, target=None):
        query = self.build_query(message, profile, intent, target)
        res = self.col.query(query_texts=[query], n_results=self.k)
        chunks = []
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            chunks.append({"text": doc, **meta})
        if target:  # also pull every section of the named scheme for complete answers
            hit = self.col.get(where={"scheme": {"$eq": chunks[0]["scheme"]}}) if chunks else None
            if hit and hit["documents"]:
                seen = {c["text"] for c in chunks}
                for doc, meta in zip(hit["documents"], hit["metadatas"]):
                    if doc not in seen:
                        chunks.append({"text": doc, **meta})
        return chunks


def format_context(chunks) -> str:
    return "\n\n".join(
        f"[{i + 1}] ({c['scheme']} | {c['section']} | {c['source']})\n{c['text']}"
        for i, c in enumerate(chunks)
    )


# ----------------------------------------------------------- Eligibility
ELIGIBILITY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Eligibility Agent. Using ONLY the scheme excerpts provided, evaluate the "
            "citizen's profile against each scheme's eligibility criteria and exclusions. "
            "Do not use outside knowledge about criteria. For every scheme that appears in the "
            "excerpts and is plausibly relevant, give a verdict:\n"
            "- eligible: all stated criteria are met\n"
            "- likely: no criterion fails, but small details are unconfirmed\n"
            "- need_info: a decisive criterion depends on a detail missing from the profile\n"
            "- not_eligible: a criterion or exclusion clearly fails\n"
            "Skip schemes that are clearly irrelevant to this person (e.g. a scholarship for a "
            "non-student) rather than listing them as not_eligible.\n\n"
            "Return ONLY JSON: "
            '{{"results": [{{"scheme": str, "verdict": str, "reasons": [str], '
            '"missing_info": [str], "sources": [int]}}]}} '
            "where sources are the excerpt numbers you relied on.",
        ),
        ("human", "Citizen profile:\n{profile}\n\nUser message: {message}\n\nExcerpts:\n{context}"),
    ]
)


def eligibility_chain():
    return ELIGIBILITY_PROMPT | get_llm() | JsonOutputParser()


# ----------------------------------------------------------- Application
APPLICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Application Agent. For each scheme listed, extract from the excerpts "
            "ONLY: the benefit (one line), required documents, and numbered step-by-step "
            "application guidance. If something is not in the excerpts, say 'not stated in the "
            "source - check the official portal' instead of inventing it.\n"
            "Return ONLY JSON: "
            '{{"schemes": [{{"scheme": str, "benefit": str, "documents": [str], '
            '"steps": [str], "portal": str, "sources": [int]}}]}}',
        ),
        ("human", "Schemes to cover: {schemes}\n\nExcerpts:\n{context}"),
    ]
)


def application_chain():
    return APPLICATION_PROMPT | get_llm() | JsonOutputParser()


# ------------------------------------------------------------- Explainer
EXPLAINER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Explainer Agent for ordinary citizens. Write in simple, short, friendly "
            "sentences (Class 8 reading level), in {language}. No jargon; explain any unavoidable "
            "term. Use the structured results below as the only source of facts.\n"
            "Structure:\n"
            "1. One-line summary.\n"
            "2. For each scheme: name, verdict in plain words (You qualify / You probably qualify / "
            "We need one more detail / You do not qualify), why, the benefit, documents needed and "
            "numbered steps. Cite excerpt numbers like [1] after claims.\n"
            "3. If details are missing, list them under 'To confirm, please tell me:'.\n"
            "4. End with a reminder to verify on the official portal before applying.\n"
            "Do not add schemes or facts that are not in the results.",
        ),
        (
            "human",
            "User asked: {message}\nIntent: {intent}\n\nEligibility results:\n{eligibility}\n\n"
            "Application details:\n{application}",
        ),
    ]
)


def explainer_chain():
    return EXPLAINER_PROMPT | get_llm(0.2) | StrOutputParser()