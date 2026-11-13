
"""LLM setup, prompt templates and four agents using direct Ollama calls."""

import json
import re

from llm import ask_llm
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


# --------------------------------------------------------- Helpers

def render_prompt(template, values):
    """Replace named placeholders without modifying JSON braces."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def parse_json_response(response):
    """Extract JSON safely from an LLM response."""
    response = response.strip()

    # Remove Markdown code fences if present.
    response = re.sub(r"^```(?:json)?\s*", "", response)
    response = re.sub(r"\s*```$", "", response)

    # Extract JSON if the model adds extra text.
    start = response.find("{")
    end = response.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in LLM response: {response}")

    return json.loads(response[start:end + 1])


class Agent:
    """Simple replacement for LangChain chains."""

    def __init__(self, system_prompt, user_prompt, temperature=0.0, json_output=True):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.temperature = temperature
        self.json_output = json_output

    def invoke(self, inputs):
        system = render_prompt(self.system_prompt, inputs)
        user = render_prompt(self.user_prompt, inputs)

        response = ask_llm(
            system,
            user,
            temperature=self.temperature
        )

        if self.json_output:
            return parse_json_response(response)

        return response.strip()


# ---------------------------------------------------------------- Router

ROUTER_SYSTEM = """
You route requests for an Indian government-scheme assistant and extract the citizen's profile.

Intents:
- discovery: wants to know which schemes they may qualify for
- eligibility_check: asks whether they qualify for a specific named scheme
- application_guidance: asks how to apply / which documents for a specific scheme
- other: greetings or unrelated questions

Profile fields you may fill (use null if not stated; never guess):
{fields}

Already known profile: {profile}

Return ONLY valid JSON:
{"intent": "discovery", "target_scheme": null, "profile_updates": {"age": 35, "occupation": "farmer"}}
"""

ROUTER_USER = "{message}"


def router_chain():
    return Agent(ROUTER_SYSTEM, ROUTER_USER)


# ------------------------------------------------------------ Retrieval

class RetrievalAgent:
    """Agentic RAG: builds a profile-aware query and searches ChromaDB."""

    def __init__(self, k=10):
        self.col = get_collection()
        self.k = k

    def build_query(self, message, profile, intent, target):
        if target:
            return f"{target} {message}"

        bits = [
            f"{k}: {v}"
            for k, v in profile.items()
            if v not in (None, "", [])
        ]

        return f"{message}. Citizen profile: " + "; ".join(bits)

    def run(self, message, profile, intent, target=None):
        query = self.build_query(message, profile, intent, target)

        res = self.col.query(
            query_texts=[query],
            n_results=self.k
        )

        chunks = []

        for doc, meta in zip(
            res["documents"][0],
            res["metadatas"][0]
        ):
            chunks.append({"text": doc, **meta})

        if target:
            hit = (
                self.col.get(
                    where={"scheme": {"$eq": chunks[0]["scheme"]}}
                )
                if chunks else None
            )

            if hit and hit["documents"]:
                seen = {c["text"] for c in chunks}

                for doc, meta in zip(
                    hit["documents"],
                    hit["metadatas"]
                ):
                    if doc not in seen:
                        chunks.append({"text": doc, **meta})

        return chunks


def format_context(chunks):
    return "\n\n".join(
        f"[{i + 1}] ({c['scheme']} | {c['section']} | {c['source']})\n{c['text']}"
        for i, c in enumerate(chunks)
    )


# ----------------------------------------------------------- Eligibility

ELIGIBILITY_SYSTEM = """
You are the Eligibility Agent for an Indian government-scheme navigator.

Evaluate each relevant scheme using ONLY the provided official excerpts
and citizen profile.

STRICT RULES:

1. Never assume a missing profile field is true or false.
2. Never assume a person owns land just because they are a farmer.
3. Never assume a person is free from government-employee or income-tax
   exclusions when those details are missing.
4. Check ALL eligibility criteria and exclusions explicitly stated
   in the excerpts.
5. Do not use annual income as a deciding factor unless the scheme
   explicitly specifies an income criterion.

VERDICTS:

- eligible: All applicable stated criteria are verified and no
  applicable exclusion is present.
- likely: All decisive criteria appear satisfied, but a minor
  non-decisive detail remains unconfirmed.
- need_info: A decisive eligibility criterion or exclusion cannot
  be checked because profile information is missing.
- not_eligible: An explicitly stated criterion fails or an
  applicable exclusion is confirmed.

IMPORTANT:
- A farmer is not automatically a landholder farmer.
- Never classify someone as not_eligible based only on occupation.
- Occupation can disqualify someone ONLY when an official excerpt
  explicitly identifies that occupation as an exclusion.
- Do not infer eligibility or ineligibility from stereotypes,
  assumptions, or unrelated scheme conditions.
- Missing exclusion information must result in need_info,
  not eligible.
- Every reason must be directly supported by the provided excerpts
  and relevant to the specific scheme.
- Do not invent criteria that are absent from the excerpts.
- Skip clearly irrelevant schemes.
- When multiple facts are missing, list all relevant missing fields.

Return ONLY valid JSON:

{"results": [{"scheme": "scheme name", "verdict": "need_info", "reasons": ["reason"], "missing_info": ["information"], "sources": [1]}]}
"""

ELIGIBILITY_USER = """
Citizen profile:
{profile}

User message:
{message}

Excerpts:
{context}
"""


def eligibility_chain():
    return Agent(
        ELIGIBILITY_SYSTEM,
        ELIGIBILITY_USER
    )


# ----------------------------------------------------------- Application

APPLICATION_SYSTEM = """
You are the Application Agent.

For each scheme listed, extract from the excerpts ONLY:

1. The benefit (one line).
2. Required documents.
3. Numbered step-by-step application guidance.

If something is not in the excerpts, say:
'not stated in the source - check the official portal'

Do not invent information.

Return ONLY JSON:
{"schemes": [{"scheme": "scheme name", "benefit": "benefit", "documents": ["document"], "steps": ["step"], "portal": "portal", "sources": [1]}]}
"""

APPLICATION_USER = """
Schemes to cover:
{schemes}

Excerpts:
{context}
"""


def application_chain():
    return Agent(
        APPLICATION_SYSTEM,
        APPLICATION_USER
    )


# ------------------------------------------------------------- Explainer


EXPLAINER_SYSTEM = """
You are the final response agent for an Indian government-scheme assistant.

Write in simple, short, friendly sentences (Class 8 reading level), in {language}.

Use ONLY the citizen profile, eligibility results and official scheme excerpts provided.

For each scheme in the eligibility results:
1. Give its name and eligibility verdict in plain words:
   - eligible: You qualify
   - likely: You probably qualify
   - need_info: We need one more detail
   - not_eligible: You do not qualify
2. Explain why, using the eligibility reasons.
3. Explain the scheme benefit using the official excerpts.
4. List required documents found in the excerpts.
5. Give numbered application steps found in the excerpts.

Rules:
- Never invent eligibility criteria, benefits, documents or application steps.
- NEVER change, upgrade, downgrade or reinterpret an eligibility verdict.
- Copy each scheme's verdict exactly from the Eligibility results.
- If verdict is need_info, clearly explain what information is missing.
- If verdict is likely, never present it as confirmed eligibility.
- If verdict is eligible, do not claim eligibility unless every stated criterion is verified.
- Do not assume missing profile details are true or false.
- Do not reject a person based only on their occupation unless the official criteria explicitly exclude it.
- Do not introduce schemes that are absent from the Eligibility results.
- Only use excerpts that actually relate to the scheme being explained.
- If a benefit, document or application step is not supported by an excerpt, say:
  'Not stated in the provided source - check the official portal.'
- Cite source excerpt numbers like [1] after factual claims.

End with a reminder to verify details on the official government portal.
"""

EXPLAINER_USER = """
User asked:
{message}

Intent:
{intent}

Citizen profile:
{profile}

Eligibility results:
{eligibility}

Official scheme excerpts:
{context}
"""



def explainer_chain():
    return Agent(
        EXPLAINER_SYSTEM,
        EXPLAINER_USER,
        temperature=0.2,
        json_output=False
    )
