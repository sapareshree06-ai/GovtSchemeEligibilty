
"""Orchestrates intent routing -> retrieval -> eligibility -> explanation."""

from agents import (
    PROFILE_FIELDS,
    RetrievalAgent,
    eligibility_chain,
    explainer_chain,
    format_context,
    router_chain,
)

# Minimum details needed before discovery / eligibility answers are worth giving.
REQUIRED_FOR_DISCOVERY = {
    "age": "your age",
    "occupation": "what work you do (e.g. farmer, student, vendor, artisan, salaried)",
    "annual_income": "your approximate yearly family income",
    "state": "the state you live in",
}


def merge_profile(profile: dict, updates: dict | None) -> dict:
    merged = dict(profile)
    for k, v in (updates or {}).items():
        if k in PROFILE_FIELDS and v not in (None, "", "null"):
            merged[k] = v
    return merged

def missing_for_discovery(profile: dict) -> list[str]:
    return [
        q for k, q in REQUIRED_FOR_DISCOVERY.items()
        if profile.get(k) in (None, "")
    ]


def validate_eligibility(results: dict, profile: dict) -> dict:
    """Apply deterministic eligibility safeguards."""

    for r in results.get("results", []):

        scheme = r.get("scheme", "").upper()

        # PM-KISAN validation
        if "PM-KISAN" in scheme:
            missing = []

            if profile.get("owns_land") is None:
                missing.append(
                    "Whether your family owns cultivable land recorded "
                    "in state land records"
                )
            elif profile.get("owns_land") is False:
                r["verdict"] = "not_eligible"
                r.setdefault("reasons", []).append(
                    "The scheme requires a family to own cultivable land."
                )

            if profile.get("income_tax_payer") is None:
                missing.append(
                    "Whether you or your family members are income-tax payers"
                )

            if profile.get("govt_employee") is None:
                missing.append(
                    "Whether you or your family members fall under "
                    "the excluded government employee categories"
                )

            if missing and r["verdict"] in {"eligible", "likely"}:
                r["verdict"] = "need_info"

            r.setdefault("missing_info", []).extend(
                item for item in missing
                if item not in r.get("missing_info", [])
            )

        # PMAY-Urban validation
        if "PMAY-URBAN" in scheme:
            missing = []

            if profile.get("area") is None:
                missing.append("Whether you reside in an urban area")

            if profile.get("owns_pucca_house") is None:
                missing.append(
                    "Whether you or any family member owns a pucca house "
                    "anywhere in India"
                )

            if profile.get("annual_income") is None:
                missing.append("Your annual household income")

            if missing:
                r["verdict"] = "need_info"
                r["reasons"] = [
                    reason for reason in r.get("reasons", [])
                    if "farmer" not in reason.lower()
                    and "occupation" not in reason.lower()
                ]
                r["missing_info"] = list(dict.fromkeys(
                    r.get("missing_info", []) + missing
                ))

            elif (
                profile.get("area") == "rural"
                or profile.get("owns_pucca_house") is True
            ):
                r["verdict"] = "not_eligible"

    return results


class Navigator:
    def __init__(self):
        self.router = router_chain()
        self.retriever = RetrievalAgent()
        self.eligibility = eligibility_chain()
        self.explainer = explainer_chain()

    def handle(
        self,
        message: str,
        profile: dict,
        language: str = "English",
        ask_first: bool = True
    ):
        """Returns dict(answer, profile, intent, sources, trace)."""

        trace = []

        # 1. Route the user's request and extract profile information.
        route = self.router.invoke({
            "message": message,
            "profile": profile,
            "fields": _fields_text()
        })

        intent = route.get("intent", "discovery")
        target = route.get("target_scheme")

        profile = merge_profile(profile, route.get("profile_updates"))

        trace.append(f"Router: intent={intent}, target={target}")

        if intent == "other":
            return {
                "answer": (
                    "Hello! Tell me a little about yourself "
                    "(age, work, income, state) and I will find "
                    "government schemes you may qualify for, or "
                    "ask me about a specific scheme."
                ),
                "profile": profile,
                "intent": intent,
                "sources": [],
                "trace": trace,
            }

        # 2. Ask for missing information when necessary.
        if ask_first and intent in {"discovery", "eligibility_check"}:
            missing = missing_for_discovery(profile)

            if missing:
                asks = "\n".join(f"- {m}" for m in missing)

                return {
                    "answer": (
                        "To check which schemes fit you, I still need:\n"
                        + asks
                    ),
                    "profile": profile,
                    "intent": intent,
                    "sources": [],
                    "trace": trace + [f"Missing details: {missing}"],
                }

        # 3. Retrieve relevant official government document excerpts.
        chunks = self.retriever.run(
            message,
            profile,
            intent,
            target
        )

        context = format_context(chunks)

        trace.append(
            f"Retrieval: {len(chunks)} chunks from "
            f"{sorted({c['scheme'] for c in chunks})}"
        )

        # 4. Evaluate eligibility using official document excerpts.
        elig = {"results": []}

        if intent in {
            "discovery",
            "eligibility_check",
            "application_guidance"
        }:
            elig = self.eligibility.invoke({
                "profile": profile,
                "message": message,
                "context": context
            })
            elig = validate_eligibility(elig, profile)

            trace.append(
                "Eligibility: " + ", ".join(
                    f"{r['scheme']}={r['verdict']}"
                    for r in elig.get("results", [])
                )
            )

        # 5. Generate the final answer directly.
        # The Explainer now handles benefits, documents and application steps.
        answer = self.explainer.invoke({
            "message": message,
            "intent": intent,
            "language": language,
            "profile": profile,
            "eligibility": elig,
            "context": context,
        })

        trace.append(
            "Application: handled by Explainer (combined generation)"
        )

        # 6. Preserve source references from eligibility results.
        used = {
            n
            for r in elig.get("results", [])
            for n in r.get("sources", [])
        }

        sources = [
            {
                "n": i + 1,
                "scheme": c["scheme"],
                "section": c["section"],
                "source": c["source"]
            }
            for i, c in enumerate(chunks)
            if (i + 1) in used
        ]

        if not sources:
            sources = [
                {
                    "n": i + 1,
                    "scheme": c["scheme"],
                    "section": c["section"],
                    "source": c["source"]
                }
                for i, c in enumerate(chunks)
            ]

        return {
            "answer": answer,
            "profile": profile,
            "intent": intent,
            "sources": sources,
            "trace": trace,
        }


def _fields_text() -> str:
    return "\n".join(
        f"- {k}: {v}"
        for k, v in PROFILE_FIELDS.items()
    )
