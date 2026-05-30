# # utils/pdf_utils.py

# from io import BytesIO

# from reportlab.lib.styles import (
#     getSampleStyleSheet
# )

# from reportlab.platypus import (
#     Paragraph,
#     Spacer,
#     SimpleDocTemplate
# )

# from reportlab.lib.pagesizes import letter

# # ============================================================
# # BASE PDF
# # ============================================================


# def build_pdf(
#     title: str,
#     sections: list[tuple[str, str]]
# ):

#     buffer = BytesIO()

#     doc = SimpleDocTemplate(
#         buffer,
#         pagesize=letter
#     )

#     styles = getSampleStyleSheet()

#     story = []

#     story.append(
#         Paragraph(
#             title,
#             styles["Title"]
#         )
#     )

#     story.append(
#         Spacer(1, 12)
#     )

#     for heading, content in sections:

#         story.append(
#             Paragraph(
#                 heading,
#                 styles["Heading2"]
#             )
#         )

#         story.append(
#             Paragraph(
#                 str(content).replace(
#                     "\n",
#                     "<br/>"
#                 ),
#                 styles["BodyText"]
#             )
#         )

#         story.append(
#             Spacer(1, 10)
#         )

#     doc.build(story)

#     buffer.seek(0)

#     return buffer


# # ============================================================
# # WATCH PARTY PDF
# # ============================================================


# def create_watch_party_pdf(plan):

#     sections = [

#         (
#             "Theme",
#             plan.theme
#         ),

#         (
#             "Selections",
#             "\n".join(
#                 plan.selections
#             )
#         ),

#         (
#             "Narrative Arc",
#             plan.narrative_arc
#         )
#     ]

#     return build_pdf(
#         title="Movie Night Plan",
#         sections=sections
#     )


# # ============================================================
# # PITCH PDF
# # ============================================================


# def create_pitch_pdf(
#     pitch,
#     market_summary,
#     success_probability,
#     confidence
# ):

#     sections = [

#         (
#             "Pitch Title",
#             pitch.title
#         ),

#         (
#             "Logline",
#             pitch.logline
#         ),

#         (
#             "Concept",
#             pitch.concept
#         ),

#         (
#             "Data Integrity",
#             pitch.data_integrity_report
#         ),

#         (
#             "Market Analysis",
#             market_summary
#         ),

#         (
#             "Success Probability",
#             f"{success_probability}%"
#         ),

#         (
#             "Analytical Confidence",
#             f"{confidence}%"
#         )
#     ]

#     return build_pdf(
#         title="Creative Pitch Report",
#         sections=sections
#     )

# utils/pdf_utils.py

from io import BytesIO

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Paragraph,
    Spacer,
    SimpleDocTemplate
)

from reportlab.lib.pagesizes import letter


# ============================================================
# BASE PDF BUILDER
# ============================================================

def build_pdf(title: str, sections: list[tuple[str, str]]):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(title, styles["Title"])
    )

    story.append(Spacer(1, 12))

    for heading, content in sections:

        story.append(
            Paragraph(heading, styles["Heading2"])
        )

        story.append(
            Paragraph(
                str(content).replace("\n", "<br/>"),
                styles["BodyText"]
            )
        )

        story.append(Spacer(1, 10))

    doc.build(story)

    buffer.seek(0)

    return buffer


# ============================================================
# WATCH PARTY PDF (FIXED)
# ============================================================

def create_watch_party_pdf(plan: dict):

    sections = [
        ("Theme", plan.get("theme", "")),

        ("Selections", "\n".join(plan.get("selections", []))),

        ("Narrative Arc", plan.get("narrative", ""))
    ]

    return build_pdf(
        title="Movie Night Plan",
        sections=sections
    )


# ============================================================
# PITCH PDF (SAFE DICT VERSION)
# ============================================================

def create_pitch_pdf(
    pitch,
    market_summary: str,
    success_probability: int,
    confidence: int
):

    # ============================================================
    # SAFE FIELD ACCESS (DICT OR OBJECT)
    # ============================================================

    def safe_get(obj, key, default=""):

        if obj is None:
            return default

        # dict case
        if isinstance(obj, dict):
            return obj.get(key, default)

        # object case
        return getattr(obj, key, default)

    sections = [
        ("Pitch Title", safe_get(pitch, "title")),

        ("Logline", safe_get(pitch, "logline")),

        ("Concept", safe_get(pitch, "concept")),

        ("Data Integrity", safe_get(pitch, "data_report")),

        ("Market Analysis", market_summary),

        ("Success Probability", f"{success_probability}%"),

        ("Analytical Confidence", f"{confidence}%")
    ]

    return build_pdf(
        title="Creative Pitch Report",
        sections=sections
    )