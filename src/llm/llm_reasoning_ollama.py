import ollama
import re
from typing import Any, Dict, List
from .prompts import PLAIN_ENGINE_REPORT_TEMPLATE

# -----------------------------------------------------------------------------
# HUMAN-READABLE SENSOR LABELS (DO NOT CHANGE)
# -----------------------------------------------------------------------------
SENSOR_LABELS = {
    "s2": "Fan Inlet Temperature",
    "s3": "LPC Outlet Temperature",
    "s4": "HPC Outlet Temperature",
    "s7": "HPC Outlet Pressure",
    "s8": "LPT Outlet Temperature",
    "s9": "Fan Speed (Nc)",
    "s11": "Core Speed (Nf)",
    "s12": "HPC Bleed Pressure",
    "s13": "Bleed Enthalpy",
    "s14": "Fan Vibration",
    "s15": "Core Vibration",
    "s17": "Fuel Flow",
    "s20": "Pressure Ratio (P2/P15)",
    "s21": "Bypass Ratio",
}

# -----------------------------------------------------------------------------
# CLEANING HELPERS
# -----------------------------------------------------------------------------
def _clean_line(text: str) -> str:
    # remove numbering (ex: "1. Title")
    text = re.sub(r"^\d+[\.\)]\s*", "", text.strip())

    # remove markdown emphasis artifacts
    text = text.replace("**", "").replace("*", "")

    # cleanup whitespace
    return text.strip()

# -----------------------------------------------------------------------------
# HEADING MATCHER (ROBUST)
# -----------------------------------------------------------------------------
def _matches_heading(line: str, target: str) -> bool:
    line = _clean_line(line).lower().strip(" :")
    target = target.lower().strip()

    if not line:
        return False

    # exact or near match
    return (
        line == target
        or target in line
        or line.startswith(target.split()[0])
        or line.replace("summary", "").strip() == target.replace("summary", "").strip()
    )

# -----------------------------------------------------------------------------
# BULLET PARSER — Converts DeepSeek bullets to "- "
# -----------------------------------------------------------------------------
UNICODE_BULLETS = ["•", "●", "▪", "‣", "○", "◦", "–", "*", "·"]

def _extract_bullets_and_content(lines: List[str]):
    bullets = []
    content = []

    for raw in lines:
        if raw is None:
            continue

        line = raw.strip()
        if not line:
            continue

        # Replace any Unicode or markdown bullet → "- "
        for b in UNICODE_BULLETS:
            if line.startswith(b):
                line = "- " + line[len(b):].strip()

        # Ensure "- " is treated as bullet
        if line.startswith("- "):
            bullets.append(_clean_line(line[2:]))
        else:
            content.append(_clean_line(line))

    return {
        "content": "\n".join(content).strip(),
        "bullets": bullets,
    }

# -----------------------------------------------------------------------------
# MAIN LLM EXPLANATION PIPELINE
# -----------------------------------------------------------------------------
def llm_engine_explanation(
    subset,
    rul_pred,
    hi_pred,
    sensor_stats,
    attention_weights,
    model_name="deepseek-r1",
):
    # Keep exact input order for sensors
    ordered_sensor_pairs = [
        f"{code} ({SENSOR_LABELS[code]}): {sensor_stats[code]}"
        for code in sensor_stats
        if code in SENSOR_LABELS
    ]
    sensor_str = ", ".join(ordered_sensor_pairs)

    attention_str = ", ".join([str(a) for a in attention_weights])

    # final LLM prompt
    prompt = PLAIN_ENGINE_REPORT_TEMPLATE.format(
        subset=subset,
        rul_pred=rul_pred,
        hi_pred=hi_pred,
        sensor_stats=sensor_str,
        attention_weights=attention_str,
    )

    # DeepSeek call
    response = ollama.generate(
        model=model_name,
        prompt=prompt,
        options={"temperature": 0.15, "format": "text"},
    )

    raw_lines = response.get("response", "").split("\n")

    TITLES = [
        "Overall Diagnostic Assessment",
        "Sensor Deviations Summary",
        "Failure Mode Assessment",
        "Maintenance Recommendations",
    ]

    sections = []
    current_title = None
    buffer = []

    # Parse line-by-line
    for line in raw_lines:
        cleaned_line = _clean_line(line)

        # heading?
        matched = None
        for title in TITLES:
            if _matches_heading(cleaned_line, title):
                matched = title
                break

        if matched:
            # flush previous section
            if current_title is not None:
                parsed = _extract_bullets_and_content(buffer)
                sections.append(
                    {
                        "title": current_title,
                        "content": parsed["content"],
                        "bullet_points": parsed["bullets"],
                    }
                )
            current_title = matched
            buffer = []
        else:
            buffer.append(cleaned_line)

    # flush last section
    if current_title:
        parsed = _extract_bullets_and_content(buffer)
        sections.append(
            {
                "title": current_title,
                "content": parsed["content"],
                "bullet_points": parsed["bullets"],
            }
        )

    # Ensure all four sections exist (prevent empty DOCX)
    found = {sec["title"] for sec in sections}
    for title in TITLES:
        if title not in found:
            sections.append({"title": title, "content": "", "bullet_points": []})

    # sort in correct final order
    sections.sort(key=lambda s: TITLES.index(s["title"]))

    return sections
# ============================================================================