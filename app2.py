import streamlit as st
import streamlit.components.v1 as components
import re
import html
import json
import requests

st.set_page_config(page_title="Gender Bias Language Checker", layout="wide")

OLLAMA_MODEL = "llama3"

BIAS_DICTIONARY = {
    "chairman": {"alternative": "chair / chairperson", "category": "Gendered noun", "explanation": "Uses male as the default for a leadership role."},
    "spokesman": {"alternative": "spokesperson / representative", "category": "Gendered noun", "explanation": "Assumes the spokesperson is male."},
    "mankind": {"alternative": "humanity / humankind / people", "category": "Male-as-generic", "explanation": "Uses 'man' to refer to all people."},
    "manpower": {"alternative": "workforce / staff / human resources", "category": "Male-as-generic", "explanation": "Frames labour as male by default."},
    "policeman": {"alternative": "police officer", "category": "Gendered occupation", "explanation": "Gendered job title."},
    "businessman": {"alternative": "business executive / businessperson", "category": "Gendered occupation", "explanation": "Assumes the person in business is male."},
    "salesman": {"alternative": "salesperson / sales representative", "category": "Gendered occupation", "explanation": "Gendered job title."},
    "stewardess": {"alternative": "flight attendant", "category": "Gendered occupation", "explanation": "Gendered job title."},
    "repairman": {"alternative": "technician / repairer", "category": "Gendered occupation", "explanation": "Assumes the worker is male."},
    "bossy": {"alternative": "assertive", "category": "Gendered adjective", "explanation": "Often used negatively for women in leadership."},
    "hysterical": {"alternative": "upset / distressed / irrational", "category": "Gendered adjective", "explanation": "Historically used to trivialise women’s emotions."},
    "emotional": {"alternative": "passionate / empathetic / expressive", "category": "Gendered adjective", "explanation": "Can carry gendered connotations depending on context."},
    "virile": {"alternative": "strong / energetic / decisive", "category": "Gendered adjective", "explanation": "Associates strength with masculinity."},
    "man up": {"alternative": "be brave / be resilient / stay strong", "category": "Stereotypical phrase", "explanation": "Links courage and toughness with masculinity."},
    "like a girl": {"alternative": "poorly / weakly / not effectively", "category": "Stereotypical phrase", "explanation": "Uses femininity as a negative comparison."},
    "dear sir": {"alternative": "Dear Sir or Madam / Dear colleagues / Hello", "category": "Non-inclusive greeting", "explanation": "Assumes the recipient is male."}
}

GENERIC_PRONOUN_PATTERNS = [
    {"pattern": r"\beach applicant must submit his\b", "alternative": "each applicant must submit their", "category": "Generic male pronoun", "explanation": "Uses 'his' as the default pronoun."},
    {"pattern": r"\bevery employee .* himself\b", "alternative": "all employees ... themselves", "category": "Generic male pronoun", "explanation": "Uses male pronoun as generic."},
    {"pattern": r"\bhe or she\b", "alternative": "they", "category": "Binary pronoun", "explanation": "Can be simplified and made more inclusive with singular 'they'."}
]


def first_alternative(alternative):
    return alternative.split("/")[0].strip()


def find_bias_rule_based(text):
    results = []

    for term, info in BIAS_DICTIONARY.items():
        pattern = r"\b" + re.escape(term) + r"\b"
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            results.append({
                "term": match.group(),
                "start": match.start(),
                "end": match.end(),
                "category": info["category"],
                "alternative": info["alternative"],
                "correction": first_alternative(info["alternative"]),
                "explanation": info["explanation"],
                "source": "Rule-based"
            })

    for rule in GENERIC_PRONOUN_PATTERNS:
        for match in re.finditer(rule["pattern"], text, flags=re.IGNORECASE):
            results.append({
                "term": match.group(),
                "start": match.start(),
                "end": match.end(),
                "category": rule["category"],
                "alternative": rule["alternative"],
                "correction": first_alternative(rule["alternative"]),
                "explanation": rule["explanation"],
                "source": "Rule-based"
            })

    return sorted(results, key=lambda x: x["start"])


def extract_json_from_text(raw_text):
    raw_text = raw_text.strip()

    try:
        data = json.loads(raw_text)
    except Exception:
        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except Exception:
                return []
        else:
            return []

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        for key in ["results", "items", "issues", "biases", "detections"]:
            if key in data and isinstance(data[key], list):
                return [item for item in data[key] if isinstance(item, dict)]

    return []


def find_bias_llama_local(text):
    prompt = f"""
You are a gender-sensitive communication assistant for Research & Innovation communication.

Analyze the text and identify gender-biased, gender-stereotypical, exclusionary, or non-inclusive expressions.

Rules:
- Detect expressions even if they are NOT predefined keywords.
- Focus on gendered assumptions, male-default language, biased professional titles, stereotypical descriptors, greetings, invisibility, omission, trivialisation, and exclusionary phrasing.
- Return only exact phrases that appear in the text.
- Do not invent phrases.
- Return ONLY valid JSON.
- The JSON must be either:
  [
    {{
      "term": "exact phrase from the text",
      "category": "short category",
      "alternative": "inclusive alternative / second alternative",
      "explanation": "short explanation"
    }}
  ]
- If there are no issues, return [].

Text:
{text}
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=180
        )

        if response.status_code != 200:
            st.error(f"Ollama error {response.status_code}: {response.text}")
            return []

        raw_output = response.json().get("response", "")
        llm_items = extract_json_from_text(raw_output)

    except Exception as e:
        st.error(f"Local Llama analysis failed: {e}")
        st.info("Ollama is probably not reachable. Test it with: curl http://localhost:11434/api/tags")
        return []

    results = []

    for item in llm_items:
        if not isinstance(item, dict):
            continue

        term = str(item.get("term", "")).strip()
        if not term:
            continue

        match = re.search(re.escape(term), text, flags=re.IGNORECASE)

        if match:
            alternative = str(item.get("alternative", "inclusive alternative"))

            results.append({
                "term": match.group(),
                "start": match.start(),
                "end": match.end(),
                "category": str(item.get("category", "Llama-detected bias")),
                "alternative": alternative,
                "correction": first_alternative(alternative),
                "explanation": str(item.get("explanation", "Detected by local Llama as potentially non-inclusive.")),
                "source": "Local Llama"
            })

    return results


def find_bias_hybrid(text):
    rule_results = find_bias_rule_based(text)
    llama_results = find_bias_llama_local(text)

    combined = rule_results + llama_results
    unique = []
    seen = set()

    for item in combined:
        key = (item["start"], item["end"], item["term"].lower())
        if key not in seen:
            unique.append(item)
            seen.add(key)

    return sorted(unique, key=lambda x: x["start"])


def calculate_bias_score(text, results):
    words = re.findall(r"\b\w+\b", text)
    if len(words) == 0:
        return 0
    return min(round((len(results) / len(words)) * 100, 2), 100)


def get_auto_corrected_text(text, results):
    corrected = text
    for item in sorted(results, key=lambda x: x["start"], reverse=True):
        corrected = corrected[:item["start"]] + first_alternative(item["alternative"]) + corrected[item["end"]:]
    return corrected


def clickable_highlighted_text(text, results):
    safe_text = html.escape(text)
    offset_results = []

    for item in results:
        before = html.escape(text[:item["start"]])
        term = html.escape(text[item["start"]:item["end"]])
        start = len(before)
        end = start + len(term)

        alternatives = [alt.strip() for alt in item["alternative"].split("/")]

        buttons_html = ""
        for alt in alternatives:
            safe_alt = html.escape(alt)
            buttons_html += (
                f'<button class="suggestion-btn" onclick="chooseCorrection(event, this)" '
                f'data-correction="{safe_alt}">{safe_alt}</button>'
            )

        offset_results.append({
            **item,
            "safe_start": start,
            "safe_end": end,
            "buttons_html": buttons_html
        })

    highlighted = safe_text

    for item in sorted(offset_results, key=lambda x: x["safe_start"], reverse=True):
        original = highlighted[item["safe_start"]:item["safe_end"]]
        replacement = (
            f'<span class="bias-word" onclick="toggleSuggestions(event, this)">'
            f'<span class="suggestion-box">{item["buttons_html"]}</span>'
            f'{original}</span>'
        )
        highlighted = highlighted[:item["safe_start"]] + replacement + highlighted[item["safe_end"]:]

    return f"""
    <html>
    <head>
        <style>
            body {{
                background-color: #0e1117;
                color: white;
                font-family: sans-serif;
                margin: 0;
                padding: 55px 0 8px 0;
            }}
            .text-box {{
                font-size: 16px;
                line-height: 2.3;
                color: white;
                white-space: pre-wrap;
                word-wrap: break-word;
                overflow-wrap: break-word;
            }}
            .bias-word {{
                background-color: #ffd966;
                color: black;
                padding: 1px 4px;
                border-radius: 3px;
                cursor: pointer;
                position: relative;
                display: inline;
                margin: 0 1px;
            }}
            .suggestion-box {{
                display: none;
                position: absolute;
                bottom: 130%;
                left: 0;
                background-color: #1f2937;
                padding: 6px;
                border-radius: 7px;
                white-space: nowrap;
                z-index: 9999;
                gap: 5px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.35);
            }}
            .suggestion-box.show {{
                display: inline-flex;
            }}
            .suggestion-btn {{
                background-color: #2ecc71;
                color: black;
                border: none;
                padding: 4px 8px;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
                cursor: pointer;
            }}
            .corrected {{
                background-color: #2ecc71 !important;
                color: black !important;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="text-box">{highlighted}</div>
        <script>
            function toggleSuggestions(event, element) {{
                event.stopPropagation();
                document.querySelectorAll(".suggestion-box").forEach(box => {{
                    if (box !== element.querySelector(".suggestion-box")) {{
                        box.classList.remove("show");
                    }}
                }});
                const box = element.querySelector(".suggestion-box");
                box.classList.toggle("show");
            }}

            function chooseCorrection(event, button) {{
                event.stopPropagation();
                const correction = button.getAttribute("data-correction");
                const parent = button.closest(".bias-word");
                parent.textContent = correction;
                parent.classList.add("corrected");
            }}

            document.addEventListener("click", function() {{
                document.querySelectorAll(".suggestion-box").forEach(box => {{
                    box.classList.remove("show");
                }});
            }});
        </script>
    </body>
    </html>
    """


def create_report_html(text, corrected_text, results, bias_score):
    report_html = f"""
    <html>
    <head>
        <title>Gender Bias Language Checker Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                padding: 40px;
                line-height: 1.6;
                color: #222;
            }}
            h1 {{
                border-bottom: 2px solid #222;
                padding-bottom: 10px;
            }}
            .box {{
                border: 1px solid #ccc;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 25px;
                white-space: pre-wrap;
                background: #f7f7f7;
            }}
            .score {{
                font-size: 24px;
                font-weight: bold;
                color: #c0392b;
            }}
        </style>
    </head>
    <body>
        <h1>Gender Bias Language Checker Report</h1>

        <h2>Detected gender-biased language score</h2>
        <p class="score">{bias_score}%</p>

        <h2>Original text</h2>
        <div class="box">{html.escape(text)}</div>

        <h2>Corrected text</h2>
        <div class="box">{html.escape(corrected_text)}</div>

        <h2>Detected issues</h2>
        <ul>
    """

    for item in results:
        report_html += f"""
            <li>
                <strong>{html.escape(item['term'])}</strong><br>
                Source: {html.escape(item.get('source', 'Unknown'))}<br>
                Category: {html.escape(item['category'])}<br>
                Suggested alternative: {html.escape(item['alternative'])}<br>
                Explanation: {html.escape(item['explanation'])}
            </li>
        """

    report_html += """
        </ul>
        <script>window.print();</script>
    </body>
    </html>
    """

    return report_html


st.title("Gender Bias Language Checker")
st.write("Hybrid prototype: rule-based + local Llama gender-sensitive language detection.")

example_text = """The chairman said that mankind needs more manpower.
Each applicant must submit his CV. The spokesman explained that the best businessman should lead the team.
Dear Sir, the nurse should be caring and emotional."""

text = st.text_area("Paste your text here:", value=example_text, height=250)

analysis_mode = st.radio(
    "Analysis mode",
    ["Hybrid: Rule-based + Local Llama", "Rule-based only"],
    horizontal=True
)

if st.button("Analyze text"):
    st.session_state["text"] = text

    if analysis_mode == "Hybrid: Rule-based + Local Llama":
        with st.spinner("Analyzing with rule-based system and local Llama..."):
            st.session_state["results"] = find_bias_hybrid(text)
    else:
        st.session_state["results"] = find_bias_rule_based(text)

if "results" in st.session_state:
    results = st.session_state["results"]
    text = st.session_state["text"]

    bias_score = calculate_bias_score(text, results)
    corrected_text = get_auto_corrected_text(text, results)

    st.subheader("Bias sensitivity score")
    st.metric(label="Detected gender-biased language", value=f"{bias_score}%")
    st.caption("This score is based on detected biased words/phrases compared to the total number of words.")

    st.subheader("Highlighted text")

    if results:
        components.html(
            clickable_highlighted_text(text, results),
            height=260,
            scrolling=False
        )
    else:
        st.success("No gender-biased terms or expressions were detected.")

    st.subheader("Detected issues")

    if not results:
        st.success("No predefined or Llama-detected gender-biased terms were detected.")
    else:
        for i, item in enumerate(results, start=1):
            st.markdown(f"""
            **{i}. Detected phrase:** `{item['term']}`  
            **Source:** {item.get('source', 'Unknown')}  
            **Category:** {item['category']}  
            **Suggested alternative:** `{item['alternative']}`  
            **Explanation:** {item['explanation']}
            ---
            """)

    st.subheader("Before / After Report")
    st.write("This report uses automatic correction with the first suggested alternative.")

    if st.button("Generate report for PDF"):
        report_html = create_report_html(text, corrected_text, results, bias_score)
        components.html(report_html, height=0)



#"Dear Sir,The chairman explained that mankind needs more manpower in technology projects. He said that every engineer must submit his report on time and that the best man for the job should lead the team.
#The female scientist was described as emotional and bossy, while the male researcher was called assertive and visionary. The nurse should be caring and gentle, while the engineer must be strong and decisive.
#The spokesman added that the project needs young girls to help with communication tasks and strong men to manage technical decisions.
#Kind regards,
#The Research Team
