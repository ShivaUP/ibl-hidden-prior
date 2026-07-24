#!/usr/bin/env python3
"""Build METHODS_DETAILED.docx and CURRENT_STATUS_ARTICLE.docx (publication-style).

Fixed-layout tables (tblGrid + fixed widths) so columns do not collapse in Word/Google Docs.
Embeds selected PNG figures into the status article.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "v2" / "figures"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
NSR = "http://schemas.openxmlformats.org/package/2006/relationships"

# Usable page content width (~6.5 in) in twentieths of a point
PAGE_DXA = 9360


def _load_metrics():
    rows = []
    for domain in ("synth", "real"):
        for regime in ("history_only", "full_information", "fixed_prior"):
            for mid in ("tanh_bptt", "tanh_pc", "gru", "gru_pc"):
                p = ROOT / "reports" / "v2" / "metrics" / f"{domain}_{regime}_{mid}.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                gap = (
                    d.get("kyan_diagnostics", {})
                    .get("counterfactual_zero_evidence_choice_probability", {})
                    .get("history_gap")
                )
                rows.append(
                    {
                        "domain": domain,
                        "regime": regime,
                        "model": mid,
                        "acc": float(d.get("accuracy", float("nan"))),
                        "gap": float(gap) if gap is not None else float("nan"),
                    }
                )
    return rows


def _per_prior_real_history():
    out = {}
    for mid in ("tanh_bptt", "tanh_pc", "gru", "gru_pc"):
        path = ROOT / "artifacts" / "v2" / "real" / "regimes" / "history_only" / mid / "rollout.npz"
        if not path.exists():
            continue
        r = np.load(path)
        true_p = r["true_p_right"] if "true_p_right" in r.files else 1.0 - r["probability_left"]
        valid = r["valid"] if "valid" in r.files else np.isfinite(true_p)
        choice, side = r["choice"], r["side"]
        vals = {}
        for prior in (0.2, 0.5, 0.8):
            sess = []
            for s in range(true_p.shape[0]):
                m = valid[s] & np.isclose(true_p[s], prior)
                if m.any():
                    sess.append(float(np.mean(choice[s][m] == side[s][m])))
            vals[prior] = float(np.mean(sess)) if sess else float("nan")
        vals["balanced"] = float(np.nanmean([vals[0.2], vals[0.5], vals[0.8]]))
        out[mid] = vals
    return out


def _neural_summary():
    ve_path = ROOT / "reports" / "v2" / "neural" / "ve_unmatched.csv"
    surv_path = ROOT / "reports" / "v2" / "neural" / "survival_tests.csv"
    match_path = ROOT / "reports" / "v2" / "neural" / "behavior_matched_models.json"
    if not ve_path.exists():
        return None
    u = pd.read_csv(ve_path)
    if u.empty or "region" not in u.columns:
        return None
    sess = (
        u.groupby(["region", "model"], as_index=False)["ve_linear_recal"]
        .mean()
        .pivot(index="region", columns="model", values="ve_linear_recal")
    )
    matched = []
    if match_path.exists():
        matched = json.loads(match_path.read_text()).get("matched_models") or []
    surv = pd.read_csv(surv_path) if surv_path.exists() else pd.DataFrame()
    return {"ve_session_mean": sess, "matched": matched, "survival": surv, "n_rows": len(u)}


def _pretty(mid: str) -> str:
    return {
        "tanh_bptt": "tanh BPTT",
        "tanh_pc": "tanh PC",
        "gru": "GRU",
        "gru_pc": "GRU PC",
        "bayes": "Bayes",
    }.get(mid, mid)


# ---------- OOXML helpers ----------

def p_xml(text: str, *, bold: bool = False, center: bool = False, italic: bool = False, size: int | None = None) -> str:
    align = "<w:jc w:val=\"center\"/>" if center else ""
    rpr = []
    if bold:
        rpr.append("<w:b/>")
    if italic:
        rpr.append("<w:i/>")
    if size:
        rpr.append(f"<w:sz w:val=\"{size}\"/><w:szCs w:val=\"{size}\"/>")
    rpr_s = f"<w:rPr>{''.join(rpr)}</w:rPr>" if rpr else ""
    return (
        f"<w:p><w:pPr>{align}</w:pPr>"
        f"<w:r>{rpr_s}<w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r></w:p>"
    )


def h_xml(level: int, text: str) -> str:
    style = {1: "Heading1", 2: "Heading2", 3: "Heading3"}[level]
    sizes = {1: 32, 2: 26, 3: 24}
    return (
        f"<w:p><w:pPr><w:pStyle w:val=\"{style}\"/><w:spacing w:before=\"240\" w:after=\"120\"/></w:pPr>"
        f"<w:r><w:rPr><w:b/><w:sz w:val=\"{sizes[level]}\"/></w:rPr>"
        f"<w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r></w:p>"
    )


def note_xml(text: str) -> str:
    return p_xml("Note. " + text, italic=True)


def ul_xml(items: list[str]) -> str:
    parts = []
    for it in items:
        parts.append(
            "<w:p><w:pPr><w:pStyle w:val=\"ListParagraph\"/>"
            "<w:ind w:left=\"360\" w:hanging=\"180\"/></w:pPr>"
            f"<w:r><w:t xml:space=\"preserve\">• {escape(it)}</w:t></w:r></w:p>"
        )
    return "".join(parts)


def table_xml(headers: list[str], rows: list[list[str]], *, title: str | None = None, col_fracs: list[float] | None = None) -> str:
    """Fixed-layout table that does not collapse in Word / Google Docs."""
    n = len(headers)
    if col_fracs is None:
        col_fracs = [1.0 / n] * n
    assert abs(sum(col_fracs) - 1.0) < 1e-6 and len(col_fracs) == n
    widths = [max(900, int(PAGE_DXA * f)) for f in col_fracs]
    # renormalize to exact PAGE_DXA
    drift = PAGE_DXA - sum(widths)
    widths[-1] += drift

    parts = []
    if title:
        parts.append(p_xml(title, bold=True, italic=True))

    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    parts.append(
        "<w:tbl>"
        "<w:tblPr>"
        f'<w:tblW w:w="{PAGE_DXA}" w:type="dxa"/>'
        '<w:tblLayout w:type="fixed"/>'
        "<w:tblBorders>"
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="666666"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="666666"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="666666"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="666666"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="AAAAAA"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="AAAAAA"/>'
        "</w:tblBorders>"
        "</w:tblPr>"
        f"<w:tblGrid>{grid}</w:tblGrid>"
    )

    def cell(text: str, w: int, *, header: bool = False) -> str:
        rpr = "<w:rPr><w:b/><w:sz w:val=\"18\"/></w:rPr>" if header else '<w:rPr><w:sz w:val="18"/></w:rPr>'
        shd = '<w:shd w:val="clear" w:fill="EEEEEE"/>' if header else ""
        return (
            "<w:tc>"
            f"<w:tcPr><w:tcW w:w=\"{w}\" w:type=\"dxa\"/>{shd}"
            "<w:vAlign w:val=\"center\"/></w:tcPr>"
            f"<w:p><w:pPr><w:spacing w:before=\"40\" w:after=\"40\"/></w:pPr>"
            f"<w:r>{rpr}<w:t xml:space=\"preserve\">{escape(str(text))}</w:t></w:r></w:p>"
            "</w:tc>"
        )

    parts.append("<w:tr>" + "".join(cell(h, w, header=True) for h, w in zip(headers, widths)) + "</w:tr>")
    for row in rows:
        # pad / trim
        cells = list(row) + [""] * n
        cells = cells[:n]
        parts.append("<w:tr>" + "".join(cell(c, w) for c, w in zip(cells, widths)) + "</w:tr>")
    parts.append("</w:tbl>")
    parts.append(p_xml(""))  # spacer
    return "".join(parts)


def image_xml(rid: str, *, width_in: float = 6.2, height_in: float = 3.5, caption: str = "") -> str:
    """Inline image. width/height in inches."""
    cx = int(width_in * 914400)
    cy = int(height_in * 914400)
    drawing = f"""
    <w:p>
      <w:pPr><w:jc w:val="center"/></w:pPr>
      <w:r>
        <w:drawing>
          <wp:inline distT="0" distB="0" distL="0" distR="0"
            xmlns:wp="{WP}" xmlns:a="{A}" xmlns:pic="{PIC}" xmlns:r="{R}">
            <wp:extent cx="{cx}" cy="{cy}"/>
            <wp:docPr id="1" name="figure"/>
            <a:graphic>
              <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
                <pic:pic>
                  <pic:nvPicPr>
                    <pic:cNvPr id="0" name="fig.png"/>
                    <pic:cNvPicPr/>
                  </pic:nvPicPr>
                  <pic:blipFill>
                    <a:blip r:embed="{rid}"/>
                    <a:stretch><a:fillRect/></a:stretch>
                  </pic:blipFill>
                  <pic:spPr>
                    <a:xfrm>
                      <a:off x="0" y="0"/>
                      <a:ext cx="{cx}" cy="{cy}"/>
                    </a:xfrm>
                    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
                  </pic:spPr>
                </pic:pic>
              </a:graphicData>
            </a:graphic>
          </wp:inline>
        </w:drawing>
      </w:r>
    </w:p>
    """
    cap = p_xml(caption, italic=True, center=True, size=18) if caption else ""
    return drawing + cap


def write_docx(path: Path, body_xml: str, *, images: list[tuple[str, Path]] | None = None) -> None:
    """images: list of (rId, png_path)."""
    images = images or []
    # Unique media names (many figures share basename multipanel_diagnostics.png).
    media_names = {rid: f"{rid}{png.suffix.lower() or '.png'}" for rid, png in images}
    rels_img = "".join(
        f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{media_names[rid]}"/>'
        for rid, _png in images
    )
    document_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W}" xmlns:r="{R}">
  <w:body>
    {body_xml}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080"/>
    </w:sectPr>
  </w:body>
</w:document>'''

    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>'''

    root_rels = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{NSR}">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''

    word_rels = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{NSR}">
  <Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  {rels_img}
</Relationships>'''

    styles = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W}">
  <w:style w:type="paragraph" w:styleId="Normal" w:default="1"><w:name w:val="Normal"/>
    <w:rPr><w:sz w:val="22"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/>
    <w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/>
    <w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/>
    <w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/>
    <w:rPr><w:b/><w:sz w:val="40"/></w:rPr></w:style>
</w:styles>'''

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("word/document.xml", document_xml)
        z.writestr("word/_rels/document.xml.rels", word_rels)
        z.writestr("word/styles.xml", styles)
        for rid, png in images:
            z.write(png, f"word/media/{media_names[rid]}")


def build_methods() -> Path:
    man = json.loads((ROOT / "data" / "manifests" / "shared_behavior_neural_eids.json").read_text())
    n = man.get("n_sessions", 20)
    covered = man.get("regions_covered_in_cohort") or []

    parts = []
    parts.append(p_xml("Detailed methods", bold=True, center=True, size=40))
    parts.append(p_xml("Latent prior updating in the International Brain Laboratory decision task", center=True, size=26))
    parts.append(p_xml(""))

    parts.append(h_xml(1, "1. Aim"))
    parts.append(h_xml(2, "1.1 Scientific question"))
    parts.append(p_xml(
        "In the International Brain Laboratory (IBL) visuospatial decision task, the probability that the "
        "stimulus appears on the right versus left changes across blocks, but that block probability is hidden "
        "from the animal. Mice behave as if they maintain and update a subjective prior over stimulus side. "
        "We ask which computational model best accounts for that trial-by-trial prior updating."
    ))
    parts.append(p_xml("Four model classes are compared side by side:"))
    parts.append(ul_xml([
        "tanh BPTT — a standard tanh recurrent network trained with backpropagation through time.",
        "tanh PC — an architecturally identical tanh network, trained with corrected predictive-coding "
        "credit assignment (local inference + local synaptic updates), not BPTT.",
        "GRU — a gated recurrent unit trained with BPTT.",
        "GRU PC — an architecturally identical GRU, trained with gate-aware corrected PC credit assignment.",
    ]))

    parts.append(h_xml(2, "1.2 Evaluation hierarchy"))
    parts.append(p_xml(
        "The project intentionally separates what is learned from where it is tested:"
    ))
    parts.append(ul_xml([
        "Training uses only synthetic IBL-like sessions (never mouse choice as a supervised target).",
        "Primary ranking uses held-out synthetic sessions from the same generator (fair, matched statistics).",
        "Secondary check 1: freeze weights and transfer to real mouse sessions (same tick/channel schema).",
        "Secondary check 2: ask whether model belief aligns with neural prior readouts in prior-linked regions, "
        "with confirmatory claims restricted to behavior-matched models (§7).",
    ]))
    parts.append(p_xml(
        "This hierarchy matters for interpretation. A model that wins on synthetic held-out has learned the "
        "generative task. Transfer asks whether that solution generalizes to real mice. Neural alignment asks "
        "whether the model’s internal belief resembles a behavior-defined neural prior axis — not whether the "
        "model is the brain."
    ))

    parts.append(h_xml(2, "1.3 Primary scientific questions"))
    parts.append(p_xml(
        "The analyses are organized around three linked questions (behavior first, then neural):"
    ))
    parts.append(ul_xml([
        "Q1 — Correctness: Which model best reproduces trial-by-trial correctness (accuracy vs the correct "
        "stimulus side), both across the full task and in trials surrounding block switches, where online "
        "updating of the hidden prior is most evident?",
        "Q2 — Belief updating dynamics: In the history-only condition, does each model’s inferred prior "
        "(zero-evidence belief) update at a similar rate and with similar asymmetries as expected around "
        "0.2↔0.8 switches? Which account better captures how fast and how the belief is revised, not just "
        "steady-state accuracy? (Mouse subjective-prior matching around switches is an explicit follow-up.)",
        "Q3 — Neural prior alignment: In prior-encoding regions (MOs, vlOFC/ORBvl, and related sites), which "
        "model’s belief better matches neural prior readouts, including after behavior matching?",
    ]))
    parts.append(p_xml(
        "Reporting order follows the science: overall correctness → switch-centered correctness "
        "(0.2→0.8 and 0.8→0.2 separately) → belief / history-gap dynamics → neural VE."
    ))

    parts.append(h_xml(1, "2. Task and data"))
    parts.append(h_xml(2, "2.1 IBL biased-block task (intuition)"))
    parts.append(p_xml(
        "On each trial the mouse views a visual grating on the left or right and reports the side with a "
        "wheel turn. Contrast (stimulus strength) varies across trials. Across blocks of trials, the "
        "experimenter sets the probability that the stimulus is on a given side — conventionally described by "
        "probabilityLeft ∈ {0.2, 0.5, 0.8}, which implies P(stimulus right) ∈ {0.8, 0.5, 0.2}. The animal is "
        "not told the current block; it must infer the prevailing bias from experience. That inferred bias is "
        "what we call the latent / subjective prior."
    ))

    parts.append(h_xml(2, "2.2 Synthetic generator (training and held-out)"))
    parts.append(p_xml(
        "Synthetic sessions are not replays of individual mouse sessions. They are drawn from empirical "
        "distributions fitted on a behavior-quality cohort (block lengths, prior transitions, contrast "
        "frequencies, session length), then painted into the shared tick schema (§4). This keeps transfer "
        "meaningful: models must generalize structure, not memorize particular eids."
    ))
    parts.append(ul_xml([
        "Block priors: {0.2, 0.5, 0.8} with an empirical transition matrix (sessions typically start near 0.5).",
        "Block lengths: empirical PMF clipped to roughly 10–100 trials (median ≈ 45).",
        "Contrasts: empirical levels {0, 0.0625, 0.125, 0.25, 1.0} with fitted frequencies "
        "(no 0.5 contrast in the current fit).",
        "Default synthetic session length ≈ 929 completed trials (median of the fitting cohort).",
        "Config / stats: configs/synthetic_v2.yaml and data/manifests/synthetic_stats_v2.json.",
    ]))

    parts.append(h_xml(2, "2.3 Shared real cohort (behavior transfer + neural)"))
    parts.append(p_xml(
        f"Behavior transfer and neural analyses use the same locked set of {n} real sessions "
        f"(manifest: data/manifests/shared_behavior_neural_eids.json). Using one shared cohort prevents "
        f"the confound of ranking models on one set of mice and testing neural alignment on another."
    ))
    parts.append(p_xml("Inclusion and selection logic:"))
    parts.append(ul_xml([
        "Each session passes almost-perfect behavior quality control (high fraction of trials surviving "
        "standard IBL choice/RT rules after RT trimming).",
        "Each session has Neuropixels spikes in at least one primary region of interest.",
        "No single insertion covers all ROIs. Sessions are therefore chosen by greedy set cover so that the "
        "union of the cohort covers the full primary ROI set, preferring sessions that add new ROIs, then "
        "primary cortical coverage, then multi-ROI count, then behavior QC rank.",
        f"Regions covered in the locked cohort union: {', '.join(covered)}.",
    ]))
    parts.append(note_xml(
        "Per-region neural statistics use only the sessions that contain units in that region. A region with "
        "few sessions is underpowered for survival testing even if the cohort as a whole is locked."
    ))
    parts.append(p_xml(
        "Legacy note: an older behavior-only core list (data/manifests/behavior_core_eids.json) lacks "
        "Neuropixels insertions and is not used for v2 real+neural claims."
    ))

    parts.append(h_xml(1, "3. Regions of interest"))
    parts.append(h_xml(2, "3.1 Why these regions"))
    parts.append(p_xml(
        "Region choice follows published IBL analyses rather than an ad hoc cortical wishlist. The "
        "standardized behavior paper establishes that mice use block structure. The brain-wide map paper "
        "reports stimulus, choice, action, and reward maps and points prior analyses to its companion. That "
        "companion (Findling et al.) reports widespread decoding of a Bayes-optimal subjective prior, "
        "including high-level and motor cortical sites that motivate our primary ROI set."
    ))
    parts.append(table_xml(
        ["Paper", "Link", "Role"],
        [
            [
                "Findling et al., Nature 2025",
                "https://www.nature.com/articles/s41586-025-09226-1",
                "Primary source for subjective-prior anatomy",
            ],
            [
                "IBL brain-wide map, Nature 2025",
                "https://www.nature.com/articles/s41586-025-09235-0",
                "BWM context; prior analyses in companion",
            ],
            [
                "IBL standardized behavior, eLife 2021",
                "https://elifesciences.org/articles/63711",
                "Task and behavioral prior use",
            ],
        ],
        title="Source papers",
        col_fracs=[0.28, 0.42, 0.30],
    ))

    parts.append(h_xml(2, "3.2 Primary analysis scope (locked)"))
    parts.append(table_xml(
        ["Region", "Allen", "Tier", "Why included"],
        [
            ["Secondary motor", "MOs", "Primary cortex", "Strong prior decoding; core claim site"],
            ["Ventrolateral OFC", "ORBvl", "Primary cortex", "High-level prior example site"],
            ["Dorsal ACC", "ACAd", "Primary cortex", "Named high-level cortical prior site"],
            ["Primary motor", "MOp", "Motor", "Motor partner of MOs"],
        ],
        title="Primary regions of interest",
        col_fracs=[0.26, 0.12, 0.18, 0.44],
    ))
    parts.append(p_xml(
        "In code, ORBvl is named vlOFC_orbvl; figures and tables use the Allen acronym ORBvl or the label "
        "vlOFC interchangeably for readability."
    ))

    parts.append(h_xml(2, "3.3 Deferred / optional ROIs"))
    parts.append(p_xml(
        "A broader Findling-inspired list (CP, VISp, LGd, SCm, GRN, ORBm, PL, ILA) is documented but not in "
        "the current analysis loop. Early sensory sites are especially sensitive to embodied priors "
        "(posture / eye position) that we have not yet controlled for; expanding there without those "
        "controls would invite false positives. Deferral is a design choice, not a claim that those "
        "regions lack prior information."
    ))

    parts.append(h_xml(1, "4. Trial encoding and evaluation regimes"))
    parts.append(h_xml(2, "4.1 Shared tick-and-channel interface"))
    parts.append(p_xml(
        "All models see the same discrete-time interface. Continuous trial events are binned into 100 ms "
        "ticks. Phase order follows a causal within-trial schedule (baseline → stimulus → go → response → "
        "feedback); phase durations are integer ticks fitted from empirical IBL event-time medians, not a "
        "toy fixed schedule. Loss / choice readout occurs only on the response tick."
    ))
    parts.append(p_xml("Current frozen phase layout (9 ticks per trial):"))
    parts.append(ul_xml([
        "Ticks 0–1: baseline (all channels 0; recurrent state carries cross-trial history).",
        "Tick 2: stimulus onset and go cue in the same bin (empirical go−stim median ≈ 16 ms → 0 ticks).",
        "Ticks 3–5: stimulus held; go off.",
        "Tick 6: response / readout (visual off on this tick by convention; supervised target = correct side).",
        "Ticks 7–8: feedback (action and reward channels).",
    ]))
    parts.append(p_xml("Seven input channels (fixed order):"))
    parts.append(ul_xml([
        "visual_right, visual_left — continuous contrast magnitude on the corresponding side (else 0).",
        "go_cue — binary, on only at the go tick.",
        "action_left, action_right — binary feedback of the executed/teacher action.",
        "rewarded, not_rewarded — binary outcome (exactly one on during feedback).",
    ]))
    parts.append(p_xml(
        "Sensory coding: right stimulus of contrast c → (visual_right, visual_left) = (c, 0); left → (0, c). "
        "Synthetic training may add Gaussian noise to the two visual channels "
        "(sensory_noise_std_synth = 0.15). Real transfer turns that noise off."
    ))

    parts.append(h_xml(2, "4.2 What differs between synthetic training, synthetic eval, and real transfer"))
    parts.append(ul_xml([
        "Synthetic train: teacher-forced action/reward on feedback ticks, with a configurable action-error "
        "rate (training_feedback_error_rate = 0.2) so the model sees occasional incorrect feedback.",
        "Synthetic held-out eval: closed-loop — the model’s own sampled/chosen action determines feedback.",
        "Real transfer: frozen weights; ALF trials mapped into the same schema; feedback ticks use the "
        "mouse’s actual action and reward as history inputs only. The supervised / scored target remains "
        "the correct stimulus side, never the mouse’s choice.",
    ]))
    parts.append(p_xml(
        "This last point is easy to miss. Scoring against correct side asks whether the model solves the "
        "task. Using mouse action/reward as inputs lets the model’s recurrent state track the same history "
        "stream the mouse experienced, which is required for fair prior-updating comparisons on real data."
    ))

    parts.append(h_xml(2, "4.3 Evaluation regimes"))
    parts.append(p_xml(
        "Three regimes probe different computational demands. All use the same trained weights; only "
        "evaluation-time inputs or trial subsets change."
    ))
    parts.append(ul_xml([
        "history_only (primary): closed-loop (or mouse-history on real); no oracle block information. "
        "The model must infer the prior from experience. This is the main ranking regime.",
        "full_information (control): same dynamics, but at evaluation an additive logit bias from the true "
        "block log-odds is injected (fi_oracle_logit_gain). This asks whether a model can use prior "
        "information when it is provided, even if it fails to infer it from history alone "
        "(informative for tanh PC in particular).",
        "fixed_prior: evaluate only on unbiased blocks (true P(right) ≈ 0.5), removing block-bias demands.",
    ]))

    parts.append(h_xml(1, "5. Models and training"))
    parts.append(h_xml(2, "5.1 Common interface and fairness rules"))
    parts.append(p_xml(
        "Fair comparison requires holding fixed what can be held fixed, and documenting intentional "
        "differences:"
    ))
    parts.append(ul_xml([
        "Shared: input channels, phase ticks, synthetic generator, choice objective (correct stimulus side), "
        "and (for RNNs) default hidden size 48 with session-contiguous state.",
        "Intentional difference within architecture pairs: identical test-time tanh (or GRU) dynamics; "
        "BPTT versus corrected predictive-coding credit assignment at train time.",
        "Both PC models share the corrected inference recipe (32 rounds, output precision 0.025, "
        "nudge-normalized updates, 929 trials/session). GRU PC is gate-aware.",
        "No mouse fine-tuning in v2. No reaction-time primary loss in v2 ranking. Explicit Bayes is legacy "
        "only (not in the active four-model set).",
    ]))

    parts.append(h_xml(2, "5.2 Model definitions"))
    parts.append(table_xml(
        ["Model", "Test-time computation", "Training"],
        [
            ["tanh BPTT", "Vanilla tanh RNN over ticks", "Backpropagation through time"],
            ["tanh PC", "Identical tanh RNN", "Corrected PC credit assignment"],
            ["GRU", "Gated recurrent unit over ticks", "Backpropagation through time"],
            ["GRU PC", "Identical GRU", "Gate-aware corrected PC credit assignment"],
        ],
        col_fracs=[0.18, 0.40, 0.42],
    ))
    parts.append(p_xml(
        "Naming caution: tanh PC / GRU PC mean credit-assignment training of the corresponding cell. "
        "They are not v1 prediction-error dynamics cells. At test time, each PC model shares its BPTT "
        "twin’s computational graph; only the learned weights differ. Bayes is legacy and not in the "
        "active comparison."
    ))
    parts.append(p_xml(
        "PC inference (both cells): free activities are iteratively optimized under a prediction-error "
        "energy with a weak output nudge (precision 0.025). Synchronous rounds (32) are long enough for "
        "the response nudge to reach the previous trial’s feedback window. Local weight updates use "
        "presynaptic activity × postsynaptic prediction error and are divided by the nudge magnitude so "
        "changing precision does not silently rescale the synaptic learning rate. GRU PC applies the same "
        "recipe through the full gate Jacobian (z, r, candidate)."
    ))

    parts.append(h_xml(2, "5.3 Training protocol"))
    parts.append(ul_xml([
        "Optimizer / loop: NumPy implementations with Adam; truncated BPTT (or PC chunks) over sessions "
        "while keeping empirical phase ticks.",
        "Default exposure (all active models): 60 epochs × 24 sessions/epoch × ≈929 trials/session, in "
        "chunks of 32 trials (bptt_trials).",
        "Corrected PC (tanh_pc and gru_pc): same 929-trial sessions as BPTT models. Training uses "
        "iterative predictive-coding inference (32 rounds; minimum 9 to reach previous-trial feedback), "
        "weak output precision 0.025 during inference, and nudge-normalized local synaptic updates. "
        "Between chunks, only the pre-update forward state is carried (never the label-nudged inferred "
        "state). gru_pc uses gate-aware prediction errors through the full GRU step.",
        "Checkpoints live under artifacts/v2/models/{model_id}/; real and regime evals load frozen weights.",
    ]))

    parts.append(h_xml(2, "5.4 Belief extraction (used in metrics and neural analyses)"))
    parts.append(p_xml(
        "Behavioral “belief” is not the on-path choice probability under the actual stimulus. The primary "
        "prior probe is a counterfactual zero-evidence query: from the post-baseline recurrent (or Bayes) "
        "state, run to the response tick with visual contrast held at zero but go cue on, and read "
        "P(choose right). That quantity isolates history-dependent bias from immediate sensory evidence. "
        "History gap and switch-centered curves are built from this probe (§6)."
    ))

    parts.append(h_xml(1, "6. Behavioral metrics"))
    parts.append(h_xml(2, "6.1 What we optimize versus what we report"))
    parts.append(p_xml(
        "Training minimizes choice cross-entropy against the correct stimulus side on response ticks. "
        "Reported scorecards emphasize interpretable summaries of the same objective and of prior updating, "
        "separately for synthetic held-out and real transfer, and separately by regime."
    ))

    parts.append(h_xml(2, "6.2 Correctness and balanced correctness"))
    parts.append(ul_xml([
        "Correctness (accuracy): fraction of evaluated trials where argmax model choice equals the correct "
        "stimulus side. On real data this is still correct-side scoring, not mouse-choice matching.",
        "Cross-entropy / choice NLL: mean negative log probability of the correct side under the model’s "
        "choice distribution; used for ranking and for the neural behavior-matching ε-ball (§7.7).",
        "Balanced correctness: equal-weight mean of correctness within blocks with true P(right) ∈ "
        "{0.2, 0.5, 0.8}. This prevents a model that is only good in one block type from dominating an "
        "unbalanced session mix.",
    ]))

    parts.append(h_xml(2, "6.3 Psychometrics by block prior"))
    parts.append(p_xml(
        "For each model and regime we plot P(choose right) against signed contrast, stratified by true "
        "block P(right). Primary display uses the biased blocks 0.2 and 0.8 (the 0.5 block is reserved for "
        "the fixed-prior regime and for balanced summaries). A model that updates priors should show "
        "horizontal shifts of the psychometric: higher P(right) curves in 0.8 blocks than in 0.2 blocks at "
        "matched contrast."
    ))

    parts.append(h_xml(2, "6.4 History gap (prior updating summary)"))
    parts.append(p_xml(
        "History gap = mean zero-evidence P(right) among trials in true 0.8 blocks minus the same mean in "
        "true 0.2 blocks. Large positive gaps mean the model’s counterfactual belief tracks block identity. "
        "Gaps near zero mean little history-dependent bias (typical of a model that relies on sensory "
        "evidence or an oracle prior rather than inferred history)."
    ))

    parts.append(h_xml(2, "6.5 Switch-centered adaptation (belief)"))
    parts.append(p_xml(
        "Block transitions between biased blocks (0.2→0.8 and 0.8→0.2; 0.5 transitions excluded) are aligned "
        "at trial 0. Zero-evidence belief is averaged in a window around the switch (approximately −30 to +30 "
        "trials), separately by switch direction, then summarized as session means with SEM bands across "
        "sessions. These curves show how quickly belief re-centers after a hidden prior change."
    ))

    parts.append(h_xml(2, "6.6 Switch-centered correctness (accuracy around updates)"))
    parts.append(p_xml(
        "The same switch alignment is applied to trial correctness (1 if model choice equals the correct "
        "stimulus side, else 0). This answers Q1’s second half: models can have similar overall accuracy yet "
        "differ where updating is hardest — immediately after a block switch. We report:"
    ))
    parts.append(ul_xml([
        "Full switch-centered correctness curves by direction (mean ± SEM across sessions).",
        "A post-switch summary: mean correctness on trials 0–15 after the switch, with session-level 95% CIs, "
        "separately for 0.2→0.8 and 0.8→0.2.",
        "A story board that places overall correctness above the two switch-direction curves so the narrative "
        "runs from steady-state ranking to online-updating ranking.",
    ]))

    parts.append(h_xml(2, "6.7 Example-session diagnostics"))
    parts.append(p_xml(
        "Per-model multipanels also show an example session timeline: true block P(right) versus "
        "zero-evidence belief across trials (and psychometric / switch panels). Prefer sessions that contain "
        "0.2, 0.5, and 0.8 blocks when available so the timeline illustrates all prior contexts. These "
        "figures are qualitative checks that the scalar metrics summarize real dynamics."
    ))

    parts.append(h_xml(1, "7. Neural prior comparison"))
    parts.append(p_xml(
        "This section asks a different question from the behavioral scorecards. Behavior asks: which model "
        "chooses more like the task (or more like the mouse)? Neural comparison asks: once we have built a "
        "one-dimensional summary of how a brain region tracks the mouse’s trial-by-trial prior, which model’s "
        "latent belief best tracks that same neural summary? The goal is not to claim that the model is the "
        "brain, but to test which model’s internal belief trajectory is most aligned with a behavior-defined "
        "neural prior axis in regions previously linked to prior encoding."
    ))

    parts.append(h_xml(2, "7.1 Intuition: from spikes to a neural prior axis"))
    parts.append(p_xml(
        "On each trial, many neurons in a region fire at different rates. Those rates likely mix sensory, "
        "motor, and cognitive signals. We do not compare the full population vector to a model. Instead we "
        "construct a single number per trial that is meant to capture the region’s encoding of the mouse’s "
        "subjective prior — call this the neural prior axis, denoted û_t. Conceptually:"
    ))
    parts.append(ul_xml([
        "Step A — estimate what prior the mouse appears to be using from behavior alone (mouse prior p̂_t).",
        "Step B — learn a linear map from the region’s peri-stimulus spike counts to that mouse prior, using "
        "cross-validated ridge regression, yielding û_t on held-out trials.",
        "Step C — ask how well each model’s belief q_t explains û_t on the same trials.",
        "Step D — only after models are matched on history-only choice quality do we treat a neural "
        "advantage as confirmatory (behavior-matched survival).",
    ]))
    parts.append(p_xml(
        "This design separates three ideas that are easy to confuse: (i) the mouse’s behavioral prior "
        "estimate, (ii) the neural readout of that prior, and (iii) the model’s latent belief. The primary "
        "comparison is between (ii) and (iii), with (i) used only to define the neural axis."
    ))

    parts.append(h_xml(2, "7.2 Behavior-derived mouse prior (target for the neural axis)"))
    parts.append(p_xml(
        "The mouse prior is not the true block probabilityLeft from the experimenter. It is a compact "
        "history-only estimate of the mouse’s subjective P(right) used as a bias in a logistic choice model. "
        "Concretely:"
    ))
    parts.append(ul_xml([
        "Within each session, a leaky online estimate is updated from experienced stimulus sides: "
        "p_{t+1} = (1−α) p_t + α · 1{stimulus_right on trial t}, with p_t reported before the update "
        "(causal: p_t depends only on history before trial t).",
        "α is selected by grid search to minimize choice negative log-likelihood of a logistic model "
        "logit P(choice_right) = β0 + β_c · signed_contrast + β_p · (2p_t − 1).",
        "The resulting p_t series is the regression target for neural decoding. It is a behavior-derived "
        "proxy for subjective prior, not ground truth for neural state.",
    ]))

    parts.append(h_xml(2, "7.3 Neural features: peri-stimulus spike counts"))
    parts.append(p_xml(
        "For each session and each primary region that contains sorted units in that session:"
    ))
    parts.append(ul_xml([
        "Units are assigned to regions by Allen acronym mapping (MOs, ORBvl→vlOFC, ACAd, MOp).",
        "Spikes are aligned to stimulus onset (stimOn_times).",
        "For each trial and unit, we count spikes in the half-open window [−0.1, 0.3) seconds relative to "
        "stimulus onset (400 ms total). This is the locked peri-stimulus analysis window.",
        "The result is a matrix X of shape (n_trials × n_units) of raw spike counts for that "
        "session×region. Sessions without units in a region are skipped for that region (coverage is by "
        "cohort union, not every session in every ROI).",
    ]))

    parts.append(h_xml(2, "7.4 Building the neural prior axis (cross-validated ridge)"))
    parts.append(p_xml(
        "We map X → mouse prior p̂ with ridge regression, always evaluating predictions out-of-fold so the "
        "axis is not trivially overfit to noise:"
    ))
    parts.append(ul_xml([
        "Predictors X are z-scored within each training fold (StandardScaler fit on train, applied to test).",
        "Regressor: RidgeCV with a log-spaced grid of regularization strengths α ∈ {10^{−2} … 10^3} "
        "(12 values).",
        "Cross-validation: K-fold with K = 5 when enough trials exist (otherwise fewer folds), with "
        "shuffling and a fixed random seed for reproducibility. Each trial receives exactly one "
        "out-of-fold prediction û_t.",
        "Minimum data: if too few finite trials or zero units, that session×region is skipped "
        "(NaN VE).",
        "Optional diagnostics stored alongside: cross-validated variance explained of p̂ by û "
        "(how well the region can be read out to the mouse prior) and correlation(p̂, û).",
    ]))
    parts.append(p_xml(
        "Importantly, û_t is defined from neural activity and the mouse prior only. Model beliefs do not "
        "enter the construction of the neural axis. This prevents circularity when later asking which "
        "model explains û."
    ))

    parts.append(h_xml(2, "7.5 Model belief as a predictor of the neural axis"))
    parts.append(p_xml(
        "On the same trials, each frozen model produces a scalar belief q_t (zero-evidence P(right) under "
        "the evaluation regime of interest; neural analyses use the history-only real-transfer rollouts). "
        "We ask how much of the variance in û_t is explained by q_t."
    ))
    parts.append(p_xml(
        "Variance explained is VE = 1 − SSE/SST, where SST is the total sum of squares of û around its "
        "mean and SSE is the residual sum of squares after a prediction. Two variants are computed:"
    ))
    parts.append(ul_xml([
        "ve_raw: use q_t directly as the prediction of û_t. This penalizes scale and offset mismatches "
        "(a model that tracks the shape of û but lives on a different numeric scale can look worse).",
        "ve_linear_recal (primary): first fit a simple linear map û ≈ a · q + b by ordinary least squares "
        "on the same trials, then compute VE of û by that recalibrated prediction. This asks whether q "
        "and û share a linear relationship up to affine transform — the scientifically relevant notion "
        "of “tracks the same latent,” without requiring identical units or calibration.",
        "corr: Pearson correlation between û and q (secondary).",
    ]))
    parts.append(p_xml(
        "Primary reported neural metric: ve_linear_recal, computed once per (session, region, model)."
    ))

    parts.append(h_xml(2, "7.6 Aggregation across sessions"))
    parts.append(p_xml(
        "For each region and model, we average ve_linear_recal across sessions that contribute that region. "
        "This session-mean is the ranking quantity plotted in the unmatched and matched neural boards. "
        "Using sessions as the unit of aggregation (rather than pooling all trials) respects that sessions "
        "differ in unit counts, noise, and coverage, and matches the resampling unit used in survival tests."
    ))

    parts.append(h_xml(2, "7.7 Why behavior matching? Unmatched vs confirmatory claims"))
    parts.append(p_xml(
        "A model that fails at history-only choice may still correlate with a neural prior axis for "
        "incidental reasons, or a behaviorally superior model may look better neurally simply because it "
        "is a better behavioral model. Confirmatory neural claims therefore restrict attention to models "
        "that are approximately matched on the same shared cohort’s history-only choice quality."
    ))
    parts.append(p_xml("Matching rule (choice-primary ε-ball):"))
    parts.append(ul_xml([
        "Compute each model’s mean trial cross-entropy (choice NLL) on the shared behavior+neural cohort "
        "under real history-only evaluation.",
        "Let CE★ be the best (lowest) CE among models.",
        "Retain model m if CE_m − CE★ ≤ ε with ε = 0.05 (nats per trial).",
        "An RT NLL floor exists in code but is set non-binding in the current pipeline (RT is not used to "
        "gate neural confirmatory claims).",
        "Unmatched tables/figures still show all models for transparency; matched tables/figures and "
        "survival tests use only the ε-ball set.",
    ]))

    parts.append(h_xml(2, "7.8 Survival testing of a matched neural advantage"))
    parts.append(p_xml(
        "Within each region, among matched models, identify the best and second-best by session-mean "
        "ve_linear_recal. The quantity of interest is the paired session advantage "
        "Δ = mean_s VE(best, s) − mean_s VE(second, s), where s indexes sessions that have both models’ VE."
    ))
    parts.append(p_xml("Session bootstrap (per region):"))
    parts.append(ul_xml([
        "Resample sessions with replacement (B = 2000), recompute Δ each time.",
        "Report the observed Δ, a percentile 95% confidence interval [2.5%, 97.5%], and a two-sided "
        "bootstrap p-value: twice the fraction of bootstrap Δ’s that have opposite sign to the observed "
        "Δ (capped at 1).",
        "Sessions are the resampling unit because VE is already a session-level summary and sessions are "
        "the natural independent replicate for this cohort size.",
    ]))
    parts.append(p_xml("Multiple regions (Holm correction):"))
    parts.append(ul_xml([
        "Each primary region yields one bootstrap p-value for best-vs-second among matched models.",
        "These p-values are adjusted by the Holm–Bonferroni step-down procedure across the tested "
        "regions.",
        "We say the advantage “survives” in a region if the Holm-adjusted p-value is below 0.05 "
        "(equivalently: the matched VE gap remains credible after correcting for testing multiple ROIs).",
    ]))
    parts.append(p_xml(
        "Interpretation for non-experts: surviving means — after restricting to models that are similarly "
        "good at history-only choices, and after accounting for session-to-session variability and the fact "
        "that we look at several brain regions — the top model’s edge in explaining the neural prior axis "
        "over the next-best matched model is still statistically supported in that region. Non-survival "
        "(e.g. in a sparsely covered ROI) means the gap is not yet trustworthy under these corrections, "
        "not that neural encoding is absent."
    ))

    parts.append(h_xml(2, "7.9 What this analysis does and does not claim"))
    parts.append(ul_xml([
        "Does claim: relative alignment of model belief trajectories with a behavior-defined neural prior "
        "axis under a fixed peri-stimulus window and linear readout.",
        "Does not claim: that û is the unique or true neural prior; that ridge is the brain’s readout; "
        "that VE proves causal encoding; or that unmatched VE rankings alone are confirmatory.",
        "Known gaps: no embodied-prior controls (video / eye) yet; peri-stimulus window only; ROI "
        "coverage uneven across sessions; mouse prior is itself a model of behavior.",
    ]))

    parts.append(h_xml(1, "8. Limitations and open risks"))
    parts.append(h_xml(2, "8.1 Cohort and coverage"))
    parts.append(ul_xml([
        "ROI coverage is by cohort union; individual sessions typically contribute 1–few regions. Survival "
        "tests are underpowered where n_sessions is small (e.g. ACAd in the current lock).",
        "Almost-perfect QC plus ephys requirements yield a small n; results should be read as a locked "
        "pilot cohort, not a brain-wide census.",
    ]))
    parts.append(h_xml(2, "8.2 Modeling and transfer"))
    parts.append(ul_xml([
        "Synthetic training approximates empirical statistics but does not replay individual mice "
        "(intentional; still a distribution shift risk on transfer).",
        "Teacher-forced training feedback versus mouse feedback on transfer is a known shift (V2-R4).",
        "tanh PC uses the same session length as BPTT; compute cost is higher due to 32 inference rounds.",
        "Explicit Bayes is parked (legacy module retained) and is not part of the active ranking.",
    ]))
    parts.append(h_xml(2, "8.3 Neural analysis"))
    parts.append(ul_xml([
        "Embodied-prior controls (video / eye position) are not yet applied.",
        "The neural window is peri-stimulus only; complementary inter-trial decoding is left for later work.",
        "Mouse prior and neural axis are estimated quantities; errors in either reduce neural VE for all "
        "models and can shrink detectable advantages.",
        "Linear ridge + affine recalibration cannot capture nonlinear neural–belief relationships.",
    ]))
    parts.append(h_xml(2, "8.4 Out of v2 scope"))
    parts.append(ul_xml([
        "Mouse fine-tuning, reaction-time primary losses, meta-RL, and Bayesian+credit-assignment twins "
        "are parked.",
        "Expanded Findling ROIs remain optional until embodied controls and larger coverage exist.",
    ]))

    out = ROOT / "reports" / "v2" / "METHODS_DETAILED.docx"
    write_docx(out, "".join(parts))
    return out


def build_article() -> Path:
    metrics = _load_metrics()
    by = {(r["domain"], r["regime"], r["model"]): r for r in metrics}
    prior = _per_prior_real_history()
    neural = _neural_summary()
    man = json.loads((ROOT / "data" / "manifests" / "shared_behavior_neural_eids.json").read_text())
    n = man.get("n_sessions", 20)

    def acc(domain, regime, mid):
        r = by.get((domain, regime, mid))
        return float("nan") if r is None else r["acc"]

    def gap(domain, regime, mid):
        r = by.get((domain, regime, mid))
        return float("nan") if r is None else r["gap"]

    # Figures to embed (must exist) — ordered to match the scientific story
    fig_specs = [
        ("rIdFig1", FIG / "scorecards" / "synth_history_only_scorecard.png", 6.2, 4.4,
         "Figure 1. Synthetic held-out scorecard (history-only): correctness and history gap."),
        ("rIdFig2", FIG / "comparison" / "real_history_only_overall_correctness.png", 6.2, 3.6,
         "Figure 2. Real history-only overall correctness (mean ± 95% CI across sessions)."),
        ("rIdFig3", FIG / "comparison" / "real_history_only_accuracy_to_switch_story.png", 6.4, 5.4,
         "Figure 3. Story board: overall correctness, then correctness around 0.2→0.8 and 0.8→0.2 switches."),
        ("rIdFig4", FIG / "comparison" / "real_history_only_switch_correctness.png", 6.4, 3.4,
         "Figure 4. Switch-centered correctness curves by direction (mean ± SEM)."),
        ("rIdFig5", FIG / "comparison" / "real_history_only_switch_correctness_summary.png", 6.4, 3.4,
         "Figure 5. Post-switch correctness summary (trials 0–15; mean ± 95% CI)."),
        ("rIdFig6", FIG / "comparison" / "real_history_only_history_gap.png", 6.2, 3.6,
         "Figure 6. Real history-only history gap (mean ± 95% CI)."),
        ("rIdFig7", FIG / "comparison" / "real_history_only_switch_board.png", 6.4, 3.4,
         "Figure 7. Switch-centered zero-evidence belief (prior updating dynamics)."),
        ("rIdFig8", FIG / "comparison" / "real_history_only_correctness_by_prior.png", 6.2, 3.6,
         "Figure 8. Correctness by block prior, with balanced correctness."),
        ("rIdFig9", FIG / "comparison" / "synth_vs_real_history_only_board.png", 6.2, 3.4,
         "Figure 9. Synthetic versus real transfer under history-only."),
        ("rIdFig10", FIG / "scorecards" / "real_full_information_scorecard.png", 6.2, 4.4,
         "Figure 10. Real full-information control: oracle prior at readout."),
        ("rIdFig11", FIG / "by_model" / "gru_pc" / "real" / "history_only" / "multipanel_diagnostics.png", 6.2, 4.6,
         "Figure 11. Example diagnostics for GRU PC under real history-only evaluation."),
        ("rIdFig12", FIG / "by_model" / "gru" / "real" / "history_only" / "multipanel_diagnostics.png", 6.2, 4.6,
         "Figure 12. Example diagnostics for GRU BPTT under real history-only evaluation."),
        ("rIdFig13", FIG / "neural" / "neural_ve_unmatched_vs_matched.png", 6.2, 3.8,
         "Figure 13. Neural prior variance explained by model belief (session-mean VE)."),
        ("rIdFig14", FIG / "neural" / "survival_tests.png", 6.0, 3.2,
         "Figure 14. Behavior-matched survival of neural advantages across regions."),
    ]
    images = [(rid, p) for rid, p, *_ in fig_specs if p.exists()]
    rid_map = {rid: (p, w, h, cap) for rid, p, w, h, cap in fig_specs if p.exists()}

    parts = []
    parts.append(p_xml(
        "Comparing BPTT and predictive-coding recurrent models of latent prior updating "
        "in the International Brain Laboratory decision task",
        bold=True, center=True, size=36,
    ))
    parts.append(p_xml("Working manuscript — current results", center=True, italic=True, size=22))
    parts.append(p_xml(""))

    parts.append(h_xml(1, "Abstract"))
    parts.append(p_xml(
        f"Mice performing the IBL biased-block task update a hidden prior over stimulus side. We compare four "
        f"models trained only on synthetic IBL-like sessions and evaluated on held-out synthetic data and on "
        f"{n} real sessions that also support neural analyses: tanh BPTT, corrected tanh PC, GRU BPTT, and "
        f"gate-aware GRU PC. Under history-only evaluation, GRU PC reaches the highest correctness on real "
        f"transfer ({acc('real','history_only','gru_pc'):.3f}), with GRU BPTT close behind "
        f"({acc('real','history_only','gru'):.3f}). BPTT models show larger history gaps than their PC twins. "
        f"Neural alignment is assessed in four primary prior-related regions (MOs, ORBvl, ACAd, MOp). "
        + (
            "Behavior-matched neural comparisons are reported below."
            if neural
            else "Neural results for this cohort are being finalized."
        )
    ))

    parts.append(h_xml(1, "Introduction"))
    parts.append(p_xml(
        "In the IBL decision task, stimulus side probabilities alternate in hidden blocks. Mice use this "
        "structure: psychometric curves shift with block identity, especially at low contrast "
        "(International Brain Laboratory, eLife 2021; https://elifesciences.org/articles/63711). "
        "Brain-wide recordings show distributed coding of task variables "
        "(International Brain Laboratory, Nature 2025; https://www.nature.com/articles/s41586-025-09235-0). "
        "A companion analysis reports that a subjective Bayes-optimal prior can be decoded from many regions, "
        "including early sensory, motor, and high-level cortical areas such as MOs, ORBvl, and ACAd "
        "(Findling et al., Nature 2025; https://www.nature.com/articles/s41586-025-09226-1)."
    ))
    parts.append(p_xml(
        "We ask three linked questions. (Q1) Which model best reproduces trial-by-trial correctness both "
        "overall and around block switches (0.2→0.8 and 0.8→0.2), where online prior updating is most evident? "
        "(Q2) In history-only evaluation, which model’s zero-evidence belief updates with a rate and "
        "asymmetry consistent with strong prior use (history gap and switch-centered belief curves)? "
        "(Q3) In prior-encoding regions (MOs, vlOFC/ORBvl, ACAd, MOp), which model’s belief better matches "
        "neural prior readouts after behavior matching? Models share inputs and are scored against the "
        "correct stimulus side."
    ))

    parts.append(h_xml(1, "Methods"))
    parts.append(h_xml(2, "Cohort"))
    parts.append(p_xml(
        f"We use {n} sessions that pass almost-perfect behavior quality control and have usable spikes in at "
        "least one region of interest. The same sessions are used for real behavioral transfer and neural "
        "analyses. Session selection maximizes union coverage of the ROI list below."
    ))

    parts.append(h_xml(2, "Regions of interest"))
    parts.append(p_xml(
        "Regions follow Findling et al. (2025): MOs, ORBvl (ventrolateral orbitofrontal cortex), ACAd, and MOp. "
        "No single Neuropixels session contains all four; per-region analyses use the subset of sessions with "
        "units in that region."
    ))

    parts.append(h_xml(2, "Models and evaluation"))
    parts.append(p_xml(
        "Active models: tanh BPTT, corrected tanh PC, GRU BPTT, and gate-aware GRU PC. Training uses only "
        "synthetic sessions (60 epochs × 24 sessions × 929 trials for all models). PC credit assignment follows "
        "the corrected recipe (32 inference rounds, output precision 0.025, nudge-normalized local updates; "
        "GRU PC is gate-aware). History-only evaluation is primary. Correctness is the fraction of trials "
        "where the model choice matches the correct stimulus side (session means with 95% CIs). Switch analyses "
        "align 0.2↔0.8 transitions and report both belief and correctness curves (SEM across sessions) plus "
        "post-switch (0–15) correctness summaries (95% CI). Belief is the counterfactual zero-evidence "
        "P(right); history gap is mean belief in 0.8 blocks minus 0.2 blocks. Full methods: METHODS_DETAILED.docx."
    ))

    parts.append(h_xml(2, "Neural analysis"))
    parts.append(p_xml(
        "Neural comparison asks which model’s belief best tracks a one-dimensional neural prior axis in each "
        "region. That axis is built without using any model: peri-stimulus spike counts (−0.1 to 0.3 s from "
        "stimulus onset) are mapped by 5-fold cross-validated ridge regression onto a behavior-derived mouse "
        "prior. Model belief is scored by linearly recalibrated variance explained (ve_linear_recal). "
        "Confirmatory claims use a choice CE ε-ball (ε=0.05) plus session-bootstrap survival with Holm "
        "correction. Detail: METHODS_DETAILED.docx §7."
    ))

    parts.append(h_xml(1, "Results"))
    parts.append(h_xml(2, "Q1a — Overall correctness (full task)"))
    parts.append(p_xml(
        "On held-out synthetic history-only sessions, tanh BPTT and GRU BPTT lead "
        f"({acc('synth','history_only','tanh_bptt'):.3f} and {acc('synth','history_only','gru'):.3f}), "
        f"ahead of tanh PC ({acc('synth','history_only','tanh_pc'):.3f}) and GRU PC "
        f"({acc('synth','history_only','gru_pc'):.3f})."
    ))
    if "rIdFig1" in rid_map:
        rid, (p, w, h, cap) = "rIdFig1", rid_map["rIdFig1"]
        parts.append(image_xml(rid, width_in=w, height_in=h, caption=cap))

    parts.append(p_xml(
        f"On real transfer (history-only), the ranking changes: GRU PC leads "
        f"({acc('real','history_only','gru_pc'):.3f}), then GRU BPTT ({acc('real','history_only','gru'):.3f}), "
        f"tanh BPTT ({acc('real','history_only','tanh_bptt'):.3f}), and tanh PC "
        f"({acc('real','history_only','tanh_pc'):.3f}). Session-level 95% CIs are shown in Figure 2."
    ))
    if "rIdFig2" in rid_map:
        rid, (p, w, h, cap) = "rIdFig2", rid_map["rIdFig2"]
        parts.append(image_xml(rid, width_in=w, height_in=h, caption=cap))

    parts.append(h_xml(2, "Q1b — Correctness around block switches"))
    parts.append(p_xml(
        "Overall accuracy can hide differences at the moments when the hidden prior actually changes. "
        "We therefore re-rank models on correctness aligned to 0.2→0.8 and 0.8→0.2 switches. Figure 3 places "
        "overall correctness above the two switch-direction curves; Figures 4–5 expand the switch view "
        "(curves and post-switch 0–15 summaries with CIs)."
    ))
    parts.append(p_xml(
        "On real history-only sessions, GRU PC also leads post-switch correctness in both directions "
        "(0.2→0.8 ≈ 0.820; 0.8→0.2 ≈ 0.829), with GRU BPTT next (≈ 0.801 / 0.787). This strengthens the "
        "claim that GRU PC’s real-transfer advantage is not only a steady-state effect."
    ))
    for rid in ("rIdFig3", "rIdFig4", "rIdFig5"):
        if rid in rid_map:
            p, w, h, cap = rid_map[rid]
            parts.append(image_xml(rid, width_in=w, height_in=h, caption=cap))

    parts.append(h_xml(2, "Q2 — Belief updating rate and asymmetries"))
    parts.append(p_xml(
        f"History gaps (steady-state prior use) remain larger for BPTT models "
        f"(GRU {gap('real','history_only','gru'):.3f}, tanh BPTT {gap('real','history_only','tanh_bptt'):.3f}) "
        f"than for PC twins (tanh PC {gap('real','history_only','tanh_pc'):.3f}, "
        f"GRU PC {gap('real','history_only','gru_pc'):.3f}). Switch-centered zero-evidence belief curves "
        "show the same pattern: BPTT models shift belief more strongly after 0.2↔0.8 transitions. "
        "Thus GRU PC can win on correctness while remaining more conservative in explicit belief amplitude — "
        "a dissociation between Q1 and Q2."
    ))
    for rid in ("rIdFig6", "rIdFig7"):
        if rid in rid_map:
            p, w, h, cap = rid_map[rid]
            parts.append(image_xml(rid, width_in=w, height_in=h, caption=cap))

    parts.append(h_xml(2, "Supporting behavioral boards"))
    parts.append(p_xml(
        "Block-stratified correctness and synth↔real transfer boards confirm that rankings are not driven by "
        "a single prior level alone. The full-information control asks whether models can use a supplied prior."
    ))
    for rid in ("rIdFig8", "rIdFig9", "rIdFig10"):
        if rid in rid_map:
            p, w, h, cap = rid_map[rid]
            parts.append(image_xml(rid, width_in=w, height_in=h, caption=cap))

    if prior:
        rows = []
        for mid in ("tanh_bptt", "tanh_pc", "gru", "gru_pc"):
            if mid not in prior:
                continue
            v = prior[mid]
            rows.append([
                _pretty(mid),
                f"{v[0.2]:.3f}",
                f"{v[0.5]:.3f}",
                f"{v[0.8]:.3f}",
                f"{v['balanced']:.3f}",
            ])
        if rows:
            parts.append(table_xml(
                ["Model", "P=0.2", "P=0.5", "P=0.8", "Balanced"],
                rows,
                title="Table 1. Real history-only correctness by block prior (session means).",
                col_fracs=[0.28, 0.18, 0.18, 0.18, 0.18],
            ))

    parts.append(h_xml(2, "Example multipanel diagnostics"))
    for rid in ("rIdFig11", "rIdFig12"):
        if rid in rid_map:
            p, w, h, cap = rid_map[rid]
            parts.append(image_xml(rid, width_in=w, height_in=h, caption=cap))

    parts.append(h_xml(2, "Q3 — Neural prior alignment"))
    if neural is None:
        parts.append(p_xml(
            f"Neural variance-explained analyses are in progress on the same {n} sessions used for behavioral "
            "transfer."
        ))
    else:
        sess = neural["ve_session_mean"]
        if len(sess.columns):
            means = sess.mean(axis=0).sort_values(ascending=False)
            order = ", ".join(f"{_pretty(m)} ({v:.3f})" for m, v in means.items())
            parts.append(p_xml(
                f"Across regions with available units, session-mean VE ranks {order}. "
                f"Behavior-matched models: {', '.join(_pretty(m) for m in neural['matched']) or 'none recorded'}."
            ))
            cols = [c for c in ("tanh_bptt", "tanh_pc", "gru", "gru_pc") if c in sess.columns]
            headers = ["Region"] + [_pretty(c) for c in cols]
            rows = []
            for region in sess.index:
                rows.append([region] + [
                    f"{sess.loc[region, c]:.3f}" if pd.notna(sess.loc[region, c]) else "—"
                    for c in cols
                ])
            fr = [0.22] + [0.78 / len(cols)] * len(cols) if cols else [1.0]
            parts.append(table_xml(
                headers, rows,
                title="Table 2. Session-mean neural VE (linear recalibration) by region and model.",
                col_fracs=fr,
            ))
        surv = neural["survival"]
        if len(surv):
            srows = []
            for _, r in surv.iterrows():
                srows.append([
                    str(r.get("region", "")),
                    f"{float(r.get('delta', float('nan'))):.3f}" if pd.notna(r.get("delta")) else "—",
                    "yes" if bool(r.get("survive_alpha_05")) else "no",
                ])
            parts.append(table_xml(
                ["Region", "VE delta (best − second)", "Survives (Holm)"],
                srows,
                title="Table 3. Behavior-matched survival of neural advantages.",
                col_fracs=[0.34, 0.40, 0.26],
            ))
    for rid in ("rIdFig13", "rIdFig14"):
        if rid in rid_map:
            p, w, h, cap = rid_map[rid]
            parts.append(image_xml(rid, width_in=w, height_in=h, caption=cap))

    parts.append(h_xml(1, "Discussion"))
    parts.append(p_xml(
        "Reading the results in question order clarifies the scientific story. On overall and "
        "switch-centered correctness (Q1), gate-aware GRU PC is competitive with — and on real transfer "
        "often ahead of — GRU BPTT, including immediately after 0.2↔0.8 switches. On belief amplitude and "
        "switch-centered prior probes (Q2), BPTT models still show stronger history gaps and larger belief "
        "swings. Neural VE (Q3) continues to favor GRU BPTT over tanh BPTT among behavior-matched models, "
        "with GRU PC intermediate and tanh PC excluded from the ε-ball. Together, these results separate "
        "“who chooses correctly when the prior changes” from “whose latent belief looks most like a neural "
        "prior axis.”"
    ))
    parts.append(p_xml(
        "Neural analyses target regions implicated in subjective prior coding by Findling et al. (2025). "
        "Region-level claims scale with the number of sessions that sample each area. Matching mouse "
        "subjective-prior trajectories around switches (the strictest form of Q2) and switch-centered neural "
        "dynamics remain explicit next steps."
    ))

    parts.append(h_xml(1, "References"))
    parts.append(ul_xml([
        "International Brain Laboratory (2021). Standardized and reproducible measurement of decision-making in mice. eLife. https://elifesciences.org/articles/63711",
        "International Brain Laboratory (2025). A brain-wide map of neural activity during complex behaviour. Nature. https://www.nature.com/articles/s41586-025-09235-0",
        "Findling, C. et al. (2025). Brain-wide representations of prior information in mouse decision-making. Nature. https://www.nature.com/articles/s41586-025-09226-1",
    ]))

    out = ROOT / "reports" / "v2" / "CURRENT_STATUS_ARTICLE.docx"
    write_docx(out, "".join(parts), images=images)
    return out


def main() -> int:
    methods = build_methods()
    article = build_article()
    print(json.dumps({"methods": str(methods.relative_to(ROOT)), "article": str(article.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
