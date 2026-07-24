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
    surv = pd.read_csv(surv_path) if surv_path.exists() else pd.DataFrame()
    return {"ve_session_mean": sess, "survival": surv, "n_rows": len(u)}


def _mlp_switch_decode_summary() -> dict | None:
    path = ROOT / "reports" / "v2" / "switch_block_decoding" / "mlp_switch_block_decode_metrics.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    out: dict = {"models": {}, "real_belief": {}, "neural": {}}

    def _pack(agg: dict) -> dict:
        packed = {}
        for name, cur in agg.items():
            off = np.asarray(cur["offsets"], dtype=float)
            mean = np.asarray(cur["mean"], dtype=float)
            packed[name] = {
                "window": float(np.nanmean(mean)),
                "pre": float(np.nanmean(mean[(off >= -15) & (off <= -1)])),
                "post": float(np.nanmean(mean[(off >= 0) & (off <= 15)])),
                "at0": float(mean[off == 0][0]) if np.any(off == 0) else float("nan"),
            }
        return packed

    if "models" in raw and "aggregate" in raw["models"]:
        out["models"] = _pack(raw["models"]["aggregate"])
    if "real_belief" in raw and "aggregate" in raw["real_belief"]:
        out["real_belief"] = _pack(raw["real_belief"]["aggregate"])
    if "neural" in raw and "aggregate" in raw["neural"]:
        out["neural"] = _pack(raw["neural"]["aggregate"])
    return out


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
        "with confirmatory claims based on all-model VE rankings plus session-bootstrap survival (§7).",
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
        "updating of the hidden prior is most evident? Includes overall-versus-peri-switch (−30…+30) boards "
        "with Wilcoxon+Holm tests across history-only, full-information, and fixed-prior regimes.",
        "Q2 — Belief updating dynamics: In the history-only condition, does each model’s inferred prior "
        "(zero-evidence belief) update at a similar rate and with similar asymmetries around 0.2↔0.8 "
        "switches, and how does that compare to the mouse subjective prior? Operational probes: history gap, "
        "switch-centered belief curves, and the three-panel MLP block-decode (A2: mouse prior + model q_t).",
        "Q3 — Neural prior alignment: In prior-encoding regions (MOs, vlOFC/ORBvl, ACAd, MOp), which model’s "
        "belief better matches neural prior readouts? Complementarily, can an MLP "
        "decode true block identity around switches from neural û (panel A3)? Survival tests ask whether "
        "best-vs-second VE gaps are stable across sessions (Holm across regions).",
    ]))
    parts.append(p_xml(
        "Reporting order follows the science and the locked working-manuscript figure layout: overall "
        "correctness → switch-centered correctness (0.2→0.8 and 0.8→0.2) → belief / history-gap dynamics → "
        "three-panel MLP decode (Figure 15 with Q2) → supporting boards → neural VE → overall-versus-peri-switch "
        "boards (Figures 16–21)."
    ))
    parts.append(note_xml(
        "v2 amendment: active models are tanh BPTT, corrected tanh PC, GRU BPTT, and gate-aware GRU PC. "
        "Explicit Bayes and meta-RL RNNs are parked (literature-motivated follow-ups). Display order is always "
        "tanh BPTT → tanh PC → GRU → GRU PC with twin-complement colors. Behavior-matching before neural "
        "claims is parked; primary neural analysis uses all models."
    ))

    parts.append(h_xml(1, "2. Task and data"))
    parts.append(h_xml(2, "2.1 IBL biased-block task (intuition)"))
    parts.append(p_xml(
        "Mice perform a two-choice visual decision task. On each trial, a grating appears on either the left "
        "or right side of the screen, and the mouse reports its choice by turning a wheel to bring the grating "
        "to the center. The mouse is rewarded when it makes the correct choice, and it receives no reward or "
        "negative feedback when it makes the wrong choice."
    ))
    parts.append(p_xml(
        "The task is not perfectly balanced. Stimulus location is biased in blocks: for a while the grating is "
        "more likely to appear on one side, then the bias switches. In the IBL task that prior is typically "
        "described by probabilityLeft ∈ {0.2, 0.5, 0.8} (equivalently P(stimulus right) ∈ {0.8, 0.5, 0.2}). "
        "Block changes are not explicitly signaled; the mouse must infer the current bias from recent trial "
        "history. When the stimulus is weak or ambiguous, that inferred prior improves decisions by combining "
        "current sensory evidence with past experience."
    ))
    parts.append(p_xml(
        "Reward feedback is tightly linked to correctness and shapes future choices: outcomes of previous "
        "trials update the animal’s internal estimate of the current prior, especially after a block switch. "
        "That inferred bias is what we call the latent / subjective prior."
    ))

    parts.append(h_xml(2, "2.1b Brain-wide data scale (context)"))
    parts.append(p_xml(
        "The brain-wide release aggregates recordings from 699 Neuropixels probe insertions across 139 mice and "
        "12 laboratories, with 621,733 neurons recorded overall (documentation also describes 459 sessions and "
        "75,708 good-quality units after quality control). For this project we intentionally do not analyze all "
        "regions at once: we lock a focused primary ROI set and a shared behavior+neural session subset "
        "(§2.3–§3)."
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
        "choice distribution; used for ranking (legacy behavior-matching ε-ball is parked; §7.7).",
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
        "Step C — ask how well each model’s belief q_t explains û_t on the same trials "
        "(variance explained; all models shown).",
        "Step D — ask whether the top model’s neural edge over the next-best model is statistically "
        "reliable across sessions (survival test; plain-language explanation in §7.8).",
    ]))
    parts.append(p_xml(
        "This design separates three ideas that are easy to confuse: (i) the mouse’s behavioral prior "
        "estimate, (ii) the neural readout of that prior, and (iii) the model’s latent belief. The primary "
        "comparison is between (ii) and (iii), with (i) used only to define the neural axis."
    ))
    parts.append(note_xml(
        "v2 current-phase amendment: behavior-matching (ε-ball filtering before neural claims) is parked. "
        "Primary neural figures and survival tests use all active models. Legacy matched CSVs remain on disk "
        "for archive only."
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
        "This session-mean is the ranking quantity plotted in the neural VE boards. "
        "Using sessions as the unit of aggregation (rather than pooling all trials) respects that sessions "
        "differ in unit counts, noise, and coverage, and matches the resampling unit used in survival tests."
    ))

    parts.append(h_xml(2, "7.7 Behavior matching (parked in this phase)"))
    parts.append(p_xml(
        "An earlier pipeline gated confirmatory neural claims on a choice-primary ε-ball: keep only models "
        "whose real history-only cross-entropy is within ε = 0.05 of the best model. That filter is no longer "
        "part of the primary current-phase analysis. All four active models are shown in VE boards, and "
        "survival tests compare best vs second among all models. The ε-ball code and matched CSV/JSON "
        "artifacts remain available as a legacy archive only."
    ))

    parts.append(h_xml(2, "7.8 Survival testing — what it is, in plain language"))
    parts.append(p_xml(
        "The VE bars answer: “On average, which model’s belief lines up best with this brain region’s prior "
        "axis?” The survival test answers a tougher follow-up: “Is the winner’s lead over the runner-up big "
        "enough that we should trust it, given that we only have a handful of sessions and we look at several "
        "regions?”"
    ))
    parts.append(p_xml("How it is done (current phase):"))
    parts.append(ul_xml([
        "Within each brain region, rank all active models by session-mean VE and pick the best and the "
        "second-best.",
        "Compute the gap Δ = (average VE of the best) − (average VE of the second-best), averaging over "
        "sessions that have both.",
        "Ask how stable that gap is by resampling sessions with replacement many times (bootstrap, B = 2000). "
        "Each resample recomputes Δ. If many resamples flip the sign of Δ, the lead is fragile.",
        "From the bootstrap we get a confidence interval for Δ and a p-value for whether Δ is consistent "
        "with zero.",
        "Because we test several regions, we adjust the p-values with the Holm–Bonferroni procedure so that "
        "looking at four ROIs does not inflate false positives.",
        "We say the advantage “survives” in a region when the Holm-adjusted p-value is below 0.05. Green bars "
        "in the survival figure mark regions that survive; gray bars mark regions that do not.",
    ]))
    parts.append(p_xml(
        "Non-technical takeaway: surviving means “the top model’s edge over the next-best model in explaining "
        "this region’s prior signal looks real after we account for session-to-session luck and for testing "
        "multiple regions.” Not surviving does not mean the region has no prior signal — it means the "
        "best-vs-second gap is not yet trustworthy under these corrections (often because few sessions cover "
        "that ROI)."
    ))

    parts.append(h_xml(2, "7.9 What this analysis does and does not claim"))
    parts.append(ul_xml([
        "Does claim: relative alignment of model belief trajectories with a behavior-defined neural prior "
        "axis under a fixed peri-stimulus window and linear readout, plus whether the top-vs-second gap is "
        "stable across sessions after multi-region correction.",
        "Does not claim: that û is the unique or true neural prior; that ridge is the brain’s readout; "
        "that VE proves causal encoding; or that a surviving gap proves the model is implemented in that region.",
        "Known gaps: no embodied-prior controls (video / eye) yet; peri-stimulus window only; ROI "
        "coverage uneven across sessions; mouse prior is itself a model of behavior; behavior-matching is "
        "parked (not used to filter models in this phase).",
    ]))

    parts.append(h_xml(1, "8. MLP switch-centered block decoding (Q2 three-way probe)"))
    parts.append(p_xml(
        "A complementary probe asks whether true biased-block identity (P(right) = 0.2 vs 0.8) can be "
        "decoded around genuine block switches from (A1) synth model latents, (A2) real mouse prior and "
        "model zero-evidence belief, and (A3) neural prior readouts. Implemented in "
        "scripts/16_plot_mlp_switch_block_decoding.py and src/models_v2/block_decode.py "
        "(spec: docs/spec_switch_block_decoding.md). Panel A2 is the primary Q2 three-way scalar comparison."
    ))

    parts.append(h_xml(2, "8.1 What is decoded"))
    parts.append(ul_xml([
        "Target label: biased-block identity on each trial in an isolated genuine 0.2↔0.8 switch window "
        "(−30 … +30 trials relative to the switch; transitions involving 0.5 are excluded).",
        "A1 (synth, history-only): concatenated within-trial hidden states under a zero-current-evidence "
        "probe (visual / action / reward channels zeroed; go cue retained). Capacity / latent readability.",
        "A2 (real shared cohort): scalar mouse_prior_hat and each model’s zero-evidence belief q_t from "
        "history-only real rollouts (same n=8 eids / session order as the neural cohort). Primary Q2 panel.",
        "A3 (real shared cohort): scalar out-of-fold CV-Ridge neural prior readout û_t in each primary ROI "
        "(same construction as §7). One decode curve per region.",
    ]))

    parts.append(h_xml(2, "8.2 Decoder and evaluation protocol"))
    parts.append(ul_xml([
        "Decoder: one-hidden-layer MLP (hidden size 64) trained with early stopping on a validation split "
        "(settings shared across panels; see DecoderSettings).",
        "A1: held-out synthetic sessions (48 × 929 trials) with a deterministic train / validation / "
        "test session split; matched inputs across models; accuracy on the test split vs trials-from-switch.",
        "A2–A3: leave-one-session-out evaluation on the shared behavior+neural cohort (n=8).",
        "Uncertainty bands: A1 — sample SD across available task-model seeds (canonical seed 7 if no "
        "replicates); A2–A3 — sample SD across held-out sessions.",
        "Primary figure: reports/v2/figures/switch_block_decoding/mlp_rnn_vs_pc_switch_decoding.png "
        "(three panels). Metrics JSON: reports/v2/switch_block_decoding/mlp_switch_block_decode_metrics.json.",
    ]))

    parts.append(h_xml(2, "8.3 How to read the curves"))
    parts.append(p_xml(
        "Decode accuracy near 1.0 far from the switch means block identity is strongly represented in the "
        "features. A dip at trial 0 is expected: the label changes while beliefs / latents / neural readouts "
        "update with a delay. A1 asks which latent supports readable block identity without current evidence; "
        "A2 asks whether mouse prior and model q_t track block identity similarly around switches; A3 asks "
        "where neural prior readouts carry readable block information. Related to, but distinct from, neural "
        "VE (§7): VE asks whether model belief explains û; block decoding asks whether û, mouse prior, or "
        "model belief explain true block identity around switches."
    ))

    parts.append(h_xml(1, "9. Limitations and open risks"))
    parts.append(h_xml(2, "9.1 Cohort and coverage"))
    parts.append(ul_xml([
        "ROI coverage is by cohort union; individual sessions typically contribute 1–few regions. Survival "
        "tests are underpowered where n_sessions is small (e.g. ACAd in the current lock).",
        "Almost-perfect QC plus ephys requirements yield a small n; results should be read as a locked "
        "pilot cohort, not a brain-wide census.",
    ]))
    parts.append(h_xml(2, "9.2 Modeling and transfer"))
    parts.append(ul_xml([
        "Synthetic training approximates empirical statistics but does not replay individual mice "
        "(intentional; still a distribution shift risk on transfer).",
        "Teacher-forced training feedback versus mouse feedback on transfer is a known shift (V2-R4).",
        "tanh PC / GRU PC use the same session length as BPTT; compute cost is higher due to 32 inference rounds.",
        "Explicit Bayes is parked (legacy module retained) and is not part of the active ranking.",
    ]))
    parts.append(h_xml(2, "9.3 Neural analysis"))
    parts.append(ul_xml([
        "Embodied-prior controls (video / eye position) are not yet applied.",
        "The neural window is peri-stimulus only; complementary inter-trial decoding is left for later work.",
        "Mouse prior and neural axis are estimated quantities; errors in either reduce neural VE for all "
        "models and can shrink detectable advantages.",
        "Linear ridge + affine recalibration cannot capture nonlinear neural–belief relationships.",
        "MLP block decoding on scalar mouse prior / model q_t / neural û is nearly a calibrated threshold; "
        "few sessions per ROI widen session SD and thin far-window switches.",
    ]))
    parts.append(h_xml(2, "9.4 Out of v2 scope"))
    parts.append(ul_xml([
        "Mouse fine-tuning, reaction-time primary losses, meta-RL, and Bayesian+credit-assignment twins "
        "are parked.",
        "Expanded Findling ROIs remain optional until embodied controls and larger coverage exist.",
        "Trajectory-distance matching of model belief to mouse prior (beyond shared MLP block-decode "
        "curves in panel A2) remains an explicit follow-up.",
    ]))

    out = ROOT / "reports" / "v2" / "METHODS_DETAILED.docx"
    write_docx(out, "".join(parts))
    return out


def _overall_vs_switch_summary() -> dict | None:
    path = ROOT / "reports" / "v2" / "metrics" / "overall_vs_switch_correctness.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_article() -> Path:
    """Presentation-ready current-status manuscript organized around Q1–Q3."""

    metrics = _load_metrics()
    by = {(r["domain"], r["regime"], r["model"]): r for r in metrics}
    prior = _per_prior_real_history()
    neural = _neural_summary()
    mlp_sw = _mlp_switch_decode_summary()
    ovs = _overall_vs_switch_summary()
    man = json.loads((ROOT / "data" / "manifests" / "shared_behavior_neural_eids.json").read_text())
    n = int(man.get("n_sessions", 8))

    def acc(domain, regime, mid):
        r = by.get((domain, regime, mid))
        return float("nan") if r is None else r["acc"]

    def gap(domain, regime, mid):
        r = by.get((domain, regime, mid))
        return float("nan") if r is None else r["gap"]

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
        ("rIdFig15", FIG / "switch_block_decoding" / "mlp_rnn_vs_pc_switch_decoding.png", 6.8, 3.6,
         "Figure 15. Three-panel MLP block decode: synth latents · mouse prior + model belief · neural û."),
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
         "Figure 13. Neural prior variance explained by model belief (all models; session-mean VE)."),
        ("rIdFig14", FIG / "neural" / "survival_tests.png", 6.0, 3.2,
         "Figure 14. Survival test of best-vs-second VE gaps across regions."),
        ("rIdFig16", FIG / "comparison" / "real_history_only_overall_vs_switch_correctness.png", 6.4, 3.8,
         "Figure 16. Real history-only: overall vs peri-switch correctness (−30…+30)."),
        ("rIdFig17", FIG / "comparison" / "synth_history_only_overall_vs_switch_correctness.png", 6.4, 3.8,
         "Figure 17. Synth history-only: overall vs peri-switch correctness (−30…+30)."),
        ("rIdFig18", FIG / "comparison" / "real_full_information_overall_vs_switch_correctness.png", 6.4, 3.8,
         "Figure 18. Real full-information (oracle): overall vs peri-switch correctness."),
        ("rIdFig19", FIG / "comparison" / "synth_full_information_overall_vs_switch_correctness.png", 6.4, 3.8,
         "Figure 19. Synth full-information (oracle): overall vs peri-switch correctness."),
        ("rIdFig20", FIG / "comparison" / "real_fixed_prior_overall_vs_switch_correctness.png", 6.4, 3.6,
         "Figure 20. Real fixed-prior: overall correctness only."),
        ("rIdFig21", FIG / "comparison" / "synth_fixed_prior_overall_vs_switch_correctness.png", 6.4, 3.6,
         "Figure 21. Synth fixed-prior: overall correctness only."),
    ]
    images = [(rid, p) for rid, p, *_ in fig_specs if p.exists()]
    rid_map = {rid: (p, w, h, cap) for rid, p, w, h, cap in fig_specs if p.exists()}

    def fig(rid: str) -> None:
        if rid in rid_map:
            p, w, h, cap = rid_map[rid]
            parts.append(image_xml(rid, width_in=w, height_in=h, caption=cap))

    parts: list[str] = []
    parts.append(p_xml(
        "Latent prior updating in the IBL decision task: "
        "BPTT versus predictive-coding recurrent models",
        bold=True, center=True, size=36,
    ))
    parts.append(p_xml(
        "Current-status manuscript — organized for presentation around three scientific questions",
        center=True, italic=True, size=22,
    ))
    parts.append(p_xml(""))

    # ------------------------------------------------------------------ #
    # Opening frame (user-supplied text, updated to current implementation)
    # ------------------------------------------------------------------ #
    parts.append(h_xml(1, "1. Task summary"))
    parts.append(p_xml(
        "Mice perform a two-choice visual decision task. On each trial, a grating appears on either the left "
        "or right side of the screen, and the mouse reports its choice by turning a wheel to bring the grating "
        "to the center. The mouse is rewarded when it makes the correct choice, and it receives no reward or "
        "negative feedback when it makes the wrong choice."
    ))
    parts.append(p_xml(
        "The task is not perfectly balanced. Instead, the stimulus location is biased in blocks: for a while, "
        "the grating is more likely to appear on one side, and then later the bias switches to the other side. "
        "In the IBL task, that prior probability is typically something like 0.8 for one side and 0.2 for the "
        "other side (also 0.5), and the block changes are not explicitly signaled to the mouse. This means the "
        "mouse has to infer the current bias from recent trial history rather than being told directly."
    ))
    parts.append(p_xml(
        "That prior matters because when the visual stimulus is weak or ambiguous, the mouse can improve its "
        "decision by using what it has learned about which side is currently more likely. So the animal is "
        "combining current sensory evidence with past experience. In other words, it is not just reacting to "
        "the stimulus on the screen; it is also using an internal expectation about the hidden state of the task."
    ))
    parts.append(p_xml(
        "The reward system is tightly linked to correctness. A correct choice usually leads to reward delivery, "
        "while an incorrect choice leads to no reward or negative feedback. This reward feedback is important "
        "because it helps shape future choices: the mouse can use the outcome of previous trials to update its "
        "internal estimate of the current prior and adjust its behavior after a block switch."
    ))

    parts.append(h_xml(1, "2. Data size and scope"))
    parts.append(p_xml(
        "The brain-wide release aggregates recordings from 699 Neuropixels probe insertions across 139 mice and "
        "12 laboratories, with 621,733 neurons recorded overall. The documentation also describes 459 sessions "
        "and 75,708 good-quality units after quality control, and the dataset spans a very large number of brain "
        "regions. For a realistic first study, we do not use all regions at once. Instead we define a focused "
        f"region set and a session subset with complete behavioral and neural alignment: {n} almost-perfect QC "
        "sessions (shared behavior+neural cohort) covering the primary ROIs MOs, ORBvl (vlOFC), ACAd, and MOp."
    ))

    parts.append(h_xml(1, "3. Literature review"))
    parts.append(p_xml(
        "Animals do not solve decision tasks with a fixed policy; they continually update internal expectations "
        "from feedback in response to changing contingencies, and the International Brain Laboratory (IBL) task "
        "makes this explicit through a hidden, block-switching prior that mice infer online to improve "
        "performance (International Brain Laboratory, 2021; Findling et al., 2025). In the recent brain-wide "
        "IBL study, that subjective prior was found to be distributed across roughly 20–30% of recorded "
        "regions, spanning early sensory areas such as LGN and V1 through motor cortex and higher-order "
        "regions including dACC and vlOFC, which suggests that the computation is not confined to a single "
        "“decision center” (Findling et al., 2025). The authors interpret this widespread representation as "
        "consistent with Bayesian inference implemented through loops between areas and explicitly note that "
        "building a neural model of this process remains “a pressing, remaining task,” making the dataset "
        "unusually well suited to a mechanistic comparison of candidate computational accounts "
        "(Findling et al., 2025)."
    ))
    parts.append(p_xml(
        "Two broad classes of models could explain this kind of distributed, trial-by-trial updating. Standard "
        "task-optimized recurrent neural networks treat the relevant prior as an emergent property of recurrent "
        "dynamics: they are trained offline to perform the task, and once trained, their internal state can "
        "evolve online while the weights remain frozen (Mante et al., 2013; Barak, 2017). That makes them "
        "attractive models of flexible computation, but it also means that the prior is not represented as an "
        "explicit variable; instead, it is embedded in the network’s hidden state and learned dynamics "
        "(Mante et al., 2013; Barak, 2017). A closely related extension discussed in the literature is the "
        "meta-reinforcement-learning RNN, which learns offline so that at test time it can adapt online from "
        "reward feedback with fixed weights (Yang et al., 2019; Botvinick et al., 2019). The relevant contrast "
        "is not simply “adaptive versus non-adaptive,” but where adaptation is implemented: in the dynamics of "
        "frozen weights, or in an explicit inference process."
    ))
    parts.append(p_xml(
        "The alternative is a predictive-coding or online-learning network, which represents the prior as an "
        "explicit latent variable inferred by minimizing prediction error through local updates "
        "(Whittington & Bogacz, 2017; Jiang & Rao, 2022). In this view, the network does not merely store the "
        "prior implicitly in recurrent activity; it continually updates a latent belief state, making the prior "
        "directly addressable and more interpretable as an inference variable (Jiang & Rao, 2022; Millidge et "
        "al., 2022). Predictive-coding models are especially compelling in online, few-shot, and "
        "continual-learning regimes (Millidge et al., 2022; Song et al., 2024). This matters here because the "
        "IBL task is inherently non-stationary: block transitions create change points that force revision of "
        "belief about the hidden prior (International Brain Laboratory, 2021; Findling et al., 2025)."
    ))
    parts.append(p_xml(
        "What makes the IBL dataset particularly valuable is that behavior alone is unlikely to distinguish "
        "these accounts. Model families can produce similar choice predictions on aggregate metrics while "
        "differing sharply in internal computation (Wang et al., 2018; Song et al., 2024). The literature "
        "emphasizes that the relevant behavioral baseline is already close to optimal, leaving little room for "
        "large gains in choice accuracy alone (Findling et al., 2025). Discriminative power therefore lies in "
        "neural alignment: whether model latent states match population activity in regions that carry the "
        "prior, especially around block switches (Findling et al., 2025)."
    ))
    parts.append(p_xml(
        "The broader modeling literature supports this strategy. Task-trained RNNs are standard mechanistic "
        "models of neural computation (Mante et al., 2013; Barak, 2017). Predictive-coding networks provide "
        "biologically motivated local learning rules for continual updating (Whittington & Bogacz, 2017; "
        "Jiang & Rao, 2022). Alignment metrics such as RSA, CCA, CKA, Procrustes, DSA, and CEBRA compare "
        "latent dynamics beyond output accuracy (Kriegeskorte et al., 2008; Kornblith et al., 2019; "
        "Schneider et al., 2023). In this project’s current phase we operationalize neural comparison with a "
        "behavior-defined 1-D prior axis (ridge readout) plus variance explained and survival testing; richer "
        "geometry metrics remain available for follow-up."
    ))
    parts.append(p_xml(
        "The task is therefore not just a behavioral benchmark but a mechanistic probe. The mouse must "
        "integrate sensory evidence, feedback, and recent history to infer the block prior, and that prior "
        "influences choices over multiple trials after each switch (International Brain Laboratory, 2021; "
        "Findling et al., 2025). Because the prior is represented in multiple areas, the data support asking "
        "whether the computation is best understood as emergent recurrent dynamics or as explicit latent-state "
        "inference via local error-driven updates (Findling et al., 2025)."
    ))
    parts.append(p_xml(
        "The IBL dataset is relevant for three reasons. First, it combines a standardized decision task with "
        "an explicit hidden prior (International Brain Laboratory, 2021). Second, brain-wide recordings show "
        "that the prior is not localized to one area (Findling et al., 2025). Third, the dataset is large "
        "enough to support controlled comparisons among learning regimes — in the literature, often framed as "
        "standard RNN vs meta-RL RNN vs predictive-coding online learning (Wang et al., 2018; Findling et al., "
        "2025). Our implemented comparison instantiates that agenda as architecturally matched BPTT vs "
        "corrected predictive-coding twins on tanh and GRU backbones (four models), evaluated on a locked "
        "shared behavior+neural cohort and focused primary ROIs."
    ))
    parts.append(p_xml(
        "A concise framing: the animal’s inferred prior is the computational variable of interest; the IBL "
        "recordings tell us where and how broadly it is represented; the model comparison asks which learning "
        "regime best explains that representation (Findling et al., 2025). The main novelty is not simply "
        "predicting choice, but explaining the structure of the prior in neural data."
    ))

    parts.append(h_xml(1, "4. Motivation"))
    parts.append(p_xml(
        "Intelligent behavior requires more than learning fixed stimulus-response mappings. Natural "
        "environments are non-stationary, so animals must continuously update their beliefs while preserving "
        "useful prior knowledge, and understanding how they do this remains a central challenge in "
        "neuroscience and AI."
    ))
    parts.append(p_xml(
        "The International Brain Laboratory task is a decision task with a hidden prior that changes across "
        "blocks, so the mouse must infer the current bias from feedback and recent history "
        "(International Brain Laboratory et al., 2021; Findling et al., 2025). That makes it a strong testbed "
        "for studying online belief updating and its brain-wide representation. In the recent brain-wide IBL "
        "study, that prior was decoded across sensory, motor, and higher-order areas, suggesting that the "
        "relevant computation is distributed and dynamic, not localized or static. This is precisely where "
        "our project adds a mechanistic step: the IBL paper shows where the prior is represented, but not how "
        "it is computed, and we test which computational account best explains the same phenomenon under a "
        "more mechanistic lens."
    ))
    parts.append(p_xml(
        "A central open question is whether the mouse’s trial-by-trial updating is best explained by recurrent "
        "dynamics that implicitly encode the prior, or by a model that explicitly infers and updates a latent "
        "belief state online (Mante et al., 2013; Barak, 2017; Wang et al., 2016; Wang et al., 2018; "
        "Botvinick et al., 2019). Standard task-trained RNNs (and meta-RL RNNs in the broader literature) can "
        "learn flexible, history-dependent behavior with frozen weights at test time, but their internal "
        "“prior” is embedded in dynamics. Predictive-coding / online-learning models implement belief updating "
        "more explicitly through local error-driven updates (Whittington & Bogacz, 2017; Jiang & Rao, 2022; "
        "Millidge et al., 2022)."
    ))
    parts.append(p_xml(
        "The motivation is therefore twofold. First, we test which model best reproduces choice behavior and "
        "switch-centered correctness around hidden-prior block switches (Findling et al., 2025; Wang et al., "
        "2018; Song et al., 2024). Second, and more importantly, we identify which model best matches neural "
        "prior representation in prior-linked regions, especially around switch points. The project is not "
        "just a behavioral benchmark; it is a mechanistic test of whether the subjective prior is best "
        "understood as an emergent property of recurrent dynamics or as explicit online inference."
    ))

    parts.append(h_xml(1, "5. Proposed questions (presentation spine)"))
    parts.append(ul_xml([
        "Q1 — Which of the models best reproduces trial-by-trial correctness (accuracy vs the correct "
        "stimulus side) both across the full task and in trials surrounding block switches, where online "
        "updating of the hidden prior is most evident?",
        "Q2 — In the history-only condition, does each model’s inferred prior update at the same rate and "
        "with the same asymmetries as the mouse’s subjective prior (estimated from behavior) around block "
        "switches? Which account better captures how fast and how the belief is revised, not just "
        "steady-state accuracy?",
        "Q3 — In prior-encoding regions (MOs, vlOFC/ORBvl, ACAd, MOp), which model better matches the neural "
        "population dynamics of the prior — operationally, which model’s belief best explains a "
        "behavior-defined neural prior axis, and does the top model’s edge survive across sessions?",
    ]))
    parts.append(p_xml(
        "The remainder of this document answers these three questions in turn. Each axis documents detailed "
        "methodology (aligned with METHODS_DETAILED.docx), reports results with figures, and closes with an "
        "explicit answer suitable for a presentation narrative."
    ))

    parts.append(h_xml(1, "6. Shared methods (applies to all questions)"))
    parts.append(h_xml(2, "6.1 Evaluation hierarchy"))
    parts.append(ul_xml([
        "Train only on synthetic IBL-like sessions (never mouse choice as a supervised target).",
        "Rank and diagnose on held-out synthetic sessions from the same generator (matched statistics).",
        "Freeze weights and transfer to the shared real cohort (same tick/channel schema).",
        "On the same sessions, compare model belief to neural prior readouts in primary ROIs.",
    ]))
    parts.append(h_xml(2, "6.2 Synthetic generator"))
    parts.append(p_xml(
        "Synthetic sessions are sampled from empirical distributions (block lengths, prior transitions, "
        "contrast frequencies, session length), not replays of individual mice. Block priors ∈ {0.2, 0.5, 0.8}; "
        "contrasts ∈ {0, 0.0625, 0.125, 0.25, 1.0}; default session length ≈ 929 trials. Config: "
        "configs/synthetic_v2.yaml."
    ))
    parts.append(h_xml(2, "6.3 Shared real cohort and ROIs"))
    parts.append(p_xml(
        f"The locked cohort has {n} sessions that pass almost-perfect behavior QC and have usable spikes in "
        "at least one primary ROI. Sessions are chosen so the union covers MOs, ORBvl (vlOFC), ACAd, and MOp. "
        "Per-region neural analyses use only sessions containing that region. Scoring is always model choice "
        "versus correct stimulus side (session means ± 95% CI)."
    ))
    parts.append(h_xml(2, "6.4 Models and training (implementation)"))
    parts.append(ul_xml([
        "tanh BPTT / GRU BPTT: standard recurrent nets trained with backpropagation through time.",
        "tanh PC / GRU PC: identical architectures at test time; trained with corrected predictive-coding "
        "credit assignment (32 inference rounds, output_precision=0.025, nudge-normalized local updates; "
        "GRU PC is gate-aware).",
        "Shared schedule: 60 epochs × 24 sessions × 929 trials (same exposure across twins).",
        "Display / plot order always: tanh BPTT → tanh PC → GRU → GRU PC (twin-complement colors).",
        "Regimes: history_only (primary), full_information (oracle prior at readout), fixed_prior (control).",
    ]))
    parts.append(h_xml(2, "6.5 Belief and history gap"))
    parts.append(p_xml(
        "Model belief q_t is the counterfactual zero-evidence P(choice = right): visual contrast channels "
        "are zeroed while history (action/reward) channels and the go cue remain. History gap = mean q_t in "
        "true 0.8 blocks − mean q_t in true 0.2 blocks. Large positive gaps mean strong history-dependent "
        "prior use."
    ))
    parts.append(h_xml(2, "6.6 Mouse subjective prior (used in Q2–Q3)"))
    parts.append(p_xml(
        "The mouse prior is not the experimenter block probability. It is a causal history-only estimate: "
        "within each session a leaky update p_{t+1} = (1−α)p_t + α·1{stimulus right}, with p_t reported "
        "before the update. α is selected by grid search to minimize logistic choice NLL with signed "
        "contrast. The resulting p̂_t is the regression target for neural decoding and the mouse series in "
        "the Q2 MLP panel."
    ))

    parts.append(h_xml(1, "Axis Q1 — Correctness overall and around switches"))
    parts.append(h_xml(2, "Q1. Method (detailed)"))
    parts.append(p_xml(
        "Q1 asks which model best matches the task’s correct side, both in steady state and when the hidden "
        "prior changes."
    ))
    parts.append(ul_xml([
        "Overall correctness: fraction of trials with model choice = correct stimulus side, on held-out "
        "synthetic history-only sessions and on real history-only transfer (shared cohort).",
        "Switch-centered correctness: identify isolated genuine 0.2↔0.8 transitions; align at the switch "
        "trial; average correctness in −30…+30 separately for 0.2→0.8 and 0.8→0.2 (session means ± SEM).",
        "Post-switch summary: mean correctness on trials 0–15 after the switch (session 95% CIs).",
        "Overall-versus-peri-switch boards: for each session, overall correctness vs mean correctness in "
        "−30…+30 for each direction; paired Wilcoxon signed-rank tests; Holm correction within each "
        "domain×regime panel; fixed_prior shows overall only (no switches).",
        "Supporting: correctness by block prior (0.2/0.5/0.8 + balanced); synth↔real boards; "
        "full-information oracle control; example multipanel diagnostics.",
    ]))

    parts.append(h_xml(2, "Q1. Results"))
    parts.append(p_xml(
        "On synthetic history-only held-out data, BPTT models lead "
        f"(tanh BPTT {acc('synth','history_only','tanh_bptt'):.3f}, GRU {acc('synth','history_only','gru'):.3f}) "
        f"over PC twins (tanh PC {acc('synth','history_only','tanh_pc'):.3f}, "
        f"GRU PC {acc('synth','history_only','gru_pc'):.3f}; Figure 1)."
    ))
    fig("rIdFig1")
    parts.append(p_xml(
        "On real history-only transfer the ranking flips: "
        f"GRU PC leads ({acc('real','history_only','gru_pc'):.3f}), then GRU "
        f"({acc('real','history_only','gru'):.3f}), tanh BPTT "
        f"({acc('real','history_only','tanh_bptt'):.3f}), tanh PC "
        f"({acc('real','history_only','tanh_pc'):.3f}; Figure 2). "
        "So the PC twin that looked weaker on synthetic held-out is the strongest real-transfer chooser."
    ))
    fig("rIdFig2")
    parts.append(p_xml(
        "Around switches, the same real-transfer pattern holds. GRU PC leads post-switch correctness "
        "(trials 0–15) in both directions (~0.820 for 0.2→0.8; ~0.829 for 0.8→0.2), with GRU next "
        "(Figures 3–5). The advantage is therefore not only a steady-state effect."
    ))
    for rid in ("rIdFig3", "rIdFig4", "rIdFig5"):
        fig(rid)

    parts.append(p_xml(
        "Overall-versus-peri-switch boards (Figures 16–21) ask whether peri-switch accuracy differs from "
        "full-session accuracy. On real history-only, peri-switch means sit slightly above overall, but "
        "with n=8 sessions those paired differences generally do not survive Holm correction — the "
        "practical message is that overall and peri-switch rankings tell a consistent story, not that "
        "switches create a large statistical dissociation at this cohort size."
    ))
    for rid in ("rIdFig16", "rIdFig17", "rIdFig18", "rIdFig19", "rIdFig20", "rIdFig21"):
        fig(rid)

    parts.append(p_xml(
        "Supporting boards confirm the ranking is not an artifact of a single prior level (Figure 8), "
        "show synth↔real transfer (Figure 9), and show that models can use an oracle prior when supplied "
        "(Figure 10). Example multipanels for the two leading real-transfer models appear as Figures 11–12."
    ))
    for rid in ("rIdFig8", "rIdFig9", "rIdFig10", "rIdFig11", "rIdFig12"):
        fig(rid)

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

    parts.append(h_xml(2, "Q1. Answer"))
    parts.append(p_xml(
        "Answer to Q1. For trial-by-trial correctness on real history-only sessions — both overall and "
        "immediately after 0.2↔0.8 switches — gate-aware GRU PC is the best of the four models, with GRU "
        "BPTT a close second. BPTT wins on synthetic held-out, but that ranking does not transfer. "
        "Presentation takeaway: credit-assignment regime matters for real-animal correctness, and the "
        "advantage concentrates where updating is required."
    ))

    # ------------------------------------------------------------------ #
    # Q2
    # ------------------------------------------------------------------ #
    parts.append(h_xml(1, "Axis Q2 — Belief updating vs the mouse prior"))
    parts.append(h_xml(2, "Q2. Method (detailed)"))
    parts.append(p_xml(
        "Q2 asks whether models revise an internal prior like the mouse — not only whether they choose "
        "correctly."
    ))
    parts.append(ul_xml([
        "History gap on real history-only rollouts (steady-state prior use from zero-evidence belief).",
        "Switch-centered zero-evidence belief curves for 0.2→0.8 and 0.8→0.2 (−30…+30; mean ± SEM across "
        "sessions).",
        "Three-panel MLP block decode on isolated 0.2↔0.8 switch windows (−30…+30): target label = true "
        "biased-block identity (0.2 vs 0.8).",
        "A1 (capacity): features = concatenated within-trial zero-evidence hidden states on matched "
        "synthetic history-only sessions; train/val/test session split; uncertainty = SD across task-model "
        "seeds when available.",
        "A2 (primary mouse–model probe): features = scalar mouse prior p̂_t and each model’s q_t from real "
        "history-only rollouts on the shared cohort; leave-one-session-out MLP; uncertainty = session SD.",
        "A3 (bridge to Q3): features = scalar CV-ridge OOF neural prior readout û_t by ROI; same LOSO MLP.",
    ]))

    parts.append(h_xml(2, "Q2. Results"))
    parts.append(p_xml(
        "History gaps remain larger for BPTT models "
        f"(GRU {gap('real','history_only','gru'):.3f}, tanh BPTT {gap('real','history_only','tanh_bptt'):.3f}) "
        f"than for PC twins (tanh PC {gap('real','history_only','tanh_pc'):.3f}, "
        f"GRU PC {gap('real','history_only','gru_pc'):.3f}; Figure 6). Switch-centered belief curves show the "
        "same pattern: BPTT models swing belief more strongly after 0.2↔0.8 transitions (Figure 7)."
    ))
    fig("rIdFig6")
    fig("rIdFig7")

    if mlp_sw and mlp_sw.get("models"):
        mm = mlp_sw["models"]
        parts.append(p_xml(
            "Panel A1 (synth latents): BPTT latents are most readable for block identity "
            f"(GRU {mm.get('gru', {}).get('window', float('nan')):.3f}, "
            f"tanh BPTT {mm.get('tanh_bptt', {}).get('window', float('nan')):.3f}; "
            f"PC twins ~{mm.get('gru_pc', {}).get('window', float('nan')):.2f}). "
            "This is a capacity result, not a claim that BPTT equals the mouse."
        ))
        rows = []
        for mid in ("tanh_bptt", "tanh_pc", "gru", "gru_pc"):
            if mid not in mm:
                continue
            v = mm[mid]
            rows.append([
                _pretty(mid),
                f"{v['window']:.3f}",
                f"{v['pre']:.3f}",
                f"{v['post']:.3f}",
                f"{v['at0']:.3f}",
            ])
        if rows:
            parts.append(table_xml(
                ["Model", "Window", "Pre (−15…−1)", "Post (0…15)", "At 0"],
                rows,
                title="Table 2. A1 — MLP block-decode from zero-evidence latents (synth).",
                col_fracs=[0.22, 0.20, 0.20, 0.20, 0.18],
            ))

    if mlp_sw and mlp_sw.get("real_belief"):
        rb = mlp_sw["real_belief"]
        parts.append(p_xml(
            "Panel A2 (real shared cohort — the key mouse–model comparison): mouse prior window decode "
            f"≈ {rb.get('mouse', {}).get('window', float('nan')):.3f}. "
            f"GRU belief is closest/slightly above ({rb.get('gru', {}).get('window', float('nan')):.3f}), "
            f"tanh BPTT nearly matches the mouse ({rb.get('tanh_bptt', {}).get('window', float('nan')):.3f}), "
            f"while PC beliefs are lower (~{rb.get('tanh_pc', {}).get('window', float('nan')):.2f} / "
            f"{rb.get('gru_pc', {}).get('window', float('nan')):.2f}). "
            "Thus the model that wins Q1 correctness (GRU PC) is not the model whose scalar belief most "
            "closely tracks block identity like the mouse."
        ))
        rows = []
        for name, lab in (
            ("mouse", "mouse prior"),
            ("tanh_bptt", "tanh BPTT"),
            ("tanh_pc", "tanh PC"),
            ("gru", "GRU"),
            ("gru_pc", "GRU PC"),
        ):
            if name not in rb:
                continue
            v = rb[name]
            rows.append([
                lab,
                f"{v['window']:.3f}",
                f"{v['pre']:.3f}",
                f"{v['post']:.3f}",
                f"{v['at0']:.3f}",
            ])
        if rows:
            parts.append(table_xml(
                ["Series", "Window", "Pre (−15…−1)", "Post (0…15)", "At 0"],
                rows,
                title="Table 3. A2 — MLP block-decode from mouse prior and model belief q_t.",
                col_fracs=[0.22, 0.20, 0.20, 0.20, 0.18],
            ))

    if mlp_sw and mlp_sw.get("neural"):
        nn = mlp_sw["neural"]
        parts.append(p_xml(
            "Panel A3 (neural û): block identity is readable above chance mainly in MOs "
            f"({nn.get('MOs', {}).get('window', float('nan')):.3f}) and vlOFC "
            f"({nn.get('vlOFC_orbvl', {}).get('window', float('nan')):.3f}); MOp intermediate "
            f"({nn.get('MOp', {}).get('window', float('nan')):.3f}); ACAd near chance "
            f"({nn.get('ACAd', {}).get('window', float('nan')):.3f})."
        ))
        rows = []
        for reg, lab in (
            ("MOs", "MOs"),
            ("vlOFC_orbvl", "vlOFC"),
            ("ACAd", "ACAd"),
            ("MOp", "MOp"),
        ):
            if reg not in nn:
                continue
            v = nn[reg]
            rows.append([
                lab,
                f"{v['window']:.3f}",
                f"{v['pre']:.3f}",
                f"{v['post']:.3f}",
                f"{v['at0']:.3f}",
            ])
        if rows:
            parts.append(table_xml(
                ["Region", "Window", "Pre (−15…−1)", "Post (0…15)", "At 0"],
                rows,
                title="Table 4. A3 — MLP block-decode from neural prior readouts.",
                col_fracs=[0.18, 0.20, 0.22, 0.20, 0.20],
            ))
    fig("rIdFig15")

    parts.append(h_xml(2, "Q2. Answer"))
    parts.append(p_xml(
        "Answer to Q2. BPTT models update explicit belief more strongly (larger history gaps and larger "
        "switch-centered swings). On the shared real cohort, GRU (and tanh BPTT) scalar beliefs track "
        "block identity most like the mouse prior under the MLP probe, while GRU PC — the Q1 correctness "
        "winner — remains more conservative in belief amplitude. Presentation takeaway: choice accuracy and "
        "belief dynamics dissociate; answering “who updates like the mouse” requires the Q2 probes, not Q1 alone."
    ))

    # ------------------------------------------------------------------ #
    # Q3
    # ------------------------------------------------------------------ #
    parts.append(h_xml(1, "Axis Q3 — Neural prior alignment"))
    parts.append(h_xml(2, "Q3. Method (detailed)"))
    parts.append(p_xml(
        "Q3 asks which model’s latent belief best tracks a neural summary of the mouse prior in primary ROIs."
    ))
    parts.append(ul_xml([
        "Mouse prior p̂_t as in §6.6 (behavior-only; no model involved).",
        "Neural features: peri-stimulus spike counts in [−0.1, 0.3) s from stimulus onset, units assigned "
        "to MOs / ORBvl (vlOFC) / ACAd / MOp.",
        "Neural prior axis û_t: 5-fold CV ridge regression from counts → p̂_t; out-of-fold predictions "
        "define û_t on held-out trials.",
        "Model score: linearly recalibrated variance explained of û_t by model belief q_t "
        "(ve_linear_recal); session-mean aggregation.",
        "All four active models are compared (no behavioral pre-filter).",
        "Survival test: within each region, best vs second by session-mean VE; session bootstrap (B=2000) "
        "for Δ and a two-sided p-value; Holm–Bonferroni across the four ROIs. Survives = Holm p < 0.05.",
        "Plain-language meaning: surviving means the winner’s lead over the runner-up looks real after "
        "session-to-session variability and multi-region testing; non-survival does not mean “no prior "
        "signal,” only that the best-vs-second gap is not yet trustworthy.",
    ]))

    parts.append(h_xml(2, "Q3. Results"))
    if neural is None:
        parts.append(p_xml("Neural VE analyses are not yet available for this cohort."))
    else:
        sess = neural["ve_session_mean"]
        if len(sess.columns):
            means = sess.mean(axis=0).sort_values(ascending=False)
            order = ", ".join(f"{_pretty(m)} ({v:.3f})" for m, v in means.items())
            parts.append(p_xml(
                f"Session-mean VE across available region–session pairs ranks {order} (Figure 13). "
                "GRU BPTT is the strongest aligner of model belief to the neural prior axis."
            ))
            cols = [c for c in ("tanh_bptt", "tanh_pc", "gru", "gru_pc") if c in sess.columns]
            headers = ["Region"] + [_pretty(c) for c in cols]
            rows = []
            for region in ("MOs", "vlOFC_orbvl", "ACAd", "MOp"):
                if region not in sess.index:
                    continue
                rows.append([region.replace("vlOFC_orbvl", "vlOFC")] + [
                    f"{sess.loc[region, c]:.3f}" if pd.notna(sess.loc[region, c]) else "—"
                    for c in cols
                ])
            if rows:
                fr = [0.22] + [0.78 / len(cols)] * len(cols) if cols else [1.0]
                parts.append(table_xml(
                    headers, rows,
                    title="Table 5. Session-mean neural VE by region and model.",
                    col_fracs=fr,
                ))
        parts.append(p_xml(
            "What the survival test means here: Figure 13 says who is ahead on average. Figure 14 asks whether "
            "the winner’s lead over the runner-up is large enough to trust given few sessions and four "
            "regions. We reshuffle sessions many times; if the lead often vanishes, it does not survive. "
            "Green = survives after Holm correction; gray = not yet trustworthy."
        ))
        surv = neural["survival"]
        if len(surv):
            srows = []
            for _, r in surv.iterrows():
                srows.append([
                    str(r.get("region", "")).replace("vlOFC_orbvl", "vlOFC"),
                    _pretty(str(r.get("best_model", ""))) if pd.notna(r.get("best_model")) else "—",
                    _pretty(str(r.get("second_model", ""))) if pd.notna(r.get("second_model")) else "—",
                    f"{float(r.get('delta', float('nan'))):.3f}" if pd.notna(r.get("delta")) else "—",
                    "yes" if bool(r.get("survive_alpha_05")) else "no",
                ])
            parts.append(table_xml(
                ["Region", "Best", "Second", "VE delta", "Survives"],
                srows,
                title="Table 6. Survival of best-vs-second neural VE gaps.",
                col_fracs=[0.22, 0.18, 0.18, 0.20, 0.22],
            ))
            parts.append(p_xml(
                "GRU’s lead over tanh BPTT survives in MOs, vlOFC, and MOp; it does not survive in ACAd "
                "(sparse coverage / tiny gap)."
            ))
    fig("rIdFig13")
    fig("rIdFig14")

    parts.append(h_xml(2, "Q3. Answer"))
    parts.append(p_xml(
        "Answer to Q3. Among the four models, GRU BPTT best aligns belief with neural prior readouts in "
        "the primary ROIs, and that edge over tanh BPTT is statistically supported in MOs, vlOFC, and MOp. "
        "Presentation takeaway: the neural winner is not identical to the real-transfer correctness winner "
        "(GRU PC). Behavior and neural alignment answer different parts of the mechanistic story."
    ))

    # ------------------------------------------------------------------ #
    # Synthesis for presentation agents
    # ------------------------------------------------------------------ #
    parts.append(h_xml(1, "7. Cross-question synthesis (use this as the talk narrative)"))
    parts.append(p_xml(
        "Slide logic. (1) Motif: hidden prior + distributed neural coding → need mechanistic models. "
        "(2) Design: matched tanh/GRU twins under BPTT vs corrected PC; synth train → real+neural test. "
        "(3) Q1: GRU PC wins real correctness overall and post-switch. "
        "(4) Q2: BPTT updates belief harder; GRU belief most mouse-like on the decode probe — dissociation "
        "from Q1. "
        "(5) Q3: GRU BPTT wins neural VE with surviving gaps in MOs/vlOFC/MOp. "
        "(6) Punchline: no single model wins every axis; the scientific value is the dissociation — "
        "correctness, belief-like updating, and neural alignment are separable criteria, and PC vs BPTT "
        "shifts which criterion is optimized."
    ))
    parts.append(p_xml(
        "What not to claim. We do not claim that any model is the brain’s algorithm, that VE is causal "
        "encoding, or that ACAd lacks prior information. We claim relative rankings under locked methods "
        "on a focused ROI set and a shared n=8 cohort."
    ))

    parts.append(h_xml(1, "8. References"))
    parts.append(ul_xml([
        "International Brain Laboratory (2021). Standardized and reproducible measurement of decision-making in mice. eLife.",
        "International Brain Laboratory (2025). A brain-wide map of neural activity during complex behaviour. Nature.",
        "Findling, C. et al. (2025). Brain-wide representations of prior information in mouse decision-making. Nature.",
        "Mante, V. et al. (2013). Context-dependent computation by recurrent dynamics in prefrontal cortex. Nature.",
        "Barak, O. (2017). Recurrent neural networks as versatile tools of neuroscience research.",
        "Whittington, J. C. R. & Bogacz, R. (2017). Predictive coding approximation of backpropagation. Neural Comput.",
        "Jiang, L. P. & Rao, R. P. N. (2022). Predictive coding theories of cortical function.",
        "Millidge, B. et al. (2022). Predictive coding: a theoretical and experimental review.",
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
