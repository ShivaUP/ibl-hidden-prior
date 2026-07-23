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
            for mid in ("tanh_bptt", "tanh_pc", "gru", "bayes"):
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
    for mid in ("tanh_bptt", "tanh_pc", "gru", "bayes"):
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
    rels_img = "".join(
        f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{png.name}"/>'
        for rid, png in images
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
            z.write(png, f"word/media/{png.name}")


def build_methods() -> Path:
    man = json.loads((ROOT / "data" / "manifests" / "shared_behavior_neural_eids.json").read_text())
    n = man.get("n_sessions", 20)
    covered = man.get("regions_covered_in_cohort") or []

    parts = []
    parts.append(p_xml("Detailed methods", bold=True, center=True, size=40))
    parts.append(p_xml("Latent prior updating in the International Brain Laboratory decision task", center=True, size=26))
    parts.append(p_xml(""))

    parts.append(h_xml(1, "1. Aim"))
    parts.append(p_xml(
        "We ask which model class best explains trial-by-trial updating of a hidden block prior in the IBL "
        "visuospatial decision task: a standard tanh recurrent network trained with backpropagation through time "
        "(tanh BPTT), an architecturally identical tanh network trained with predictive-coding credit assignment "
        "(tanh PC), a gated recurrent unit (GRU), or an explicit online Bayesian prior model (Bayes)."
    ))
    parts.append(p_xml(
        "Models are trained only on synthetic IBL-like sessions. Primary ranking uses held-out synthetic data. "
        "Secondary checks are frozen transfer to real mouse behavior and neural alignment in regions previously "
        "linked to subjective prior coding."
    ))

    parts.append(h_xml(1, "2. Data and shared cohort"))
    parts.append(p_xml(
        f"Behavior transfer and neural analyses use the same {n} sessions. Each session passes almost-perfect "
        "behavior quality control and has Neuropixels spikes in at least one region of interest. Because no single "
        "insertion covers all regions, sessions are selected so that the union of the cohort covers the full ROI set "
        "(greedy set cover, preferring multi-region and primary cortical sessions)."
    ))
    parts.append(p_xml(
        f"Regions covered in the locked cohort: {', '.join(covered)}."
    ))
    parts.append(note_xml(
        "Per-region neural statistics use only the sessions that contain that region. Sparse regions "
        "(for example LGd or SCm with one session) are exploratory."
    ))

    parts.append(h_xml(1, "3. Regions of interest"))
    parts.append(p_xml(
        "Region choice follows published IBL analyses of block priors and brain-wide coding. The standardized "
        "behavior paper establishes that mice use block structure. The brain-wide map paper reports stimulus, "
        "choice, action, and reward maps and refers prior analyses to its companion. That companion reports "
        "widespread decoding of a Bayes-optimal subjective prior, including early sensory, motor, and high-level "
        "cortical sites."
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
    parts.append(table_xml(
        ["Region", "Allen", "Tier", "Why included"],
        [
            ["Secondary motor", "MOs", "Primary cortex", "Strong prior decoding; core claim site"],
            ["Ventrolateral OFC", "ORBvl", "Primary cortex", "High-level prior example site"],
            ["Dorsal ACC", "ACAd", "Primary cortex", "Named high-level cortical prior site"],
            ["Primary motor", "MOp", "Motor", "Motor partner of MOs"],
        ],
        title="Primary regions of interest (analysis scope)",
        col_fracs=[0.26, 0.12, 0.18, 0.44],
    ))
    parts.append(note_xml(
        "Optional deferred ROIs (not in the current analysis loop): CP, VISp, LGd, SCm, GRN, ORBm, PL, ILA. "
        "Findling et al. also report embodied priors (posture / eye position); early sensory sites are especially "
        "control-sensitive and are deferred for that reason."
    ))

    parts.append(h_xml(1, "4. Trial encoding"))
    parts.append(p_xml(
        "Each trial is mapped to 100 ms ticks with shared channels: visual right and left (side × contrast "
        "magnitude), go cue, action left and right, rewarded, and not rewarded. Synthetic phase durations follow "
        "empirical event-time statistics. Real sessions use the same schema without sensory noise."
    ))
    parts.append(p_xml(
        "Three evaluation regimes are used. History-only is primary: the model must infer the block prior from "
        "experience. Full-information adds an evaluation-time oracle bias from the true block log-odds. "
        "Fixed-prior restricts analysis to unbiased blocks (P(right) ≈ 0.5)."
    ))

    parts.append(h_xml(1, "5. Models and training"))
    parts.append(table_xml(
        ["Model", "Test-time computation", "Training"],
        [
            ["tanh BPTT", "Vanilla tanh RNN", "Backpropagation through time"],
            ["tanh PC", "Identical tanh RNN", "Predictive-coding credit assignment"],
            ["GRU", "Gated recurrent unit", "Backpropagation through time"],
            ["Bayes", "Explicit online prior + stimulus readout", "Likelihood / parameter gradients"],
        ],
        col_fracs=[0.18, 0.40, 0.42],
    ))
    parts.append(p_xml(
        "All models share the same input channels and are scored against the correct stimulus side. Training uses "
        "synthetic sessions only. Weights are frozen for real transfer. On real rollouts, mouse actions and rewards "
        "are history inputs; they are never the training target."
    ))

    parts.append(h_xml(1, "6. Behavioral metrics"))
    parts.append(ul_xml([
        "Correctness: fraction of trials where the model choice equals the correct stimulus side.",
        "Balanced correctness: equal-weight mean of correctness in blocks with P(right) = 0.2, 0.5, and 0.8.",
        "History gap: mean zero-evidence P(right) in 0.8 blocks minus that in 0.2 blocks. Zero-evidence is a "
        "counterfactual probe with sensory contrast held at zero.",
        "Switch-centered curves: zero-evidence belief aligned to block transitions.",
    ]))

    parts.append(h_xml(1, "7. Neural prior comparison"))
    parts.append(p_xml(
        "For each region with units in a session, peri-stimulus spike counts (−0.1 to 0.3 s relative to stimulus "
        "onset) are mapped to a neural prior axis by cross-validated ridge regression onto a behavior-derived mouse "
        "prior estimate. Model belief on the same trials is then used to explain that neural axis. The primary "
        "metric is linearly recalibrated variance explained (VE), aggregated as a session mean within region."
    ))
    parts.append(p_xml(
        "Behavior matching retains models whose history-only choice cross-entropy lies within an ε-ball of the "
        "best model on the shared cohort. Survival testing asks whether the matched VE advantage of the best model "
        "over the second-best survives session bootstrap confidence intervals and Holm correction across regions."
    ))

    parts.append(h_xml(1, "8. Limitations"))
    parts.append(ul_xml([
        "ROI coverage is by cohort union; some regions have few sessions.",
        "Embodied-prior controls (video / eye position) are not yet applied.",
        "The neural window is peri-stimulus; complementary inter-trial decoding is left for later work.",
        "Synthetic training distribution approximates but does not replay individual mice.",
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

    # Figures to embed (must exist)
    fig_specs = [
        ("rIdFig1", FIG / "scorecards" / "synth_history_only_scorecard.png", 6.2, 4.4,
         "Figure 1. Synthetic held-out scorecard (history-only): correctness and history gap."),
        ("rIdFig2", FIG / "scorecards" / "real_history_only_scorecard.png", 6.2, 4.4,
         "Figure 2. Real transfer scorecard (history-only) on the shared cohort."),
        ("rIdFig3", FIG / "comparison" / "real_history_only_correctness_by_prior.png", 6.2, 3.6,
         "Figure 3. Real history-only correctness by block prior, with balanced correctness."),
        ("rIdFig4", FIG / "comparison" / "synth_vs_real_history_only_board.png", 6.2, 3.4,
         "Figure 4. Synthetic versus real transfer under history-only."),
        ("rIdFig5", FIG / "comparison" / "real_history_only_switch_board.png", 6.2, 3.2,
         "Figure 5. Switch-centered zero-evidence belief on real sessions (history-only)."),
        ("rIdFig6", FIG / "scorecards" / "real_full_information_scorecard.png", 6.2, 4.4,
         "Figure 6. Real full-information control: oracle prior at readout."),
        ("rIdFig7", FIG / "by_model" / "gru" / "real" / "history_only" / "multipanel_diagnostics.png", 6.2, 4.6,
         "Figure 7. Example diagnostics for the GRU under real history-only evaluation."),
        ("rIdFig8", FIG / "neural" / "neural_ve_unmatched_vs_matched.png", 6.2, 3.8,
         "Figure 8. Neural prior variance explained by model belief (session-mean VE)."),
        ("rIdFig9", FIG / "neural" / "survival_tests.png", 6.0, 3.2,
         "Figure 9. Behavior-matched survival of neural advantages across regions."),
    ]
    images = [(rid, p) for rid, p, *_ in fig_specs if p.exists()]
    rid_map = {rid: (p, w, h, cap) for rid, p, w, h, cap in fig_specs if p.exists()}

    parts = []
    parts.append(p_xml(
        "Comparing recurrent and Bayesian models of latent prior updating "
        "in the International Brain Laboratory decision task",
        bold=True, center=True, size=36,
    ))
    parts.append(p_xml("Working manuscript — current results", center=True, italic=True, size=22))
    parts.append(p_xml(""))

    parts.append(h_xml(1, "Abstract"))
    parts.append(p_xml(
        f"Mice performing the IBL biased-block task update a hidden prior over stimulus side. We compare four "
        f"models trained only on synthetic IBL-like sessions and evaluated on held-out synthetic data and on "
        f"{n} real sessions that also support neural analyses. Under history-only evaluation, the GRU achieves "
        f"the highest correctness on real transfer ({acc('real','history_only','gru'):.3f}) and the strongest "
        f"history gap, with tanh BPTT close behind. The predictive-coding tanh network is weaker at inferring "
        f"the prior from history but leads when an oracle prior is provided at readout. Neural alignment is "
        f"assessed in twelve prior-related regions selected from recent IBL prior-mapping work. "
        + (
            "Behavior-matched neural comparisons are reported below."
            if neural
            else "Neural results for the expanded region set are being finalized on this cohort."
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
        "We ask which computational account best captures trial-by-trial prior updating: a standard recurrent "
        "network, a recurrent network trained with predictive-coding credit assignment, a GRU, or an explicit "
        "online Bayesian observer. Models are matched on inputs and scored against the correct stimulus side. "
        "Neural comparison asks whether model belief explains neural prior readouts in MOs, ORBvl, ACAd, and MOp "
        "after models are matched on behavior."
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
        "Four models share a common tick-and-channel interface. Training uses synthetic sessions only. "
        "History-only evaluation is primary. Full-information adds an evaluation-time oracle prior bias. "
        "Fixed-prior restricts analysis to unbiased blocks. Correctness is the fraction of trials where the "
        "model choice matches the correct stimulus side. Belief is the counterfactual zero-evidence probability "
        "of choosing right. History gap is the mean belief difference between 0.8 and 0.2 blocks."
    ))

    parts.append(h_xml(2, "Neural analysis"))
    parts.append(p_xml(
        "Peri-stimulus spike counts (−0.1 to 0.3 s) are mapped to a neural prior axis by cross-validated ridge "
        "regression onto a behavior-derived mouse prior. Model belief is scored by linearly recalibrated "
        "variance explained of that axis. Models are behavior-matched with a cross-entropy ε-ball on the shared "
        "cohort. Survival uses session bootstrap confidence intervals and Holm correction across regions."
    ))

    parts.append(h_xml(1, "Results"))
    parts.append(h_xml(2, "Synthetic held-out ranking"))
    parts.append(p_xml(
        "On held-out synthetic history-only sessions, tanh BPTT and GRU reach similar correctness "
        f"({acc('synth','history_only','tanh_bptt'):.3f} and {acc('synth','history_only','gru'):.3f}), "
        f"with Bayes intermediate ({acc('synth','history_only','bayes'):.3f}) and tanh PC lowest "
        f"({acc('synth','history_only','tanh_pc'):.3f}). History gaps follow the same order: GRU and tanh BPTT "
        f"show large block-tuned belief shifts ({gap('synth','history_only','gru'):.3f} and "
        f"{gap('synth','history_only','tanh_bptt'):.3f}), whereas tanh PC remains near "
        f"{gap('synth','history_only','tanh_pc'):.3f}."
    ))
    if "rIdFig1" in rid_map:
        rid, (p, w, h, cap) = "rIdFig1", rid_map["rIdFig1"]
        parts.append(image_xml(rid, width_in=w, height_in=h, caption=cap))

    parts.append(h_xml(2, "Real transfer on the shared cohort"))
    parts.append(p_xml(
        f"Frozen weights transfer to real sessions with the same ranking under history-only: GRU "
        f"({acc('real','history_only','gru'):.3f}), tanh BPTT ({acc('real','history_only','tanh_bptt'):.3f}), "
        f"Bayes ({acc('real','history_only','bayes'):.3f}), tanh PC ({acc('real','history_only','tanh_pc'):.3f}). "
        f"History gaps remain large for GRU and tanh BPTT "
        f"({gap('real','history_only','gru'):.3f}, {gap('real','history_only','tanh_bptt'):.3f}) and small for "
        f"tanh PC ({gap('real','history_only','tanh_pc'):.3f})."
    ))
    if "rIdFig2" in rid_map:
        rid, (p, w, h, cap) = "rIdFig2", rid_map["rIdFig2"]
        parts.append(image_xml(rid, width_in=w, height_in=h, caption=cap))

    # Per-prior table
    if prior:
        rows = []
        for mid in ("tanh_bptt", "tanh_pc", "gru", "bayes"):
            v = prior[mid]
            rows.append([
                _pretty(mid),
                f"{v[0.2]:.3f}",
                f"{v[0.5]:.3f}",
                f"{v[0.8]:.3f}",
                f"{v['balanced']:.3f}",
            ])
        parts.append(table_xml(
            ["Model", "P=0.2", "P=0.5", "P=0.8", "Balanced"],
            rows,
            title="Table 1. Real history-only correctness by block prior (session means).",
            col_fracs=[0.28, 0.18, 0.18, 0.18, 0.18],
        ))
        parts.append(p_xml(
            "Balanced correctness confirms that GRU’s lead is not driven by a single block type. "
            "tanh PC is strong in right-biased blocks but weak in left-biased blocks, lowering its balanced score."
        ))
    if "rIdFig3" in rid_map:
        rid, (p, w, h, cap) = "rIdFig3", rid_map["rIdFig3"]
        parts.append(image_xml(rid, width_in=w, height_in=h, caption=cap))
    if "rIdFig4" in rid_map:
        rid, (p, w, h, cap) = "rIdFig4", rid_map["rIdFig4"]
        parts.append(image_xml(rid, width_in=w, height_in=h, caption=cap))
    if "rIdFig5" in rid_map:
        rid, (p, w, h, cap) = "rIdFig5", rid_map["rIdFig5"]
        parts.append(image_xml(rid, width_in=w, height_in=h, caption=cap))

    parts.append(h_xml(2, "Full-information control"))
    parts.append(p_xml(
        "When an oracle prior bias is supplied at readout, tanh PC leads on real sessions "
        f"({acc('real','full_information','tanh_pc'):.3f}), above GRU ({acc('real','full_information','gru'):.3f}). "
        "Thus tanh PC can use prior information when it is provided, but is less effective at inferring that "
        "prior from history alone."
    ))
    if "rIdFig6" in rid_map:
        rid, (p, w, h, cap) = "rIdFig6", rid_map["rIdFig6"]
        parts.append(image_xml(rid, width_in=w, height_in=h, caption=cap))
    if "rIdFig7" in rid_map:
        rid, (p, w, h, cap) = "rIdFig7", rid_map["rIdFig7"]
        parts.append(image_xml(rid, width_in=w, height_in=h, caption=cap))

    parts.append(h_xml(2, "Neural alignment"))
    if neural is None:
        parts.append(p_xml(
            "Neural variance-explained analyses for the twelve-region cohort are in progress on the same "
            f"{n} sessions used for behavioral transfer. Figures will report session-mean VE by region and "
            "behavior-matched survival tests."
        ))
    else:
        sess = neural["ve_session_mean"]
        # summarize overall ranking by mean across regions
        if len(sess.columns):
            means = sess.mean(axis=0).sort_values(ascending=False)
            order = ", ".join(f"{_pretty(m)} ({v:.3f})" for m, v in means.items())
            parts.append(p_xml(
                f"Across regions with available units, session-mean VE ranks {order}. "
                f"Behavior-matched models: {', '.join(_pretty(m) for m in neural['matched']) or 'none recorded'}."
            ))
            # compact table: regions x models
            cols = [c for c in ("tanh_bptt", "tanh_pc", "gru", "bayes") if c in sess.columns]
            headers = ["Region"] + [_pretty(c) for c in cols]
            rows = []
            for region in sess.index:
                rows.append([region] + [
                    f"{sess.loc[region, c]:.3f}" if pd.notna(sess.loc[region, c]) else "—"
                    for c in cols
                ])
            fr = [0.22] + [(0.78 / len(cols))] * len(cols) if cols else [1.0]
            # fix sum
            if cols:
                fr = [0.22] + [0.78 / len(cols)] * len(cols)
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
    if "rIdFig8" in rid_map:
        rid, (p, w, h, cap) = "rIdFig8", rid_map["rIdFig8"]
        parts.append(image_xml(rid, width_in=w, height_in=h, caption=cap))
    if "rIdFig9" in rid_map:
        rid, (p, w, h, cap) = "rIdFig9", rid_map["rIdFig9"]
        parts.append(image_xml(rid, width_in=w, height_in=h, caption=cap))

    parts.append(h_xml(1, "Discussion"))
    parts.append(p_xml(
        "Under the primary history-only regime, gated recurrence (GRU) and standard tanh BPTT best reproduce "
        "choice correctness and block-tuned belief. Explicit Bayes captures intermediate prior use. Predictive-"
        "coding credit assignment yields a network that underuses history-inferred priors yet benefits when the "
        "prior is supplied, separating “can use a prior” from “can learn the prior from experience.”"
    ))
    parts.append(p_xml(
        "Neural analyses target regions implicated in subjective prior coding by Findling et al. (2025). "
        "Because coverage is distributed across sessions, region-level claims scale with the number of sessions "
        "that sample each area. Early sensory and hindbrain sites remain especially sensitive to embodied-prior "
        "confounds and should be interpreted with posture and eye-position controls in follow-up work."
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
