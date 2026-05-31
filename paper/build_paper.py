"""Build the project paper as a professional PDF (reportlab).

Pulls exact numbers from results.json and figures from ../figures and
./figures, and writes paper/Tactical_Topology.pdf.
Run: python paper/build_paper.py
"""

import json
import os

from PIL import Image as PILImage
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem,
                                Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures")
PFIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "Tactical_Topology.pdf")

with open(os.path.join(HERE, "results.json"), encoding="utf-8") as fh:
    R = json.load(fh)

NAVY = HexColor("#1a1a2e")
BLUE = HexColor("#457b9d")
RED = HexColor("#e63946")
GREY = HexColor("#555555")
LIGHT = HexColor("#eef2f7")

# --- styles ---------------------------------------------------------------
S = {}
S["title"] = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=19,
                            leading=23, alignment=TA_CENTER, textColor=NAVY,
                            spaceAfter=6)
S["author"] = ParagraphStyle("author", fontName="Helvetica", fontSize=11,
                             leading=14, alignment=TA_CENTER, spaceAfter=2)
S["meta"] = ParagraphStyle("meta", fontName="Helvetica", fontSize=9,
                           leading=12, alignment=TA_CENTER, textColor=GREY,
                           spaceAfter=12)
S["abstract"] = ParagraphStyle("abstract", fontName="Times-Roman", fontSize=9.5,
                               leading=13, alignment=TA_JUSTIFY, leftIndent=24,
                               rightIndent=24, spaceAfter=4)
S["abshead"] = ParagraphStyle("abshead", fontName="Helvetica-Bold", fontSize=10,
                              leading=13, alignment=TA_CENTER, spaceAfter=3)
S["kw"] = ParagraphStyle("kw", fontName="Times-Italic", fontSize=9, leading=12,
                         alignment=TA_JUSTIFY, leftIndent=24, rightIndent=24,
                         spaceAfter=10, textColor=GREY)
S["h1"] = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=12.5,
                         leading=15, textColor=NAVY, spaceBefore=12, spaceAfter=5)
S["h2"] = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.5,
                         leading=13, textColor=HexColor("#33415c"),
                         spaceBefore=7, spaceAfter=3)
S["body"] = ParagraphStyle("body", fontName="Times-Roman", fontSize=10,
                           leading=13.5, alignment=TA_JUSTIFY, spaceAfter=6)
S["cap"] = ParagraphStyle("cap", fontName="Times-Italic", fontSize=8.5,
                          leading=11, alignment=TA_CENTER, textColor=GREY,
                          spaceBefore=3, spaceAfter=10)
S["ref"] = ParagraphStyle("ref", fontName="Times-Roman", fontSize=8.7,
                          leading=11.5, alignment=TA_JUSTIFY, leftIndent=16,
                          firstLineIndent=-16, spaceAfter=3)
S["eq"] = ParagraphStyle("eq", fontName="Times-Italic", fontSize=9.7,
                         leading=13, alignment=TA_CENTER, spaceBefore=2,
                         spaceAfter=6, textColor=HexColor("#222222"))

story = []


def P(text, style="body"):
    story.append(Paragraph(text, S[style]))


def H1(text):
    story.append(Paragraph(text, S["h1"]))


def H2(text):
    story.append(Paragraph(text, S["h2"]))


def EQ(text):
    story.append(Paragraph(text, S["eq"]))


def figure(path, caption, max_w=468, frac=1.0):
    iw, ih = PILImage.open(path).size
    w = max_w * frac
    h = w * ih / iw
    story.append(KeepTogether([
        Image(path, width=w, height=h, hAlign="CENTER"),
        Paragraph(caption, S["cap"]),
    ]))


# ==========================================================================
# Front matter
# ==========================================================================
P("Tactical Topology: Modelling Football Teams as Communication Networks", "title")
P("Nikos Raptis", "author")
P("Independent Research &nbsp;&middot;&nbsp; 2026 &nbsp;&middot;&nbsp; "
  "github.com/nraptisss/Football-as-a-Communication-Network", "meta")

P("Abstract", "abshead")
P(
    "We recast a football team's passing as a <b>communication network</b> and "
    "show that metrics from telecommunications and graph theory &mdash; "
    "fault tolerance, maximum flow and minimum cut, processing latency, and "
    "channel interference &mdash; map naturally onto recognisable tactical "
    "concepts. Using the freely available StatsBomb event data for the entire "
    "2015/16 Spanish La Liga season ({n_matches} matches, {n_rows} match&ndash;"
    "team observations), we build weighted directed passing networks and define "
    "four interpretable metrics on them: a node-failure <i>resilience</i> score "
    "that identifies a team's structurally critical players; a zone-level "
    "<i>max-flow / min-cut</i> measure of how directly a team progresses the "
    "ball; a <i>tempo</i> (latency) measure of how quickly players relay the "
    "ball; and a <i>pressing-as-interference</i> measure of how opponent "
    "pressure degrades passing quality. We then layer three deliberately "
    "interpretable machine-learning models on the resulting features: "
    "unsupervised tactical clustering, supervised match-outcome prediction from "
    "first-half structure only, and change-point detection of in-match tactical "
    "shifts. The metrics behave as football intuition predicts &mdash; the most "
    "critical Barcelona players are its midfield and ball-playing defenders, "
    "pass completion falls from {free}% to {pressed}% under pressure, and "
    "direct/counter-attacking sides score higher on progression directness than "
    "possession sides. Clustering cleanly separates the elite possession teams "
    "from the rest, and first-half network structure predicts the full-time "
    "result with {acc}% cross-validated accuracy against a {base}% majority "
    "baseline. All code, data and figures are open source and reproducible.".format(
        n_matches=R["n_matches"], n_rows=R["n_match_team_rows"],
        free=int(R["pressure"]["free_completion"] * 100),
        pressed=int(R["pressure"]["pressed_completion"] * 100),
        acc=int(R["outcome"]["cv_accuracy_mean"] * 100),
        base=int(R["outcome"]["majority_baseline"] * 100),
    ),
    "abstract",
)
P("Keywords: football analytics; network science; passing networks; "
  "communication networks; max-flow; graph resilience; machine learning; "
  "StatsBomb.", "kw")

# ==========================================================================
# 1 Introduction
# ==========================================================================
H1("1&nbsp;&nbsp;Introduction")
P(
    "Association football is, at its core, a problem of moving a single token "
    "&mdash; the ball &mdash; through a contested space using a network of "
    "eleven cooperating agents. Viewed this way, a team is strikingly similar "
    "to a <b>communication network</b>: players are nodes that receive, process "
    "and forward a signal, and passes are the messages that travel between them. "
    "This analogy is more than rhetorical. Decades of work in network science "
    "and telecommunications have produced precise, well-understood tools for "
    "exactly the questions coaches care about: which nodes are critical and what "
    "happens if they fail; how much traffic a network can route from one region "
    "to another and where the bottleneck lies; how quickly nodes relay messages; "
    "and how external interference degrades throughput.")
P(
    "Passing networks have been studied since the foundational work of Gould and "
    "Gatrell [1] and popularised for the modern game by L&oacute;pez Pe&ntilde;a "
    "and Touchette [2] and Buld&uacute; and colleagues [3, 4]. Most of this "
    "literature characterises network <i>structure</i> &mdash; centrality, "
    "clustering, motifs &mdash; and relates it to style or quality. Our "
    "contribution is to take the communication-network analogy literally and "
    "<b>operationalise four telecommunications concepts as a coherent, "
    "reusable toolkit</b>, then to validate them against football intuition and "
    "use them as features for interpretable machine learning. Figure 1 "
    "summarises the mapping.")
figure(os.path.join(PFIG, "fig_concept_mapping.png"),
       "Figure 1. The conceptual mapping at the heart of this work. Each "
       "football object on the left has a precise counterpart in communication-"
       "network theory on the right; the right-hand concepts (in red) are the "
       "four metrics we implement and validate.")
P("Concretely, the paper makes three contributions. (i) We define and implement "
  "four communication-network metrics on football passing graphs &mdash; "
  "resilience, max-flow/min-cut, tempo and pressing-interference &mdash; with "
  "explicit, parameterised definitions. (ii) We validate each against known "
  "football facts on a fixed set of ground-truth matches. (iii) We build three "
  "interpretable ML models on top of the metrics and report honest, "
  "cross-validated performance on a full league season. The entire pipeline is "
  "open source and reproducible from public data with no credentials.")

# ==========================================================================
# 2 Related work
# ==========================================================================
H1("2&nbsp;&nbsp;Related Work")
P(
    "<b>Passing networks.</b> Gould and Gatrell [1] first analysed a football "
    "match as a network. L&oacute;pez Pe&ntilde;a and Touchette [2] formalised "
    "weighted, directed passing networks for the 2010 World Cup and used "
    "centrality to rank players and probe the effect of removing them. "
    "Buld&uacute; et al. [3, 4] conducted an extensive network-science study of "
    "Guardiola's Barcelona, reporting its unusually high clustering and "
    "balanced flow across pitch zones. Gyarmati et al. [5] characterised team "
    "style through <i>flow motifs</i> (recurring three- and four-pass "
    "sub-sequences), and Cintia et al. [6] linked network indicators to team "
    "performance.")
P(
    "<b>Robustness.</b> Closest to our resilience metric, Ichinose et al. [7] "
    "removed nodes and links from J-League passing networks and found that, like "
    "scale-free communication networks [12], they are robust to random failure "
    "but vulnerable to targeted attack &mdash; the network-science framing we "
    "adopt directly. <b>Flow and tempo.</b> Bekkers and Dabadghao [8] studied "
    "passing flow at the motif level; we instead aggregate flow spatially and "
    "apply the classical max-flow/min-cut theorem [13] to locate progression "
    "bottlenecks. <b>Prediction.</b> A growing body of work feeds network "
    "features to machine-learning models to predict match outcomes [9]; we "
    "follow this line but deliberately restrict ourselves to interpretable "
    "models and first-half-only features.")
P(
    "<b>Data.</b> Open event datasets such as Pappalardo et al. [10] and "
    "StatsBomb Open Data [11] have made this research reproducible; we use the "
    "latter. Relative to prior work, our novelty is the <i>unification</i> of "
    "fault-tolerance, max-flow, latency and interference as a single "
    "communication-network toolkit, validated metric-by-metric and wired into an "
    "interpretable ML layer.")

# ==========================================================================
# 3 Data
# ==========================================================================
H1("3&nbsp;&nbsp;Data")
P(
    "We use StatsBomb Open Data [11] for the 2015/16 Spanish La Liga season "
    "(competition 11, season 27): {n} matches with full event data, two teams "
    "each, giving {r} match&ndash;team observations. Events are accessed through "
    "the <font face='Courier'>statsbombpy</font> library, which requires no "
    "credentials. From the raw events we retain passes and apply a fixed "
    "cleaning contract: drop passes with missing origin or destination "
    "coordinates; keep an explicit integer team identifier alongside team name; "
    "restrict to the first and second half by default; convert the per-period "
    "<font face='Courier'>HH:MM:SS</font> timestamp to a continuous match clock "
    "in seconds; keep incomplete passes (they matter for resilience) but flag "
    "completion as a Boolean; and coerce the under-pressure flag to Boolean. "
    "For deterministic validation we fix five ground-truth matches spanning "
    "Barcelona, Atl&eacute;tico Madrid and Real Madrid, used throughout for "
    "sanity checks and unit tests.".format(
        n=R["n_matches"], r=R["n_match_team_rows"]))

# ==========================================================================
# 4 Methods
# ==========================================================================
H1("4&nbsp;&nbsp;Methods")
figure(os.path.join(PFIG, "fig_system_architecture.png"),
       "Figure 2. The analysis pipeline. Cleaned events become passing networks, "
       "which are scored by the four communication-network metrics and fed to "
       "three machine-learning models; all intermediate artefacts are cached.")

H2("4.1&nbsp;&nbsp;Passing-network construction")
P(
    "For one team in one match we build a weighted directed graph in which nodes "
    "are players and a directed edge from passer to receiver carries a weight "
    "equal to the number of <i>completed</i> passes between that ordered pair. "
    "Self-passes are removed and substitutes are retained as nodes so that their "
    "limited involvement is reflected rather than hidden. We use the StatsBomb "
    "pitch coordinate system (120&times;80) unchanged, placing each node at the "
    "player's mean pass-origin location. A <i>dynamic</i> variant slides a "
    "five-minute window in one-minute steps to produce a time series of graphs. "
    "For each graph we summarise density, average clustering, and three "
    "complementary 'key player' views: betweenness [14] (the bridging router), "
    "weighted eigenvector centrality [15] (a hub among hubs), and weighted "
    "PageRank [16] (a player fed by influential team-mates). Graphs are built "
    "with NetworkX [17].")

H2("4.2&nbsp;&nbsp;Communication-network metrics")
P("<b>Resilience (fault tolerance).</b> Inspired by attack-tolerance analysis "
  "of complex networks [7, 12], we remove each player in turn and measure the "
  "resulting degradation in network density, average clustering, and the number "
  "of weakly connected components (fragmentation). Each impact is min-max "
  "normalised across players and averaged, together with the player's "
  "centrality, into a single score in [0, 1]:")
EQ("S(v) = mean[ norm(&Delta;density), norm(&Delta;clustering), "
   "norm(&Delta;components), norm(centrality) ]")
P("A higher score marks a more structurally critical player. <b>Max-flow / "
  "min-cut (throughput).</b> We coarse-grain the pitch into a 5&times;3 grid and "
  "build a zone graph whose edge weights count completed passes between zones. "
  "Adding a super-source over the three defensive-third zones and a super-sink "
  "over the three attacking-third zones, the max-flow/min-cut theorem [13] gives "
  "the maximum ball throughput from defence to attack and the bottleneck zones "
  "that limit it. We define <i>attacking flow efficiency</i> as the max-flow "
  "value divided by total completed passes &mdash; a measure of progression "
  "<i>directness per pass</i>.")
P("<b>Tempo (latency).</b> Treating each player as a processing node, we measure "
  "the time between receiving a pass and playing the next one. For every "
  "reception we take the player's next completed pass <i>in the same period</i> "
  "and keep the delta if it is at most 30 seconds, guarding against the "
  "half-time discontinuity in the continuous clock. We report per-player mean "
  "and median touch time, a low-latency ratio (fraction of touches under two "
  "seconds), and a touch-weighted team tempo. <b>Pressing as interference.</b> "
  "We split a team's passes into under-pressure and pressure-free sub-networks "
  "and compare their summaries, reporting the density degradation and the drop "
  "in completion rate &mdash; the signal loss induced by opponent 'noise'.")

H2("4.3&nbsp;&nbsp;Machine-learning models")
P(
    "For every match and team we assemble a feature vector of twelve numeric "
    "features: density, average clustering, the value of the top betweenness, "
    "eigenvector and PageRank nodes, node and edge counts, attacking flow "
    "efficiency, team tempo, low-latency ratio, and the two pressure-degradation "
    "measures. Crucially we store centrality <i>values</i>, not the player-name "
    "strings, so that the features are model-ready; names are kept only for "
    "display. Missing values (e.g. when a team has no under-pressure passes) are "
    "median-imputed. We then train three models with scikit-learn [18]. "
    "<i>Tactical clustering</i> applies k-means (k = 4) to the season-average "
    "feature vector of each team after standardisation &mdash; standardisation "
    "is essential because the features live on very different scales. "
    "<i>Outcome prediction</i> trains a random forest to predict the full-time "
    "result (win/draw/loss) from <b>first-half features only</b>, evaluated with "
    "stratified cross-validation; using only the first half makes the task "
    "harder and more interesting than using the whole match. <i>Tactical-shift "
    "detection</i> z-scales the [density, top-betweenness, node-count] vector of "
    "each dynamic window and flags timestamps where the cosine distance between "
    "consecutive windows exceeds the mean plus a threshold times the standard "
    "deviation.")

# ==========================================================================
# 5 Results
# ==========================================================================
H1("5&nbsp;&nbsp;Results")

H2("5.1&nbsp;&nbsp;Network structure")
P("The constructed networks have the expected shape: 13&ndash;14 nodes per team, "
  "densities from about 0.28 for low-possession sides up to 0.64 for Barcelona, "
  "and a clear high-betweenness router. Figure 3 shows Barcelona's network in "
  "the 4&ndash;0 win at Real Madrid, with Iniesta &mdash; the top-betweenness "
  "node that match &mdash; highlighted. The layout reproduces a recognisable "
  "possession shape rather than a featureless blob.")
figure(os.path.join(FIG, "fig_02_passing_network_pitch.png"),
       "Figure 3. Barcelona's passing network (El Cl&aacute;sico, 2015/16) drawn "
       "on the pitch. Node size is betweenness centrality, edge width is pass "
       "volume, edge colour is completion rate; gold marks the highlighted "
       "router (Iniesta).")

H2("5.2&nbsp;&nbsp;Resilience")
P("The resilience metric passes its sanity check decisively. For Barcelona, the "
  "three most critical players are {a}, {b} and {c} &mdash; a ball-playing "
  "full-back and the two midfield organisers &mdash; rather than orthodox "
  "centre-backs, exactly the profile expected of a possession team (Figure 4). "
  "This mirrors the targeted-attack vulnerability reported for real passing "
  "networks [7].".format(
      a=R["resilience_top3_clasico"][0].split()[1],   # Alves
      b=R["resilience_top3_clasico"][1].split()[1],   # Iniesta
      c=R["resilience_top3_clasico"][2].split()[1]))  # Busquets
figure(os.path.join(FIG, "fig_03_resilience_bar.png"),
       "Figure 4. Resilience score per Barcelona player (El Cl&aacute;sico). The "
       "three most critical nodes are gold; removing them degrades the network "
       "most.", frac=0.78)

H2("5.3&nbsp;&nbsp;Flow and directness")
P("Attacking flow efficiency behaves as a measure of <i>directness</i>, not of "
  "attacking quality. Direct and counter-attacking sides score highest, while "
  "possession Barcelona ({barca}) sits below direct teams such as "
  "Atl&eacute;tico ({atl}) and well below the most direct sides in the league. "
  "This is the correct, expected behaviour: possession teams recycle many "
  "lateral passes, inflating the denominator, so their progression per pass is "
  "lower. Figure 5 contrasts the two styles on the same matchday; the red band "
  "marks the defensive-third min-cut the ball must funnel through.".format(
      barca=R["flow_efficiency"]["barcelona"], atl=R["flow_efficiency"]["atletico"]))
figure(os.path.join(FIG, "fig_03_zone_flow_heatmap.png"),
       "Figure 5. Zone pass-flow intensity for Barcelona and Atl&eacute;tico in "
       "the same match. Brighter zones see more passing; the red-outlined zones "
       "are the defence-to-attack bottleneck (min-cut).")

H2("5.4&nbsp;&nbsp;Pressing degradation")
P("Pressure clearly degrades the passing signal. Pooled across the ground-truth "
  "matches, completion falls from {free}% when free to {pressed}% under pressure "
  "&mdash; a drop of {drop} percentage points &mdash; reproducing a well-known "
  "football fact and confirming the pressure flag behaves correctly. The radar "
  "chart in Figure 6 shows the under-pressure sub-network (red) shrinking inside "
  "the free network (blue) on density, clustering and connectivity.".format(
      free=int(R["pressure"]["free_completion"] * 100),
      pressed=int(R["pressure"]["pressed_completion"] * 100),
      drop=R["pressure"]["drop_pp"]))
figure(os.path.join(FIG, "fig_03_pressure_comparison.png"),
       "Figure 6. Barcelona's passing network free versus under pressure. The "
       "red (pressed) shape sitting inside the blue (free) shape visualises the "
       "interference-driven degradation.", frac=0.62)

H2("5.5&nbsp;&nbsp;Tactical clustering")
c = R["clusters"]
P("Clustering the season-average feature vector of each team into four groups "
  "(Figure 7) produces tactically coherent tiers. Cluster&nbsp;0 isolates the "
  "two possession giants ({c0}); cluster&nbsp;3 is a distinctive direct group "
  "({c3}); the remaining clusters split the rest of the league into an organised "
  "mid-table band and a lower-table direct band. Barcelona and Atl&eacute;tico "
  "land in different clusters, as football intuition demands.".format(
      c0=", ".join(c["0"]), c3=", ".join(c["3"])))
figure(os.path.join(FIG, "fig_04_cluster_scatter.png"),
       "Figure 7. Teams projected to two PCA dimensions and coloured by tactical "
       "cluster. Villarreal (annotated) is a mild outlier &mdash; its network "
       "metrics, not its league finish, place it in the lower-table-direct "
       "cluster.")

H2("5.6&nbsp;&nbsp;Match-outcome prediction")
P("Predicting the full-time result from first-half network structure alone "
  "reaches {acc}% &plusmn; {sd}% cross-validated accuracy with a weighted F1 of "
  "{f1}, against a {base}% majority-class baseline and a 33% random baseline "
  "(Table 1). For a deliberately interpretable model fed only structural "
  "first-half features, beating the majority baseline is a meaningful and "
  "honest result. Feature importances are spread fairly evenly, led by average "
  "clustering, attacking flow efficiency and team tempo &mdash; structure, "
  "directness and speed &mdash; which is reassuringly sensible.".format(
      acc=int(R["outcome"]["cv_accuracy_mean"] * 100),
      sd=int(R["outcome"]["cv_accuracy_std"] * 100),
      f1=R["outcome"]["cv_f1_weighted"],
      base=int(R["outcome"]["majority_baseline"] * 100)))

# Table 1: outcome metrics
t1 = [["Metric", "Value"],
      ["Cross-validated accuracy", "{:.1f}%".format(R["outcome"]["cv_accuracy_mean"] * 100)],
      ["Accuracy std. (folds)", "{:.1f}%".format(R["outcome"]["cv_accuracy_std"] * 100)],
      ["Weighted F1", "{:.3f}".format(R["outcome"]["cv_f1_weighted"])],
      ["Majority baseline", "{:.1f}%".format(R["outcome"]["majority_baseline"] * 100)],
      ["Random baseline (3-class)", "33.3%"],
      ["Top features", "avg. clustering, flow efficiency, team tempo"]]
tbl1 = Table(t1, colWidths=[180, 280], hAlign="CENTER")
tbl1.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
    ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), LIGHT]),
    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
    ("ALIGN", (1, 0), (1, -1), "LEFT"),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(KeepTogether([tbl1,
    Paragraph("Table 1. Match-outcome model performance (random forest, "
              "first-half features, stratified cross-validation).", S["cap"])]))

H2("5.7&nbsp;&nbsp;Tactical-shift detection")
sh = R["shifts"]
P("On a manually inspected match (Barcelona vs Sevilla), all {m} of {t} detected "
  "change-points fall within three minutes of a substitution or goal "
  "(Figure 8). Detection quality varies by match &mdash; an unsupervised "
  "detector also flags genuine phase changes that are not subs or goals &mdash; "
  "which is why such validation is done on a human-chosen match.".format(
      m=sh["matched"], t=sh["total"]))
figure(os.path.join(FIG, "fig_04_dynamic_metrics.png"),
       "Figure 8. Barcelona's network density and top betweenness over the match. "
       "Red dashed lines are automatically detected tactical shifts; grey dotted "
       "lines are substitutions and goals.")

# ==========================================================================
# 6 Discussion
# ==========================================================================
H1("6&nbsp;&nbsp;Discussion")
P("The central finding is that the communication-network analogy is not merely "
  "evocative but <i>operationally faithful</i>: each borrowed metric lands on "
  "the player or pattern that football experts would nominate. Fault-tolerance "
  "analysis surfaces the metronomes and ball-playing defenders; max-flow "
  "recovers the intuition that possession and directness are distinct axes; "
  "latency quantifies tempo; and interference captures the cost of pressing. "
  "Because every feature has a concrete meaning, the downstream models are "
  "interpretable by construction &mdash; a clustering tier or an outcome "
  "prediction can always be traced back to a tactical quantity. The "
  "flow-efficiency result is a good example of the analogy doing useful work: "
  "it forces a precise definition (throughput per pass) that cleanly separates "
  "two styles which a vague notion of 'attacking' would conflate.")

# ==========================================================================
# 7 Limitations
# ==========================================================================
H1("7&nbsp;&nbsp;Limitations")
P("The study is descriptive, not causal: network structure correlates with "
  "outcomes but we do not claim it produces them. We analyse a single league "
  "season, and several validations (resilience, shift detection) are inspected "
  "on a small ground-truth set rather than scored at scale. The defensive-third "
  "min-cut is often the same band across teams, so the heatmap's value lies more "
  "in the flow intensities than in the cut location. The continuous match clock "
  "assumes fixed 45-minute halves, which is harmless for within-period deltas "
  "but should not be read across half-time. Finally, the outcome model is "
  "honestly modest; richer features (possession value, expected goals) would "
  "likely help but were deliberately excluded to keep the study network-pure.")

# ==========================================================================
# 8 Conclusion
# ==========================================================================
H1("8&nbsp;&nbsp;Conclusion")
P("Treating a football team as a communication network yields a small set of "
  "precise, interpretable metrics &mdash; resilience, max-flow, tempo and "
  "interference &mdash; that align with expert intuition and serve as honest "
  "features for tactical clustering, outcome prediction and shift detection. The "
  "approach is fully reproducible from open data. Future work includes "
  "multi-season and multi-league validation, possession-value and "
  "expected-goals integration, and a multilayer treatment that combines passing "
  "with off-ball movement.")

H1("Acknowledgements")
P("This work is built entirely on StatsBomb Open Data [11]; we thank StatsBomb "
  "for making it freely available. Analysis uses NetworkX [17] and "
  "scikit-learn [18].")

# ==========================================================================
# References
# ==========================================================================
H1("References")
refs = [
    "P. Gould and A. Gatrell. A structural analysis of a game: the Liverpool v "
    "Manchester United Cup Final of 1977. <i>Social Networks</i>, 2(3):253&ndash;273, 1979/80.",
    "J. L&oacute;pez Pe&ntilde;a and H. Touchette. A network theory analysis of "
    "football strategies. arXiv:1206.6904; <i>Proc. Euromech Physics of Sports "
    "Conference</i>, 2012.",
    "J. M. Buld&uacute; et al. Using network science to analyse football passing "
    "networks: dynamics, space, time, and the multilayer nature of the game. "
    "<i>Frontiers in Psychology</i>, 9:1900, 2018.",
    "J. M. Buld&uacute;, J. Busquets, I. Echegoyen, and F. Seirul&middot;lo. "
    "Defining a historic football team: using network science to analyze "
    "Guardiola's F.C. Barcelona. <i>Scientific Reports</i>, 9:13602, 2019.",
    "L. Gyarmati, H. Kwak, and P. Rodriguez. Searching for a unique style in "
    "soccer. arXiv:1409.0308; <i>KDD Workshop on Large-Scale Sports "
    "Analytics</i>, 2014.",
    "P. Cintia, S. Rinzivillo, and L. Pappalardo. A network-based approach to "
    "evaluate the performance of football teams. <i>MLSA Workshop, "
    "ECML-PKDD</i>, 2015.",
    "G. Ichinose, T. Tsuchiya, and S. Watanabe. Robustness of football passing "
    "networks against continuous node and link removals. <i>Chaos, Solitons &amp; "
    "Fractals</i>, 147:110973, 2021.",
    "J. Bekkers and S. Dabadghao. Flow motifs in soccer: what can passing "
    "behavior tell us? <i>Journal of Sports Analytics</i>, 5(4):299&ndash;311, 2019.",
    "Graph-oriented approaches of passing networks for predictive football match "
    "outcomes. <i>Journal of Big Data</i>, 2025.",
    "L. Pappalardo et al. A public data set of spatio-temporal match events in "
    "soccer competitions. <i>Scientific Data</i>, 6:236, 2019.",
    "StatsBomb. StatsBomb Open Data. https://github.com/statsbomb/open-data.",
    "R. Albert, H. Jeong, and A.-L. Barab&aacute;si. Error and attack tolerance "
    "of complex networks. <i>Nature</i>, 406:378&ndash;382, 2000.",
    "L. R. Ford and D. R. Fulkerson. Maximal flow through a network. "
    "<i>Canadian Journal of Mathematics</i>, 8:399&ndash;404, 1956.",
    "L. C. Freeman. A set of measures of centrality based on betweenness. "
    "<i>Sociometry</i>, 40(1):35&ndash;41, 1977.",
    "P. Bonacich. Factoring and weighting approaches to status scores and clique "
    "identification. <i>Journal of Mathematical Sociology</i>, 2(1):113&ndash;120, 1972.",
    "L. Page, S. Brin, R. Motwani, and T. Winograd. The PageRank citation "
    "ranking: bringing order to the web. Stanford InfoLab, 1999.",
    "A. Hagberg, D. Schult, and P. Swart. Exploring network structure, dynamics, "
    "and function using NetworkX. <i>Proc. SciPy</i>, 2008.",
    "F. Pedregosa et al. Scikit-learn: machine learning in Python. <i>JMLR</i>, "
    "12:2825&ndash;2830, 2011.",
]
for i, r in enumerate(refs, 1):
    P("[{}]&nbsp;&nbsp;{}".format(i, r), "ref")


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawCentredString(letter[0] / 2, 0.5 * inch,
                             "Tactical Topology — %d" % doc.page)
    canvas.restoreState()


def build():
    doc = SimpleDocTemplate(
        OUT, pagesize=letter,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
        topMargin=0.85 * inch, bottomMargin=0.75 * inch,
        title="Tactical Topology: Modelling Football Teams as Communication Networks",
        author="Nikos Raptis",
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print("[paper] wrote", OUT)


if __name__ == "__main__":
    build()
