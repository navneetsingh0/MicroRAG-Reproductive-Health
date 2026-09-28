# eval_queries.py
# Evaluation query set for MicroRAG. Skim every query against your corpus before trusting the results:
# replace any question you think the 350 abstracts genuinely cannot answer.
#
# expect = "answer"  -> system should answer (measures false-abstention, faithfulness, relevance, precision)
# expect = "abstain" -> system should decline  (measures correct-abstention, and hallucination if the gate is off)

_PROFILE_TMPL = (
    "{state}. Total Lactobacillus relative abundance: {lacto}%. "
    "Dominant non-Lactobacillus taxa: {taxa}. "
    "Shannon diversity index: {shannon}. "
    "What disease risk is associated with this vaginal microbiome composition?"
)  # identical to the query built by interpret_profile() in app.py

_RAW = [
    # ---- symptoms / clinical presentation ----
    ("symptoms", "answer", "What are the common clinical symptoms of bacterial vaginosis?"),
    ("symptoms", "answer", "Is bacterial vaginosis often asymptomatic?"),
    ("symptoms", "answer", "How does vaginal pH change in bacterial vaginosis?"),
    ("symptoms", "answer", "What is the association between bacterial vaginosis and vaginal discharge?"),
    ("symptoms", "answer", "How often does bacterial vaginosis recur after treatment?"),
    ("symptoms", "answer", "What complications are associated with bacterial vaginosis during pregnancy?"),
    # ---- risk factors ----
    ("risk_factors", "answer", "What are the risk factors for developing bacterial vaginosis?"),
    ("risk_factors", "answer", "Does douching increase the risk of bacterial vaginosis?"),
    ("risk_factors", "answer", "How is sexual activity associated with bacterial vaginosis?"),
    ("risk_factors", "answer", "Is there a link between bacterial vaginosis and HIV acquisition?"),
    ("risk_factors", "answer", "How does ethnicity relate to vaginal microbiome composition?"),
    ("risk_factors", "answer", "Does smoking affect the vaginal microbiome?"),
    # ---- diagnosis ----
    ("diagnosis", "answer", "What is the Nugent score and how is it used to diagnose bacterial vaginosis?"),
    ("diagnosis", "answer", "How do Amsel criteria compare with Nugent scoring?"),
    ("diagnosis", "answer", "Can molecular methods such as PCR diagnose bacterial vaginosis?"),
    ("diagnosis", "answer", "What are the community state types of the vaginal microbiome?"),
    ("diagnosis", "answer", "What is intermediate vaginal flora in the Nugent scoring system?"),
    ("diagnosis", "answer", "Can 16S rRNA sequencing be used to characterize the vaginal microbiome?"),
    # ---- treatment ----
    ("treatment", "answer", "How is bacterial vaginosis treated with metronidazole?"),
    ("treatment", "answer", "What is the role of probiotics in treating bacterial vaginosis?"),
    ("treatment", "answer", "Why does bacterial vaginosis recur after antibiotic treatment?"),
    ("treatment", "answer", "Does treating the male partner reduce recurrence of bacterial vaginosis?"),
    ("treatment", "answer", "Can Lactobacillus crispatus supplementation restore the vaginal microbiome?"),
    ("treatment", "answer", "Is clindamycin an effective treatment for bacterial vaginosis?"),
    # ---- mechanism ----
    ("mechanism", "answer", "How does Gardnerella vaginalis form biofilms?"),
    ("mechanism", "answer", "Why does Lactobacillus dominance protect the vaginal environment?"),
    ("mechanism", "answer", "How does lactic acid produced by Lactobacillus inhibit pathogens?"),
    ("mechanism", "answer", "What role does Atopobium vaginae play in bacterial vaginosis?"),
    ("mechanism", "answer", "How do bacterial vaginosis-associated bacteria affect the vaginal immune response?"),
    ("mechanism", "answer", "What is the role of Prevotella in vaginal dysbiosis?"),
    # ---- mixed / uncertain evidence (guards against the over-triggered-fallback bug) ----
    ("uncertain_evidence", "answer", "Do probiotics prevent preterm birth in women with bacterial vaginosis?"),
    ("uncertain_evidence", "answer", "Is bacterial vaginosis causally linked to infertility?"),
    # ---- profile-style queries (exact template used by the Upload tab) ----
    ("profile", "answer", _PROFILE_TMPL.format(
        state="Lactobacillus-dominant (community state consistent with vaginal eubiosis)", lacto=92,
        taxa="Gardnerella vaginalis (3%), Prevotella (2%), Atopobium vaginae (1%)", shannon=0.45)),
    ("profile", "answer", _PROFILE_TMPL.format(
        state="Lactobacillus-depleted (community state consistent with dysbiosis)", lacto=8,
        taxa="Gardnerella vaginalis (45%), Prevotella (20%), Atopobium vaginae (12%)", shannon=1.9)),
    ("profile", "answer", _PROFILE_TMPL.format(
        state="Intermediate Lactobacillus abundance (borderline/transitional community state)", lacto=40,
        taxa="Gardnerella vaginalis (30%), Prevotella (12%), Sneathia (6%)", shannon=1.6)),
    ("profile", "answer", _PROFILE_TMPL.format(
        state="Lactobacillus-dominant but with notable residual BV-associated taxa present (atypical/incompletely resolved community state)",
        lacto=65, taxa="Gardnerella vaginalis (25%), Atopobium vaginae (5%), Prevotella (3%)", shannon=1.05)),
    # ---- clearly off-topic: should be blocked by the distance gate ----
    ("off_topic", "abstain", "What is the recommended first-line treatment for type 2 diabetes?"),
    ("off_topic", "abstain", "How does the transformer architecture work in deep learning?"),
    ("off_topic", "abstain", "How do I bake sourdough bread at home?"),
    ("off_topic", "abstain", "What is the mechanism of action of statins?"),
    # ---- near-miss: gynecological but outside the BV corpus (hardest test for the gate) ----
    ("near_miss", "abstain", "What are the symptoms of endometriosis?"),
    ("near_miss", "abstain", "What causes polycystic ovary syndrome?"),
]

TEST_QUERIES = [
    {"id": i + 1, "category": cat, "expect": exp, "query": q}
    for i, (cat, exp, q) in enumerate(_RAW)
]

if __name__ == "__main__":
    from collections import Counter
    print(len(TEST_QUERIES), "queries")
    for cat, n in Counter(t["category"] for t in TEST_QUERIES).items():
        print(f"  {cat}: {n}")
