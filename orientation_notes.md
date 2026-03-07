# Protein Design Week Orientation Notes (Memory File)

## Source Files
- `monday_presentation.pdf`: main orientation deck for the Protein Design Week program, goals, workflow options, and setup checklist.
- `PDW-Pymol-tutorial-2026.pdf`: practical tutorial for PyMOL usage in protein visualization and design validation.

## High-Level Program Framing (from monday deck)
- The week is a fun but structured learning sprint on AI-driven protein design.
- Core goal: use computational tools and wet-lab context to design/test novel proteins.
- Audience appears mixed (students with diverse backgrounds), with emphasis on hands-on collaboration.
- There is a weekend hackathon as the main practical culmination.

## Week Structure (orientation)
- Monday: introduction + workshop.
- Tuesday: office hours.
- Thursday: Adaptiv Bio lab visit.
- Friday evening event/discussion around AI-driven protein design and biosafety.
- Saturday/Sunday: hackathon.

## Communication + Resources
- Slack is the central communication hub (announcements, team channels, coordination).
- A centralized compute/resource hub is referenced (shown as "Rithub" in extraction; likely a project resource portal).
- GitHub resources/survival guide are referenced for participants.

## Setup / Checklist Before Hackathon
- Install PyMOL (or another protein visualization tool) via `https://pymol.org/`.
- Install Docker, `kubectl`, and Run:ai (explicitly marked for EPFL students only).
- Create a Hugging Face account: `https://huggingface.co`.
- Submit Hugging Face username for expanded rate limits.
- Join the HuggingScience organization.
- Fill required form to access/use AlphaFold3 workflows (marked mandatory in deck).
- Read associated survival guide on GitHub (optional).
- Attend office hours for questions (optional).

## Tooling Strategy in the monday deck
- Two broad execution modes are contrasted:
- Local/cluster workflow:
  - Pros: overnight jobs, direct post-processing, model/control flexibility.
  - Cons: setup overhead, docs/tutorial effort, extra infra requirements.
  - Mentions Docker/Kubernetes stack.
- Online/hosted workflow (e.g., platform/spaces):
  - Pros: easier startup, rapid prototyping.
  - Cons: rate limits, access constraints, account dependencies.
- Gradio/Spaces style interfaces are emphasized for quick iteration and simple pipeline wiring.

## Scientific Orientation Covered in monday deck
- Protein basics: proteins as amino-acid macromolecules encoded via DNA/gene information flow.
- Sequence -> structure -> function relationship is emphasized as central to design.
- Folding as an energy landscape problem; huge conformational search space.
- Why AI matters: prediction/generation helps navigate difficult structure/function space.
- Mentions modern model classes/workflows (transformers and model-assisted design flow).

## PyMOL Tutorial Summary (second PDF)
- PyMOL is presented as the core open-source viewer for protein structure exploration and publication-ready figures.
- Main operational goals:
  - Load and visualize protein structures.
  - Inspect sequence and structural regions.
  - Manipulate objects/chains/selections.
  - Validate mutation ideas and interface quality.

## PyMOL Interface Concepts
- GUI control pattern highlighted: `A` (Action), `S` (Show), `H` (Hide), `L` (Label), `C` (Color).
- Sequence panel usage is emphasized for selecting residues/chains.
- Chain separation and object-level selection management are part of the workflow.

## PyMOL Command-Line Concepts Mentioned
- Toggle sequence viewer: `set seq_view, 1` (toggle off with `0`).
- FASTA-like sequence print helper: `print cmd.get_fastastr(all)` (replace `all` with object/selection as needed).
- Residue iteration/listing over a selection is shown via `iterate your_selection, ...` style command.
- Tutorial repeatedly combines GUI actions and command-line methods for speed.

## PyMOL Practical Tasks in Tutorial
- Visualize important regions:
  - Binding sites.
  - Active-site cavities.
- Inspect protein-protein interfaces:
  - Buried surface/contact inspection.
  - Surface overlap and gap/pocket cues.
- Mutation validation:
  - Visualize proposed mutations.
  - Check local contacts around mutation.
  - Identify steric clashes vs favorable packing.
  - Look for potential stabilizing interactions (e.g., H-bond context in tutorial narrative).
- Surface analysis for complexes (examples include antibody-antigen and enzyme-substrate style interfaces).
- Detect geometric issues/clashes and adjust/inspect accordingly.

## Interpretation of Color/Contact Heuristics (from tutorial)
- Bad sign: residues colliding/bumping -> potential steric clash.
- Better sign: appropriate spacing and compatible local contacts.
- Interface inspection focuses on whether contact zones look plausible and buried area is coherent.

## What To Remember For Future Sessions
- If short on time, first verify the setup checklist (PyMOL + accounts + platform access + required forms).
- For hackathon productivity, choose one workflow early:
  - Local/cluster if you need maximal control and can tolerate setup.
  - Hosted/Spaces if you need speed and low friction.
- Use PyMOL not just for pretty images, but as a design QA tool:
  - Mutation neighborhood checks.
  - Clash detection.
  - Interface sanity checks.
- Keep GitHub survival guide and Slack/office hours as first-line unblockers.

## Confidence Notes
- Core setup checklist items are extracted clearly and consistently.
- Some non-checklist slide text had extraction artifacts; thematic summaries above follow the clear repeated content and context from both decks.
