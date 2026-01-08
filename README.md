# Stand-up Comedy Topic Modeling Across Cultures
## Natural Language Processing (E1282) Research Note, Dr. Sascha Göbel
- Author: Lonny Chen (216697)
- Submission Date: 8 January 2026
- **Research note PDF: [Chen_NLP_Research_Note.pdf](https://github.com/lonnychen/nlp_research_note/blob/main/Chen_NLP_Research_Note.pdf).**

## Summary
This research aims to use the Latent Dirichlet Allocation (LDiA) method of topic modeling to explore and compare differences in topics used for stand-up comedy across cultures. The dataset is collected from transcripts fetched from YouTube Shorts from Comedy Central’s global channels. The study is currently limited to English-language channels catering to US and UK audiences. The methodology faithfully follows a structured Natural Language Processing (NLP) pipeline of tasks, including iterative token selection and hyperparameter tuning. Five subjectively interpretable topics were found out of the total of 12 topics, and their distribution as the dominant topic of the Shorts is largely similar between the US and UK channels.

## Repository Structure

The repository structure below supports modular functional programming.
```repo_structure
.
├── data/                      # Data collection functions
├── ldia_model/                # LDiA modeling functions
├── plots/                     # Plots included in research note
├── results/                   # Modeling outputs
├── Chen_NLP_Research_Note.pdf # Research Note PDF
├── README.md
├── nlp_research_note.ipynb    # Run notebook
```
