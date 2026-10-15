# NLP-Project-
# Searching and aligning UK government inquiry reports

NLP coursework (COP509, Loughborough University, MSc Artificial Intelligence). Two notebooks over 12 UK government documents: six inquiry or committee reports, each paired with the government's response.

1. **Search.** Find relevant passages across all 12 documents using four retrieval methods.
2. **Extraction and alignment.** Pull out each recommendation, match it to the response that addresses it, and label the response as accepted, partially accepted or rejected.

The short version: retrieval was easy on this corpus. Classifying responses was not. Nearly every error sits on the line between "accepted" and "partially accepted", because government responses hedge even when they accept.

## Data

Six pairs of PDFs: Space Economy, UK Covid-19 Inquiry Module 1, Behaviour Change, Post Office Horizon IT, Infected Blood Inquiry and WASPI. Each pair comes with a ground truth spreadsheet of hand-verified recommendation text, response text and a status label. That gives 163 recommendations in total.

The data is not in this repo. The notebooks read it from a Google Drive folder, so you would need to edit `BASE_PATH` in the first cells.

## Notebook 1: search

`COP509_Notebook1_Search.ipynb`

- Text extracted with pdfplumber, then split into 200-word chunks with 50 words of overlap (3,949 chunks).
- Four methods: TF-IDF with bigrams (74,539 terms), LSA (100 dimensions, 17.4% of variance), sentence-transformer embeddings (`all-MiniLM-L6-v2`), and a hybrid that averages normalised TF-IDF and embedding scores.
- The Post Office response PDF is a printed web page with no text layer. I ran OCR on it with Tesseract, which produced 27 noisy chunks.
- A widget lets you search by method, dataset and document type.

### Results

On 11 queries written from vocabulary in the documents, all four methods scored 1.0 on Precision@5, Recall@5, MRR and MAP. That says the test was easy. It does not say the methods are equal.

Three harder queries separated them. TF-IDF found the right dataset on all three. Semantic search missed "money investment economic return" and returned Post Office results instead of Space Economy. The hybrid did not reliably beat either component.

Evaluation only checks whether the right dataset comes back. It does not check whether the returned passage is the best one.

## Notebook 2: extraction and alignment

`COP509_Notebook2_Extraction_Alignment.ipynb`

**Extraction.** Each document needed its own approach: header patterns for Space Economy and Covid-19, numbered paragraphs for Behaviour Change, a numbered list for Post Office, and modal verbs ("should", "would have recommended") for WASPI. All 163 recommendations were found, but not all from the PDFs:

- Infected Blood (58) is loaded from the ground truth spreadsheet. Its nested IDs such as `4a(iii)` broke my parser, which produced 148 items instead of 58.
- WASPI (3) is found by modal-verb matching, then paired with the ground truth text by word overlap.
- Two Behaviour Change items (8.32a and 8.32b/8.33) were added from the spreadsheet.

Text quality, measured by ROUGE-1 F1 against the ground truth: 0.998 Infected Blood, 0.937 Post Office, 0.930 Behaviour Change, 0.753 Covid-19, 0.605 WASPI, 0.464 Space Economy. Space Economy is weakest because the regex window does not match the span the annotators chose.

**Alignment.** Space Economy and Covid-19 responses are matched by recommendation number. Behaviour Change uses sentence-embedding similarity. Infected Blood, Post Office and WASPI response text comes from the spreadsheet, because the PDFs were unusable or too tangled to split reliably.

**Classification.** A keyword classifier with rejection, strong acceptance, soft acceptance and qualifier lists.

| Dataset | Correct | Accuracy |
|---|---|---|
| Space Economy | 18 / 40 | 45.0% |
| UK Covid-19 Module 1 | 4 / 10 | 40.0% |
| Behaviour Change | 21 / 33 | 63.6% |
| Post Office Horizon IT | 13 / 19 | 68.4% |
| Infected Blood | 35 / 58 | 60.3% |
| WASPI | 2 / 3 | 66.7% |
| **All** | **93 / 163** | **57.1%** |

Weighted F1 is 0.56. Of 86 recommendations the government accepted, the classifier labelled 47 as partial. Of 73 partials, it labelled 18 as accepted. Only 4 recommendations in the whole set were rejected, so the rejection numbers mean very little. The classifier never outputs "not addressed".

A zero-shot BART model (`facebook/bart-large-mnli`) scored 70% against 60% for the keyword classifier. That is 10 rows, so treat it as a hint and nothing more.

## Caveats

- The keyword lists were tuned against the same 163 labels used to score them. The 57.1% is probably optimistic for unseen documents.
- Two classifier versions scored 58.3% and 57.1%. Version 2 improved Behaviour Change and Post Office but made Infected Blood worse.
- The WASPI-specific handler did not fix WASPI recommendation 2. The response attributes the recommendation to the Ombudsman rather than stating the government's position, and neither classifier handles that.
- The chunk-level matching demo for Behaviour Change gave almost the same scores as the main pipeline, because that pipeline already splits on paragraphs. It shows the approach works but does not show an improvement.
- The extraction patterns suit this genre of document. A different kind of report would need new ones.

## What I would do next

Fine-tune a small transformer on the 163 labelled examples, with a held-out split so the accuracy figure means something. Replace the NumPy cosine search with a vector database that supports metadata filtering. Write relevance judgements at chunk level so retrieval is scored on passages, not datasets.

## Stack

Python, pdfplumber, pytesseract, scikit-learn, sentence-transformers, BERTopic, Hugging Face Transformers, ROUGE, pandas, ipywidgets. Developed in Google Colab.

## Files

- `COP509_Notebook1_Search.ipynb` and `.pdf`
- `COP509_Notebook2_Extraction_Alignment.ipynb` and `.pdf`
