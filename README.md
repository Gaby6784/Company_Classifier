# Company_Classifier
Veridion Engineering Challenge

## Project Description
Company Classifier is a semantic classification tool that automatically assigns insurance-relevant labels to companies based on their descriptions and business tags.
It leverages sentence embeddings and cosine similarity to match each company against a static taxonomy of industry-specific categories.

## Project aims and objectives
This project addresses the challenge of classifying companies into a predefined insurance taxonomy in the absence of labeled training data.
The classifier operates by measuring the semantic similarity between company descriptions and a static list of insurance-related labels.

It is designed to be:  
-	Simple: takes in CSV input, produces labeled output.  
-   Semantic: uses sentence-transformers (all-MiniLM-L6-v2) to encode textual meaning.  
-   Threshold-based: assigns labels only if similarity exceeds a configurable cutoff (default: 0.32).  
-   Offline: requires no API calls or external services.  
  
The result is a fast, lightweight and reproducible classification system tailored for insurance use cases.

## First steps
When I first started the challenge I jumped right into the two files provided, the list of companies and the insurance taxonomy. When I opened the list of companies and saw the description part I instantly thought of matching the words from the description to the labels in insurance taxonomy.  

![companies](https://github.com/Gaby6784/Company_Classifier/blob/main/doc_screenshots/Screenshot%202025-05-10%20at%2014.11.53.png)    

Then, I opened the taxonomy and I was expecting to see a lot more labels because of the large number of companies.  

<img src="https://github.com/Gaby6784/Company_Classifier/blob/main/doc_screenshots/Screenshot%202025-05-10%20at%2014.12.36.png" alt="Alt Text" width="300" height="570">

To start off, we need to install the required libraries:  
```bash
pip install pandas torch sentence-transformers
```
## Step 1: load the files provided into the project using pandas
```bash
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

companies_df = pd.read_csv("ml_insurance_challenge.csv")
taxonomy_df = pd.read_csv("insurance_taxonomy - insurance_taxonomy.csv")
```
## Step 2: generate a combined text by concatenating its description and tags
```bash
company_texts = (
    companies_df['description'].fillna('') + ' ' +
    companies_df['business_tags'].fillna('').astype(str)
)

taxonomy_labels = taxonomy_df['label'].dropna().unique().tolist()
```
## Step 3: transforming both company texts and taxonomy labels into numerical vectors, capturing their semantic meaning
```bash
model = SentenceTransformer("all-MiniLM-L6-v2")

company_embeddings = model.encode(company_texts.tolist(), convert_to_tensor=True, show_progress_bar=True)
taxonomy_embeddings = model.encode(taxonomy_labels, convert_to_tensor=True, show_progress_bar=True)
```
## Step 4: comparing every company vector with all taxonomy vectors using cosine similarity, keeping only labels with a score > 0.32
```bash
cosine_scores = util.cos_sim(company_embeddings, taxonomy_embeddings)

threshold = 0.32
filtered_labels = []
for i in range(len(companies_df)):
    scores = cosine_scores[i]
    valid_indices = (scores >= threshold).nonzero(as_tuple=True)[0]
    labels = [taxonomy_labels[idx] for idx in valid_indices]
    filtered_labels.append(labels)
```

## Step 5: saving the results and printing 
```bash
companies_df['insurance_label'] = filtered_labels
companies_df.to_csv("ml_insurance_challenge.csv", index=False)

print("Saved updated input file with 'insurance_label' column: ml_insurance_challenge.csv")
```

## After running the code
### Most companies receive at least one label  
Using the default similarity threshold of 0.32, the vast majority of companies are successfully matched with one or more labels from the taxonomy.
This indicates that the classifier performs well across a wide range of business types and descriptions.  

### What if a company has no label?
Some companies may remain unlabeled if their description does not meet the minimum similarity required for any label. **This is intentional** - it helps avoid assigning misleading or noisy classifications.


### Example output - labeled
> SK Fish Market is a company categorized under fish processing. It is located in Assam, India.","['Fish Processing Services', 'Packaging Services for Seafood']",Retail,"Meat, Fish & Seafood Stores",Fish and Seafood Retailers,"['Livestock Dealer Services', 'Fishing and Hunting Services', 'Meat Processing Services', 'Seafood Processing Services', 'Ice Production Services', 'Food Processing Services', 'Market Research Services']

### Example output - unlabeled
>"Kabowa Villas is a villa located in Kampala, Uganda.","['Accommodation Services', 'Villa Rental Services']",Services,Resorts,Hotels (except Casino Hotels) and Motels,[]

The examples are copied from the input file, where was created a new column 'insurance_label'.

## Evaluation Strategy

Since ground-truth labels were not available, quality was assessed by:
- Manual inspection of a random sample of predictions
- Tracking % of companies receiving no label
- Comparing noise level across thresholds and models (MiniLM vs E5)

## Why do I use sentence-transformers

I chose to use sentence-transformers because this project requires an understanding of meaning. The goal is to classify companies into insurance-related categories based on often vague, unstructured descriptions. Sentence-transformers provide a powerful alternative by encoding each company description and each taxonomy label into a semantic vector.  


By comparing these vectors using cosine similarity, the classifier can identify conceptual matches. This enables the system to assign accurate labels even when wording varies significantly, without requiring any labeled training data.  

<img src="https://github.com/Gaby6784/Company_Classifier/blob/main/doc_screenshots/Screenshot%202025-05-10%20at%2017.03.47.png" alt="Alt Text" width="250" height="470">

## Conclusion
Although the classifier does a good job overall, there are instances where companies are given labels that are either unrelated or of poor quality. Raising the similarity threshold alone won't always eliminate these misclassifications; in fact, doing so too much may result in the omission of pertinent labels.
