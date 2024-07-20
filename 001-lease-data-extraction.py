# Named Entity Recognition (NER) for Lease Data Extraction
import spacy
import pandas as pd
import matplotlib.pyplot as plt

# Load pre-trained NER model
nlp = spacy.load("en_core_web_sm")

# Sample lease text
lease_text = """
This Lease Agreement is made on January 1, 2023, between ABC Corp (Landlord) and XYZ Inc (Tenant).
The Lease Term is for five years starting from February 1, 2023, to January 31, 2028.
The monthly rent is $2,500, payable on the first day of each month.
"""

# Process the text with the NER model
doc = nlp(lease_text)

# Extract entities
entities = [(ent.text, ent.label_) for ent in doc.ents]
print("Extracted Entities:", entities)

# Visualize entities
labels = [ent.label_ for ent in doc.ents]
values = [lease_text.count(ent.text) for ent in doc.ents]

plt.bar(labels, values, color='blue')
plt.xlabel('Entity Labels')
plt.ylabel('Frequency')
plt.title('NER Entity Extraction')
plt.show()
