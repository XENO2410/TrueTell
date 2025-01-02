import pandas as pd
import re

def clean_text(text):
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s\']', ' ', text)
    text = ' '.join(text.split())
    return text

# Create lists to store claims and labels
claims = []
labels = []

# Read the text file and parse claims
with open('claims.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    
    for i in range(0, len(lines), 2):
        if i+1 < len(lines):
            claim_line = lines[i].strip()
            fact_check_line = lines[i+1].strip()
            
            if claim_line.startswith('Claim:'):
                claim = claim_line[6:].strip()
                fact_check = fact_check_line[11:].strip() if fact_check_line.startswith('Fact Check:') else ''
                
                if claim and fact_check:
                    # Clean the claim text
                    clean_claim = clean_text(claim)
                    
                    # Determine the label (0 for True, 1 for False)
                    label = 0 if fact_check.lower().startswith('true') else 1
                    
                    claims.append(clean_claim)
                    labels.append(label)

# Create DataFrame
df = pd.DataFrame({
    'text': claims,
    'label': labels
})

# Remove duplicates
df = df.drop_duplicates(subset=['text'])

# Save to CSV
df.to_csv('datasets/factdata.csv', index=False)

# Print statistics
print(f"Total samples: {len(df)}")
print(f"True claims: {len(df[df['label'] == 0])}")
print(f"False claims: {len(df[df['label'] == 1])}")
print("\nSample of the dataset:")
print(df.head())