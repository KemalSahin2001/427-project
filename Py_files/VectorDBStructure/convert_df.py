from db_structure import DatabaseStructure
import pandas as pd

## DATAFRAME CONVERSION

df = pd.read_excel("VirginAmerica.xlsx")
textembedding = DatabaseStructure(df)
textembedding.convertExcel("VirginAmerica_Embedding.xlsx")

## 
# Individual conversion
"""
conversation = "conv"
entity = "Entity1"
relationship = "Rel1"

relationship_fixed = textembedding.fix_relationships(relationship)
textembedding.text_to_embedding(conversation,entity,relationship_fixed)
"""
## 