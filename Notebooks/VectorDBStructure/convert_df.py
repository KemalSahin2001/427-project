from db_structure import DatabaseStructure
import pandas as pd

## DATAFRAME CONVERSION

df = pd.read_excel(READ_PATH)
textembedding = DatabaseStructure(df)
textembedding.convertExcel(SAVE_PATH)

## 
# Individual conversion
"""
entity = "Entity1"
relationship = "Rel1"
resolution = "Res1"
relationship_fixed = textembedding.fix_relationships(relationship, resolution)
textembedding.text_to_embedding(entity,relationship_fixed)
"""
## 