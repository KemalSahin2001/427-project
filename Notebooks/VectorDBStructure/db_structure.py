from sentence_transformers import SentenceTransformer
import pandas as pd
import re
import json
class DatabaseStructure:
  def __init__(self,dataframe = None):
    self.model = SentenceTransformer('all-MiniLM-L6-v2')
    self.df = dataframe
    self.json_structured = []

  def fix_relationships(self,relationship,resolution):
    relationship_re = r'(\{(.*?)\})' 
    matches_re = re.findall(relationship_re, relationship,flags=re.DOTALL)
    matches_remain = "["
    for match in matches_re:
        relationship = json.loads(match[0])
        
        if relationship.get("predicate") == "resolvesWith":
            if relationship.get("object") != "" and relationship.get("object") != "":
                matches_remain += json.dumps(relationship) + ","
            else:
               matches_res = re.findall(relationship_re, resolution,flags=re.DOTALL)
               for match in matches_res:
                    relationship = json.loads(match[0])
                    if relationship.get("predicate") == "resolvesWith":
                        if relationship.get("object") != "" and relationship.get("object") != "":
                            matches_remain += json.dumps(relationship) + ","
                        else:
                            continue
                    else:
                        continue


        if (relationship.get("subject") != "") and (relationship.get("object") != ""):
            matches_remain += json.dumps(relationship) + ","

        else:
          continue
    if matches_remain.endswith(","):
        matches_remain = matches_remain[:-1]
    matches_remain += "]"
    return matches_remain

  def convertExcel(self,save_path):
    company_names = self.df["company_name"]
    conversations = self.df["cleaned_conversations"]
    entities = self.df["entities"]
    relationships = self.df["relationship"]
    resolution = self.df["resolution"]

    for i in range(len(self.df)):
      rel,res = relationships[i],resolution[i]
      relationship_fixed = self.fix_relationships(rel,res)
      json_data = {
          "ChatID": str(i+1),
          "Company_name": company_names[i],
          "Conversation_History": {"conversation": conversations[i]},
          "Entities": entities[i],
          "Relationships": relationship_fixed,
          "Embedding" : self.text_to_embedding(entities[i],relationship_fixed)
      }
      self.json_structured.append(str(json_data))

    pd.DataFrame(self.json_structured,columns=["jsonSummary"]).to_excel(save_path, index=False)
    
    

  def structured_to_text(self, entity,relationship):
    

    if (isinstance(entity,str)) or (isinstance(relationship,str)):
      entity = json.loads(entity)
      relationship = json.loads(relationship)
    entity_texts = []

    for key, val in entity.items():
        entity_texts.append(f'{key}: {", ".join(val)}')

    entity_text = ". ".join(entity_texts)

    relationships_text = ""
    for item in relationship:
      for key,val in item.items():
        relationships_text += f"{val} "
      relationships_text = relationships_text.strip()
      relationships_text += ". "

    combined_text = f"{entity_text}. {relationships_text}"

    return combined_text

  def text_to_embedding(self, entity,relationship):
    ###
    ### Relationships should be fixed before passing to this function
    ### For database embedding, it is called from convertExcel function
    ### fix_relationships function should be called before this function for individual embedding
    ###

    text = self.structured_to_text(entity,relationship)
    embedding = self.model.encode(text)
    return embedding


   