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
    conversations_structured = self.df["structured_conversations"]
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
          "Embedding" : self.text_to_embedding(conversations_structured[i],entities[i],relationship_fixed)
      }
      self.json_structured.append(str(json_data))

    pd.DataFrame(self.json_structured,columns=["jsonSummary"]).to_excel(save_path, index=False)
    
    
  def process_conversation(self,structured_conversation):
    if isinstance(structured_conversation, str):
      try:
         conversation = json.loads(structured_conversation)
      except json.JSONDecodeError:
         conversation = structured_conversation.replace("'", '"') 
         conversation = json.loads(conversation)
    
    conversation_text = "Conversation: "
    for item in conversation:
        conversation_text += f"{item['role']}: {item['message']} "
    return conversation_text.strip()
      
    



  def structured_to_text(self,conversation,entity,relationship):
    if (isinstance(entity,str)) or (isinstance(relationship,str)):
      entity = json.loads(entity)
      relationship = json.loads(relationship)
    entity_texts = [] 
    for key,val in entity.items():
      entity_texts.append(f'{key}: {", ".join(val)}')
    entity_text = ". ".join(entity_texts)
    relationships_text = ""
    for item in relationship:
      for key,val in item.items():
          relationships_text += f"{val} "
      relationships_text = relationships_text.strip()
      relationships_text += ". "
    
    
    conversation_text = self.process_conversation(conversation)
    combined_text = f"{entity_text}; {relationships_text}; {conversation_text}"
    
    return combined_text
  


  def text_to_embedding(self,conversation,entity,relationship):
    ###
    ### Relationships should be fixed before passing to this function
    ### For database embedding, it is called from convertExcel function
    ### fix_relationships function should be called before this function for individual embedding
    ###

    text_intent = self.structured_to_text(conversation,entity,relationship)
    text_conversation = self.process_conversation(conversation)
    embedding_intent = self.model.encode(text_intent, padding=True, truncation=True)
    embedding_text = self.model.encode(text_conversation, padding=True, truncation=True)
    
    return embedding_intent + embedding_text






    
    
