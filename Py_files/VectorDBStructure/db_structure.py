from sentence_transformers import SentenceTransformer
import pandas as pd
import re
import json
class DatabaseStructure:
  def __init__(self,dataframe = None):
    self.model = SentenceTransformer('all-MiniLM-L6-v2')
    self.df = dataframe
    self.json_structured = []

  def fix_relationships(self,relationship):
    
    relationship_re = r'\[\{(.*?)\}\]' 
    relationship_fixed =  re.findall(relationship_re, relationship,flags=re.DOTALL)
   
    return relationship_fixed
    

  def convertExcel(self,save_path):
    company_names = self.df["company_name"]
    conversations = self.df["cleaned_conversations"]
    conversations_structured = self.df["structured_conversations"]
    entities = self.df["entities"]
    relationships = self.df["relationship"]
    

    for i in range(len(self.df)):
      rel = relationships[i]
      relationship_fixed = self.fix_relationships(rel)
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
    
    
      try:
         conversation = json.loads(structured_conversation)
         conversation = conversation[1]["conversation"]
         conversation_text = "Conversation: "
         for item in conversation:
            conversation_text += f"{item['message']} "
         return conversation_text.strip()
      except json.JSONDecodeError:
         try:
          conversation = structured_conversation.replace("'", '"') 
          conversation = json.loads(conversation)
          conversation = conversation[1]["conversation"]
          conversation_text = "Conversation: "
          for item in conversation:
              conversation_text += f"{item['message']} "
          return conversation_text.strip()
         except:
           conversation_text = "Conversation: " + structured_conversation
           return conversation_text.strip()
          
    
      
    



  def structured_to_text(self,conversation,entity,relationship):
    
    if (isinstance(entity,str)):
      entity = json.loads(entity)
    if (isinstance(relationship,str)):
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
    embedding_intent = self.model.encode(text_intent)
    
    embedding_text = self.model.encode(text_conversation)
    
    
    return embedding_intent + embedding_text






    
    