import spacy

nlp=spacy.load("output/models/model-best")

address_list=["16, 15 SILVER SPRINGS WY NW, AIRDRIE",
              "46, 1008 WOODSIDE WY NW, AIRDRIE",
              "9, 209 WOODSIDE DR NW, AIRDRIE",
              "42, 4 STONEGATE DR NW, AIRDRIE",
              "7, 209 WOODSIDE DR NW, AIRDRIE"]

# Checking predictions for the NER model
for address in address_list:
    doc=nlp(address)
    ent_list=[(ent.text, ent.label_) for ent in doc.ents]
    print("Address string -> "+address)
    print("Parsed address -> "+str(ent_list))
    print("******")


address="306, 190 KANANASKIS WAY, CANMORE"
doc=nlp(address)
ent_list=[(ent.text, ent.label_) for ent in doc.ents]
print("Address string -> "+address)
print("Parsed address -> "+str(ent_list))
