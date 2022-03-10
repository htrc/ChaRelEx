import pandas as pd
import spacy
import en_core_web_lg

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import tqdm
import json

#predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz",cuda_device=0)

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")


nlp = en_core_web_lg.load()
nlp.max_length = 5000000

with open("Gutenberg_Volumes_200.json","r") as file:    
    for l in file:
        etexts = json.loads(l)

vols = []
fname = "gutenberg_parse_ie_.json"

with open(fname,"w"):
    pass

for i,x in enumerate(etexts):
    temp_s = etexts[x]
    temp_s = temp_s.encode("ascii", "ignore")
    temp_s = temp_s.decode()
    j=0
    sent = temp_s.split("\n")
    temp_s = []
    for i,sx in enumerate(tqdm.tqdm(sent)):
        if len(sx.strip())>0:
            temp_s.append(sx.strip())
        if sx.strip()=="" and len(temp_s)>0:
            tt = nlp(" ".join(temp_s))

            for y in tt.sents:
                tt = nlp(str(y))                    

                dict_s = {"sent_id": "{}-{}".format(x,j)}
                dict_s["sent"] = str(y)
                dict_s["token"] = [str(x) for x in tt]

                try:   
                    xx = predictor.predict(str(y))
                except:
                    xx = None
                dict_s["openie"] = xx

                ents = []
                for ent in tt.ents:
                    ents.append((ent.label_, ent.text, ent.start, ent.end))
                dict_s["entities"] = ents
                with open(fname,"a") as filewrite:
                    filewrite.write("{}\n".format(json.dumps(dict_s)))   
                j+=1
            temp_s = []