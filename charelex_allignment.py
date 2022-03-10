import json
import pandas as pd
import networkx as nx
import numpy as np
import itertools
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tqdm

# read entity function
def read_ent(toks):
    prev_ent = None
    start_ent = 0   
    ent_list = {}
    for i,x in enumerate(toks):
        ll = "-".join(x.split("-")[1:])
        if ll!=prev_ent:
            if prev_ent!=None and prev_ent!="":

                ent_list[prev_ent] =(start_ent,i)
            start_ent=i
            prev_ent=ll
    if ll!="0":
        ent_list[prev_ent] =(start_ent,i+1)
    return ent_list


parsed_file = "gutenberg_parse_ie_etexts.json"
outfile = "gutenberg_network_data.csv"

sent_graph = []
sent_vol = set()
with open(parsed_file,"r") as file:
    for l in file:
        xx = json.loads(l)

        sent_graph.append(xx)

# prepare sentence graph for each volume
print("prepare sentence graph for each volume")
pd_graph = pd.DataFrame(sent_graph)
pd_graph["vol_id"] = pd_graph.sent_id.apply(lambda xx:"-".join(xx.split("-")[:-1]))

# for sample take numbers
n_samples = 20
vol_list = list(pd_graph.vol_id.unique())[:n_samples]

pd_graph = pd_graph[pd_graph.vol_id.apply(lambda x:x in vol_list)]

sent_graph_pd = pd_graph[pd_graph.openie.apply(lambda x:len(x["verbs"])>0)]

vol_ids = sent_graph_pd.vol_id.unique()


# define global volume graph
print("define global volume graph for {} volumes".format(n_samples))
vol_graph = {}

for vol in tqdm.tqdm(vol_ids):
    test_pd = sent_graph_pd.groupby("vol_id").get_group(vol)
    import numpy as np
    graph_list = []
    for x in test_pd[["sent_id","openie","entities","token"]].to_records():
        temp_graph = nx.DiGraph()
        for y in x.openie["verbs"]:
            ent_ext = x.entities
            tokens = x.openie["words"]
            net_list = []
            for xx in ent_ext:    
                mask_ent = np.zeros(len(y["tags"]),dtype=int)
                mask_ent[xx[2]:xx[3]] = 1
                S = None
                V = None
                O = None
                all_ent = read_ent(y["tags"])
                try:
                    V_pos = all_ent["V"]  
                except:
                    continue
                V = tokens[V_pos[0]:V_pos[1]]
                for key,yy in all_ent.items():                
                    if key == "V":
                        continue

                    mask = np.zeros(len(y["tags"]),dtype=int)
                    mask[yy[0]:yy[1]] = 1
                    intersect = np.sum(mask&mask_ent)
                    if intersect>0:
                        if yy[0] < V_pos[0]:
                            S = tokens[xx[2]:xx[3]]
                        else:
                            O = tokens[xx[2]:xx[3]]

                        temp_graph.add_node(" ".join(V),stype="v")                    
                        if (xx[0] == "PERSON") | (xx[0] == "ORG"):
                            if S is not None:
                                temp_graph.add_edge(" ".join(S)," ".join(V),ie=(xx,y))
                            if O is not None:                        
                                temp_graph.add_edge(" ".join(V)," ".join(O),ie=(xx,y))


        required_max_path_length = 3 # (inferior or equal to)

        G = temp_graph

        all_paths = []
        nodes_combs = itertools.permutations(G.nodes, 2)

        for source, target in nodes_combs:
            paths = nx.all_simple_paths(G, source=source, target=target, cutoff=required_max_path_length)

            for path in paths:
                if path not in all_paths and path[::-1] not in all_paths and len(path)==required_max_path_length:
                    if path[0] not in nx.get_node_attributes(G,"stype").keys():
                        all_paths.append(path)


        if len(all_paths)>0:                    
            graph_list.append((x.sent_id,temp_graph,all_paths,x.token))

    vol_graph[vol]=graph_list


# compute sentiment anlaysis 
print("preparing network and sentiment analysis")
cons_graph_vol = {}

for xvol,graph_list in tqdm.tqdm(vol_graph.items()):
    cons_graph = nx.DiGraph()
    for x in graph_list:
        is_break = False
        for iy,y in enumerate(x[2]):        
            if cons_graph.get_edge_data(y[0], y[2]) == None:
                cons_graph.add_edge(y[0],y[2],etype=[y[1]],ie=[(x[0],x[3],y[1],nx.get_edge_attributes(x[1],"ie")[(y[0],y[1])][1]
    ,nx.get_edge_attributes(x[1],"ie")[(y[1],y[2])][1])])
            else:
                cons_graph.get_edge_data(y[0], y[2])["ie"].append((x[0],x[3],y[1],nx.get_edge_attributes(x[1],"ie")[(y[0],y[1])][1]
    ,nx.get_edge_attributes(x[1],"ie")[(y[1],y[2])][1]))
                cons_graph.get_edge_data(y[0], y[2])["etype"].append(y[1])
            is_break = True

    analyzer = SentimentIntensityAnalyzer()
    
    rel_list = []
    for key,value in nx.get_edge_attributes(cons_graph,"ie").items():
        for y in value:
            temp_tags = y[4]["tags"]
            temp_tags = temp_tags + (len(y[1]) - len(y[4]["tags"])) * ["O"]
            try:
                sentence = " ".join(np.array(y[1])[np.array(temp_tags)!="O"])
            except BaseException as ex:
                raise ex
            vs = analyzer.polarity_scores(sentence)
            sent_id = y[0]
            time_frame = int(sent_id.split("-")[-1])
            rel_list.append((sent_id,time_frame,key[0],key[1],sentence,vs,y,y[4]["verb"]))
            rel_list.append((sent_id,time_frame,key[1],key[0],sentence,vs,y,y[4]["verb"]))

    cons_graph_vol[xvol] = {"cons_graph": cons_graph,"rel_list": pd.DataFrame(rel_list)}


# prepare dataframe for sentiment analysis and network relations
print("Preparing dataframe for network relations and sentiment dynamic")
new_frame = pd.DataFrame()

for xvol,val in tqdm.tqdm(cons_graph_vol.items()):
    rel_frame = val["rel_list"]
    rel_frame["sentiment"] = rel_frame[5].apply(lambda x: x["pos"] - x["neg"])
    rel_frame_group = rel_frame.groupby([1,2,3]).mean().reset_index()
        
    for x in rel_frame_group[[2,3]].groupby([2,3]).count().index:
        rel_frame_gg = rel_frame.groupby([2,3]).get_group(tuple(x))
        rel_frame_gg = rel_frame_gg.sort_values(1)
        rel_frame_gg["sentiment_rolling"] = rel_frame_gg[["sentiment"]].rolling(window=3,min_periods=1).mean()
        rel_frame_gg["delta_sentiment"] = abs(rel_frame_gg[["sentiment"]] - rel_frame_gg[["sentiment"]].shift(1))
        rel_frame_gg["delta_sum"] = rel_frame_gg.fillna(0).delta_sentiment.rolling(window=2).sum()
        rel_frame_gg["delta_time"] = rel_frame_gg[[1]] - rel_frame_gg[[1]].shift(1)
        rel_frame_gg["vol_id"] = xvol
        new_frame = new_frame.append(rel_frame_gg)

# rename columns
temp_columns = list(new_frame.columns)
temp_columns[0] = "volume_id"
temp_columns[1] = "sentence_id"
temp_columns[2] = "actor_a"
temp_columns[3] = "actor_b"
temp_columns[4] = "sentence_part"
temp_columns[5] = "sentiment"
temp_columns[6] = "tokens"
temp_columns[7] = "verbs"
new_frame.columns = temp_columns
new_frame = new_frame.reset_index(drop=True)

# save network and sentiment frame
new_frame.to_csv(outfile)
print("Finished, saved output file to {}".format(outfile))
