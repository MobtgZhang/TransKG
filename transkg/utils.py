import os
import random

from .data import Dictionary

def create_ents_rels(files_list,save_ent_file,save_rel_file):
    ent_dict = Dictionary()
    rel_dict = Dictionary()
    for filename  in files_list:
        with open(filename,mode="r",encoding="utf-8") as rfp:
            for line in rfp:
                words = line.strip().split("\t")
                head_ent,relation,tail_ent = words
                ent_dict.add(head_ent)
                ent_dict.add(tail_ent)
                rel_dict.add(relation)
    ent_dict.save(save_ent_file)
    rel_dict.save(save_rel_file)
def create_pos_neg_ids(load_filename,save_dir,ent_dict,rel_dict,tag,generate_negs = False):
    pos_data_list = []
    if generate_negs:
        tp_rel_dict = {}
    with open(load_filename,mode="r",encoding="utf-8") as rfp:
        save_pos_filename = os.path.join(save_dir,"%s_ids.txt"%tag)
        with open(save_pos_filename,mode="w",encoding="utf-8") as wfp:
            for line in rfp:
                words = line.strip().split("\t")
                w_triple = [ent_dict[words[0]],rel_dict[words[1]],ent_dict[words[2]]]
                str_line = "%d\t%d\t%d\n"%(w_triple[0],w_triple[1],w_triple[2])
                pos_data_list.append(w_triple)
                wfp.write(str_line)
                if generate_negs:
                    rel = w_triple[1]
                    if rel not in tp_rel_dict:
                        tp_rel_dict[rel] = {
                            "h":set(),
                            "t":set()
                        }
                    tp_rel_dict[rel]["h"].add(w_triple[0])
                    tp_rel_dict[rel]["t"].add(w_triple[2])    
    if generate_negs:
        # generate negative triples
        save_neg_filename = os.path.join(save_dir,"%s_neg_ids.txt"%tag)
        with open(save_neg_filename,mode="w",encoding="utf-8") as wfp:
            for item in pos_data_list:
                rel = item[1]
                n_h = len(tp_rel_dict[rel]["h"])
                n_t = len(tp_rel_dict[rel]["t"])
                tph = n_t/n_h
                hpt = n_h/n_t
                if tph>hpt:
                    # replace the head entity
                    new_tail = item[2]
                    while True:
                        new_head = random.choice(range(len(ent_dict)))
                        if new_head!=item[0]:
                            break
                else:
                    # replace the tail entity
                    new_head = item[0]
                    while True:
                        new_tail = random.choice(range(len(ent_dict)))
                        if new_tail!=item[2]:
                            break
                str_line = "%d\t%d\t%d\n"%(new_head,rel,new_tail)
                wfp.write(str_line)

