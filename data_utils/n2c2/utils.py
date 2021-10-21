

def get_concepts(concepts, start_sent, end_sent):       
    attributes = set()
    drugs = set()
    concept_sent = set()
    for key1, m in zip(concepts.keys(), concepts.values()):
        if m.start in range(start_sent,end_sent) and m.end in range(start_sent,end_sent):
            if m.ttype!='Drug':
                attributes.add(m)
            else:
                drugs.add(m)  
            concept_sent.add(m)
    return attributes, drugs, concept_sent
	
def get_relations_from_sentence(drugs, attributes, relations):
    ADE_relation = []
    set_drugs_rel = []
    set_drugs_no_rel = []
    for d in drugs:
        att = set()
        for a in attributes:
            for key, r in zip(relations.keys(), relations.values()):
                if d.tid==r.arg2.tid and a.tid==r.arg1.tid:
                    att.add(a) 
                    break
                    
        if len(att)>0:
            ADE_relation.append((d, att))
            set_drugs_rel.append(d)
        else:
            set_drugs_no_rel.append(d)
    return ADE_relation, set_drugs_rel, set_drugs_no_rel
