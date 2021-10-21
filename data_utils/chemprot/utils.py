

def get_concepts(concepts, start_sent, end_sent):       
    attributes = set()
    drugs = set()
    concept_sent = set()
    for m in concepts:
        if m.start in range(start_sent,end_sent) and m.end in range(start_sent,end_sent):
            if m.ttype!='CHEMICAL':
                attributes.add(m)
            else:
                drugs.add(m)  
            concept_sent.add(m)
    return attributes, drugs, concept_sent
	
def get_relations_from_sentence(drugs, attributes, relations):
    chem_relation = []
    for d in drugs:
        att = set()
        for a in attributes:
            for r in relations:
                if d.tid==r.arg1.tid and a.tid==r.arg2.tid:
                    att.add((a, r.rtype)) 
                    break
                    
        #if len(att)>0:
        chem_relation.append((d, att))
    return chem_relation