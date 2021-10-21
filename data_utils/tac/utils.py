
def get_section(section_label, section_sentence):
    """
    Arguments:
        section_label: all sections for a given label.
        section_sentence: all sections with boundaries sentences
        
    Outputs:
        id_section: id of section
        section: text of section
    """
    id_section = None
    section = None
    for s in section_label:
        if s.name == section_sentence.attrib['name']:
            id_section = s.id
            section = s.text
            break
    return id_section, section
	
def get_mentions(mentions, id_section):
    """
    Arguments:
        mentions: all mentions for given label
        id_section: id of section
        
    Output:
        unique_section_mentions: set of mentions for given id_section
        section_mentions: list of mentions for given section
    """
    section_mentions = []
    unique_section_mentions = set()
    for m in mentions:
        if m.section==id_section:
            section_mentions.append(m) 
                    
    #Return unique mentions
    for m in section_mentions:
        mention = (m.len, m.start, m.type, m.str, m.section)
        unique_section_mentions.add(mention)
    return unique_section_mentions, section_mentions
	
def get_relations(section_mentions, relations):
    """
    Arguments:
        section_mentions: all mentions for given section
        relations: all relations in label
    Output:
        unique_relations: set of relations given a section
    """
    unique_relations = set()
    for r in relations:
        arg1 = None
        arg2 = None
        for m in section_mentions:
            if m.id == r.arg1:
                arg1 = (m.len, m.start, m.type, m.str, m.section)
            elif m.id == r.arg2:
                arg2 = (m.len, m.start, m.type, m.str, m.section)
        unique_relations.add((arg1,r.type,arg2))
    return unique_relations

def get_mentions_relations_from_sentence(section_mentions, start, end, relations):
    set_ADE_mention = []
    set_ADE_relation = []
    for m in section_mentions:
        start_mention = m[1].split(',')
        if m[2] == 'AdverseReaction':
            modifier_set_id = set()
            for r in relations:
                if m == r[0]:
                    modifier_set_id.add((r[2][0], r[2][1], r[2][2], r[2][3], r[2][4], r[1]))
            mention = None           
            mstart = int(start_mention[0])                   
            if mstart in range(start, end):
                if len(modifier_set_id)==0:
                    mention = (m[0], m[1], m[2], m[3], m[4], 'no_rel')
                else:
                    mention = (m[0], m[1], m[2], m[3], m[4], 'rel')
                    set_ADE_relation.append((mention,modifier_set_id))
                set_ADE_mention.append(mention)
    return set_ADE_mention, set_ADE_relation
	
def return_overlapping_mentions_per_mention(start_m, end_m, modifier_set):
    contin_m = []
    discontin_m = []
    for m in modifier_set:
        start_mention = m[1].split(',')
        len_mention = m[0].split(',')       
        mstart = int(start_mention[0])                   
        mend = mstart + int(len_mention[0])
        
        if len(start_mention)==1:
            if mstart in range(start_m,end_m) or start_m in range(mstart,mend):
                contin_m.append(m)
        else:
            for b, e in zip(start_mention, len_mention):
                if int(b) in range(start_m,end_m) or start_m in range(int(b), int(b) + int(e)):
                    discontin_m.append(m)
    return contin_m, discontin_m