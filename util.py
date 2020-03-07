

def list_to_dict(lines):
    ret_dict = {x:[] for x in lines[0]}
    for line in lines[1:]:
        for i,j in zip(list(ret_dict.keys()), line):
            ret_dict[i].append(j)
    
    return ret_dict