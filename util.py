

def list_to_dict(lines):
    ret_dict = {x:[] for x in lines[0]}
    for line in lines[1:]:
        for i,j in zip(list(ret_dict.keys()), line):
            ret_dict[i].append(j)
    
    return ret_dict

def write_va(txt_path, v, a):
    f = open(txt_path, 'w')
    f.write('{} {}\n'.format(v, a))
    f.close()