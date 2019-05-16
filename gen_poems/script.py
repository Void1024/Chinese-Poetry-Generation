from random import shuffle
poems_path_list = [
    ['poems_anmt.txt','ANMT'],['poems_ebpg.txt',"EBPG"],['poems_ppg.txt','PPG'],['poems_rnnpg.txt','RNNPG'],['poems_smt.txt','SMT']
]
def gen():
    poetry_list = []
    for path in poems_path_list:
        with open(path[0],'r',encoding='utf-8') as fin:
            for poetry in fin.readlines():
                poetry = poetry[:-2]
                poetry = path[1] + ' ' + poetry
                poetry_list.append(poetry)
    shuffle(poetry_list)
    with open('gen_poems.txt','w',encoding='utf-8') as fout:
        for p in poetry_list:
            fout.write(p + '\n')

if __name__ == '__main__':
    gen()
