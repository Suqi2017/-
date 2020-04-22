
def create_dict():
    dic = {}
    num = 0
    with open('passage.txt','rt',encoding='UTF-8') as f:     
        for line in f:
            string = '' #存储符号内的字
            index = [] #存储符号索引
            if line.__len__() > 15:
                for i in range(15):
                    if line[i] == '【' or line[i] == '】':
                        index.append(i)
                if index.__len__() == 2:
                    string = line[index[0]+1:index[1]]
                    if string.__len__() > 5:
                        continue
                    if dic.get(string[0]):
                        dic[string[0]].append(string)
                    else:
                        dic[string[0]] = [string]
    return dic
    
