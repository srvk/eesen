import sys

word_map = sys.argv[2]
g = open(word_map, "rb")
lines = g.readlines()
word_map_dict = {}
for line in lines:
    line = line.strip()
    words = line.split(" ")
    word_map_dict[words[1]] = words[0]


dict_path = sys.argv[1]
f = open(dict_path,"rb")
lines = f.readlines()
for line in lines:
    words = line.strip().split(" ")
    actual_line = []
    for word in words[1:]:
       if word in word_map_dict:
           actual_line.append(word_map_dict[word])
       else:
           print word
    actual_line.append("("+words[0]+")")
    print " ".join(actual_line)
#print word_map_dict
