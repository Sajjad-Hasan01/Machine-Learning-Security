df = open('../data/dataset.csv', 'r')
print(df.readable())
print(df.readline())
# print(df.read())
print(df.readlines()[0])
# for line in df.readlines():
#     print(line)

df.close()
