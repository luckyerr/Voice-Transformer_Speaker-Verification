import os

for dirpath, dirnames, filenames in os.walk('C:/Users/why/Desktop/aaa/'):
        a = dirpath.split('aaa/')[-1]
        for filename in filenames:
            print(a + " " + dirpath.split('aaa/')[-1] + str("/") + filename)
