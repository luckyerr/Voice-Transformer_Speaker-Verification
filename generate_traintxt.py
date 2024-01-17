import os

for dirpath, dirnames, filenames in os.walk('path/data/'):
        a = dirpath.split('data/')[-1]
        for filename in filenames:
            print(a + " " + dirpath.split('data/')[-1] + str("/") + filename)
