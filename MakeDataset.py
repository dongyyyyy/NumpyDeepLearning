import numpy as np
import csv

def createRandomArray():
    result = np.random.randint(0,2,3,dtype=int)
    if(result[0]==0 and result[1]==0 and result[2] ==0):
        output = 1
    elif(result[0]==0 and result[1]==0 and result[2] == 1):
        output = 2
    elif(result[0]==0 and result[1]==1 and result[2] ==0):
        output = 3
    elif (result[0] == 0 and result[1] == 1 and result[2] == 1):
        output = 4
    elif (result[0] == 1 and result[1] == 0 and result[2] == 0):
        output = 5
    elif (result[0] == 1 and result[1] == 0 and result[2] == 1):
        output = 6
    elif (result[0] == 1 and result[1] == 1 and result[2] == 0):
        output = 7
    elif (result[0] == 1 and result[1] == 1 and result[2] == 1):
        output = 8
    noise = np.random.normal(0, 0.1, 3)
    result = result - noise
    return result , output

if __name__ == "__main__":
    saveData = []
    csvfile = open("TrainDataset.csv","w",newline="")
    csvwriter = csv.writer(csvfile)

    for i in range(20000):
        array,output = createRandomArray()
        output = np.reshape(output,(1))
        result = np.concatenate([array,output],axis=-1)
        csvwriter.writerow(result)

    csvfile.close()

    csvfile = open("TestDataset.csv", "w", newline="")
    csvwriter = csv.writer(csvfile)

    for i in range(1000):
        array,output = createRandomArray()
        output = np.reshape(output,(1))
        result = np.concatenate([array,output],axis=-1)
        csvwriter.writerow(result)

    csvfile.close()
