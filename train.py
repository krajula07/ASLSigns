import pandas as pd
import numpy as np
import glob

def main():
    buy_path = r'D:\Graduation_Courses\Mobile Computing\Assignment_2\CSV\buy' # use your path
    
    buy_all_files = glob.glob(buy_path + "/*.csv")

    li = []

    for filename in buy_all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    buy_frame = pd.concat(li, axis=0, ignore_index=True)
    
    
   
    print(len(frame))

if __name__ == "__main__":
    main()