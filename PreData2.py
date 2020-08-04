import pandas as pd
import math

NUM_OF_FEATURE = 16

#prepare Data
def EventToDF(Data:pd.DataFrame):
    event1 = [0 for i in range(NUM_OF_FEATURE)]
    #event2 = [0 for i in range(NUM_OF_FEATURE)]
    for raw in Data.iterrows():
        if raw[1]["side"] == 1:
            event1[raw[1]["event_type"]] += 1
            if math.isnan(raw[1]["event_type2"]) == False:
                event1[int(raw[1]["event_type2"])] += 1
        else:
            event1[raw[1]["event_type"]] -= 1
            if math.isnan(raw[1]["event_type2"]) == False:
                event1[int(raw[1]["event_type2"])] -= 1
    #print("team 1 have event:\n{}\nTeam 2 have event:\n{}".format(event1,event2))
    return event1

def Makeheader(Data):
    header = []
    header0 = [item for item in Data.head(0)]
    header1 = ["num of events type " + str(i) + " for size 1" for i in range(NUM_OF_FEATURE)]
    #header2 = ["num of events type " + str(i) + " for size 2" for i in range(NUM_OF_FEATURE)]
    header = header0 + header1
    return header

def main():
    # read ginf data
    Gintdata = pd.read_csv("ginf.csv")
    # Making header
    header = Makeheader(Gintdata)
    # read event data
    Eventdata = pd.read_csv("events.csv")
    #get the fauture from events
    raws = []
    #IDs = list(Gintdata["id_odsp"])
    for raw in Gintdata.values:
        Eventdatatemp = Eventdata[(Eventdata["id_odsp"] == raw[0])] # raw[0] == ID
        raws.append(list(raw)+EventToDF(Eventdatatemp))

    res = pd.DataFrame(data=raws, columns=header)
    #res = res.drop(columns=["num of events type 0 for size 1","num of events type 0 for size 2", "id_odsp"])
    res.to_csv("final data 2.csv")


if __name__ == "__main__":
    main()