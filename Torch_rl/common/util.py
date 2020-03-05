import csv
def csv_record(data,path):
    with open(path+"record.csv", "a+") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)