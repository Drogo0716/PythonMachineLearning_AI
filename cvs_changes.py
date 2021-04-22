import re
import csv
import datetime
'''
def getage(now, dob):
    years = now.year - dob.year
    months = now.month - dob.month
    if now.day < dob.day:
        months -= 1
        while months < 0:
            months += 12
            years -= 1
    return '%s'% (years)
'''
with open('HeartFailure.csv','r') as fin, open('New_HeartFailure.csv', 'w', newline ='') as fout:
    reader = csv.DictReader(fin)
    writer_clinics = csv.DictWriter(fout, reader.fieldnames, dialect="excel")
    writer_clinics.writeheader()

    for data in reader:
        #today = datetime.date.today()
        if (data["DiagNum"] == '0'):
            data["DiagNum"] = '0'
            writer_clinics.writerow(data)
        else:
            data["DiagNum"] = '1'
            writer_clinics.writerow(data)