# importing packages
import requests
import csv
import os

## Let's build the csv file first and add the column names
## Create a new csv file to save the headlines
filename="data.csv"
MyFILE=open(filename,"w")  # "a"  for append   "r" for read
## with open
### Place the column names in - write to the first row
WriteThis="OBJECTID,STATION_NUMBER,STATION_TEXT,TRAFFIC_COUNT,TRAFFIC_YEAR_COUNTED,BIKE_COUNT,BIKE_YEAR_COUNTED,STATUS,STREET_NAME,CHRIS_NUMB,PAVETYPE,FUNCTIONAL_CLASS,TrafficStationID \n"
MyFILE.write(WriteThis)
MyFILE.close()


URL="https://maps.bouldercounty.org/arcgis/rest/services/Transportation/Traffic_Stations/MapServer/0/query?outFields=*&where=1%3D1&f=geojson"

response=requests.get(URL)
print(response)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    jsontxt = response.json()
    print(jsontxt)
else:
    print('Error:', response.status_code)



# Define the CSV file path
csv_file_path = 'data.csv'

# Extract specific fields from the JSON data
data_to_extract = []
for feature in jsontxt['features']:
    extracted_data = [
        feature['properties']['OBJECTID'],
        feature['properties']['STATION_NUMBER'],
        feature['properties']['STATION_TEXT'],
        feature['properties']['TRAFFIC_COUNT'],
        feature['properties']['TRAFFIC_YEAR_COUNTED'],
        feature['properties']['BIKE_COUNT'],
        feature['properties']['BIKE_YEAR_COUNTED'],
        feature['properties']['STATUS'],
        feature['properties']['STREET_NAME'],
        feature['properties']['CHRIS_NUMB'],
        feature['properties']['PAVETYPE'],
        feature['properties']['FUNCTIONAL_CLASS'],
        feature['properties']['TrafficStationID']
    ]
    data_to_extract.append(extracted_data)

# Open the CSV file in append mode and write data row by row
with open(csv_file_path, 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Write a header row if the file is empty
    if os.path.getsize(csv_file_path) == 0:
        header_row = [
            'OBJECTID', 'STATION_NUMBER', 'STATION_TEXT',
            'TRAFFIC_COUNT', 'TRAFFIC_YEAR_COUNTED',
            'BIKE_COUNT', 'BIKE_YEAR_COUNTED',
            'STATUS', 'STREET_NAME', 'CHRIS_NUMB',
            'PAVETYPE', 'FUNCTIONAL_CLASS', 'TrafficStationID'
        ]
        csv_writer.writerow(header_row)
    
    # Write the extracted data row by row
    for row in data_to_extract:
        csv_writer.writerow(row)
