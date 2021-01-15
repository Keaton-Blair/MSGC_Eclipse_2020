import pandas as pd
from io import StringIO
import numpy as np



filePath = ""

# Open and investigate the file
contents = ""

f = open(filePath, 'r')

for line in f:  # Iterate through file, line by line
    if line.rstrip() == "Profile Data:":

        contents = f.read()  # Read in rest of file, discarding header

        break
f.close()  # Need to close opened file



# Read in the data and perform cleaning

# Need to remove space so Virt. Temp reads as one column, not two
contents = contents.replace("Virt. Temp", "Virt.Temp")
# Break file apart into separate lines
contents = contents.split("\n")
contents.pop(1)  # Remove units so that we can read table
index = -1  # Used to look for footer
for i in range(0, len(contents)):  # Iterate through lines
    if contents[i].strip() == "Tropopauses:":
        index = i  # Record start of footer
if index >= 0:  # Remove footer, if found
    contents = contents[:index]
contents = "\n".join(contents)  # Reassemble string

# Read in the data
data = pd.read_csv(StringIO(contents), delim_whitespace=True)


# Find the end of usable (ascent) data
badRows = []  # Index, soon to contain any rows to be removed
for row in range(data.shape[0]):  # Iterate through rows of data
    # noinspection PyChainedComparisons
    if not str(data['Rs'].loc[row]).replace('.', '', 1).isdigit():  # Check for nonnumeric or negative rise rate
        badRows.append(row)
    # Check for stable or decreasing altitude (removes rise rate = 0)
    elif row > 0 and np.diff(data['Alt'])[row-1] <= 0:
        badRows.append(row)
    else:
        for col in range(data.shape[1]):  # Iterate through every cell in row
            if data.iloc[row, col] == 999999.0:  # This value appears to be GRAWMET's version of NA
                badRows.append(row)  # Remove row if 999999.0 is found
                break

if len(badRows) > 0:

    data = data.drop(data.index[badRows])  # Actually remove any necessary rows
data.reset_index(drop=True, inplace=True)  # Return data frame index to [0,1,2,...,nrow]


out = list(data['Rs'])
out = [float(x) for x in out]
print(round(np.mean(out), 4))