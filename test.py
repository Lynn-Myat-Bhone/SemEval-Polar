import openpyxl

# Read hashtags from a text file
with open("hash.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Split text into individual hashtags
hashtags = text.split()

# Create a new Excel workbook
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Hashtags"

# Write each hashtag into a separate row
for i, tag in enumerate(hashtags, start=1):
    ws.cell(row=i, column=1, value=tag)

# Save the workbook
wb.save("hashtags.xlsx")

print("Excel file 'hashtags.xlsx' has been created successfully!")
