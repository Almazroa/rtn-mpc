'''
You are given a Google Doc like this one that contains a list of Unicode characters and their positions in a 2D grid. 
Your task is to write a function that takes in the URL for such a Google Doc as an argument, 
retrieves and parses the data in the document, and prints the grid of characters. 
When printed in a fixed-width font, the characters in the grid will form a graphic showing a sequence of uppercase letters, which is the secret message.

The document specifies the Unicode characters in the grid, along with the x- and y-coordinates of each character.

The minimum possible value of these coordinates is 0. There is no maximum possible value, so the grid can be arbitrarily large.

Any positions in the grid that do not have a specified character should be filled with a space character.

You can assume the document will always have the same format as the example document linked above.

both are zero indexed. x grows to the right and y grows downwards.

You may write helper functions, but there should be one function that:

1. Takes in one argument, which is a string containing the URL for the Google Doc with the input data, AND

2. When called, prints the grid of characters specified by the input data, displaying a graphic of correctly oriented uppercase letters.
'''
import requests
from bs4 import BeautifulSoup

def secret_message(url):

    
    response = requests.get(url) # get doc data
    if response.status_code != 200:
        raise Exception("Failed to retrieve data from the URL.")
    
    page = response.text

    soup = BeautifulSoup(page, 'html.parser') # prasing html
    table = soup.find('table')
    if not table:
        raise Exception("No table found in the document.")
    
    rows = table.find_all('tr') # dissecting table 
    if not rows or len(rows) < 2:   # no rows or only header row
        raise Exception("No rows found in the table.")
    
    # order should be x-coordinate | Character | y-coordinate
    x_idx = 0
    ch_idx = 1
    y_idx = 2

    data = []
    # for each row, get the x, y coordinates and character
    for row in rows[1:]:
        cells = row.find_all('td')
        try:
            x = int(cells[x_idx].get_text(strip=True)) # text at x_idx = x-coordinate
            y = int(cells[y_idx].get_text(strip=True))
        except ValueError:
            raise Exception("x or y coordinate is not an integer.")
        ch = cells[ch_idx].get_text(strip=True)
        data.append((x, y, ch))
    
    # find grid dimensions
    max_x = max(item[0] for item in data)
    max_y = max(item[1] for item in data)

    # create empty grid filled with spaces
    grid = [[' ' for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    # fill grid with data found
    for x, y, ch in data:
        grid[y][x] = ch # y is the row, x is the column -> x grows to the right and y grows downwards

    # output message to screen
    for row in grid:
        print(''.join(row))


secret_message("https://docs.google.com/document/d/e/2PACX-1vQGUck9HIFCyezsrBSnmENk5ieJuYwpt7YHYEzeNJkIb9OSDdx-ov2nRNReKQyey-cwJOoEKUhLmN9z/pub")