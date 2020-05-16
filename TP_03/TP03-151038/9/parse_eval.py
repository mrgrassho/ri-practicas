from bs4 import BeautifulSoup as soup
from re import findall
import csv

PATTERNS = '[0-9]+\.[0-9]+|[0-9]+(?:\-[0-9]+)+|[A-Z][a-z]+(?:(?:\s*|\s*\n\s*)[A-Z][a-z]+)*|[0-9]+|[a-zA-Z]+'

with open("CISI.QRY", "r") as myfile:
    x = soup(myfile)
    wl = open("stopword-list.txt", "r")
    stopword_list = wl.readlines()
    
    for top in x.find_all('top'):
        content = top.title.string
        set_elements = [se for se in {e.lower() for e in findall(PATTERNS, content)} if se not in stopword_list]
        top.title.string = " ".join(set_elements)
    
    with open("CISI.QRY.UNIQUE", "w") as myfile:
        myfile.write(str(x))
