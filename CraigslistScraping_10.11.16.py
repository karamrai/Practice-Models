
# coding: utf-8

# In[1]:

import requests, lxml.html
import pandas as pd
from IPython.display import display, Image

pd.set_option('display.max_colwidth', 100)


# In[2]:

offset = 0
response = requests.get("http://washingtondc.craigslist.org/search/apa?s=%s" % offset)
doc = lxml.html.fromstring(response.content)

rows = []
for row in doc.cssselect("div.content p.row"):
    item_id = row.get('data-pid')
    repost_of = row.get('data-repost-of')
    link = "http://washingtondc.craigslist.org" + row.cssselect('a')[0].get('href')
    row = [item_id, link]
    rows.append(row)
df = pd.DataFrame(rows, columns=['item_id', 'link'])
df.head()


# In[3]:

response = requests.get("http://washingtondc.craigslist.org/doc/apa/5549206236.html")
doc = lxml.html.fromstring(response.content)

print(doc.cssselect("section.body h2.postingtitle span.postingtitletext")[0].text_content().strip())
print("--------------------------------------------------")

first_image_in_carousel = doc.cssselect("section.body section.userbody div.slide.visible img")
if first_image_in_carousel:
    img_url = first_image_in_carousel[0].get('src')
    display(Image(url=img_url))

print("--------------------------------------------------")
    
print(doc.cssselect("#postingbody")[0].text_content())

