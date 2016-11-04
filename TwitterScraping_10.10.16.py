
# coding: utf-8

# In[1]:

import requests, lxml.html


# In[2]:

response = requests.get("https://twitter.com/Metrorailinfo")
doc = lxml.html.fromstring(response.content)


# In[3]:

for el in doc.cssselect("div.js-tweet-text-container"):
    print(el.text_content().strip())


# In[4]:

for i, el in enumerate(doc.cssselect("div.js-tweet-text-container")):
    print(i, el.text_content().strip())
    print("-------------------------------------")


# In[ ]:




# In[ ]:



