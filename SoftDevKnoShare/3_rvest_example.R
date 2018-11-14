# Retrieve expedited cases from Confluence
# https://cnfl.extge.co.uk/pages/viewpage.action?spaceKey=BTS&title=Rare+disease+programme+expedited+interpretation+requests

# Load libraries. Not necessary if you have tidyverse loaded
library(rvest); packageDescription ("rvest", fields = "Version") # "0.3.2" # Be caareful as loading rvest masks pluck!
library(httr); packageDescription ("httr", fields = "Version") # "1.3.1"

URL <- "https://cnfl.extge.co.uk/pages/viewpage.action?spaceKey=BTS&title=Rare+disease+programme+expedited+interpretation+requests"

# Identify yourself
# user.id: string containing your confluence userid (same with LDAP username)
# passwd: string containing your confluence password (same with LDAP password)
response_expedited = GET(URL, authenticate(user.id, passwd)) 
                                                            
expedited_raw <- read_html(response_expedited) # read html
expedited_list <- html_table(expedited_raw) # read table from html
expedited_cases <- expedited_list %>% as.data.frame.list() # convert list to a dataframe

