# Load libraries
library(data.table)
library(tidyverse)
library(janitor)

# Download upload report
upload_report <- as.tibble(fread("https://upload-reports.gel.zone/upload_report.latest.txt"))

# Select genome and delivery version
upload_report_trunc <- upload_report %>% select(Platekey, `Delivery Version`)

# or simply in one line
upload_report_trunc <- fread("https://upload-reports.gel.zone/upload_report.latest.txt", select = c("Platekey", "Delivery Version"))

# Use tabyl to tabulate number of genomes per delivery
upload_report %>% tabyl(`Delivery Version`)

# Use tabyl to tabulate genomes status per Delivery Version
upload_report %>% tabyl(`Delivery Version`, Status)