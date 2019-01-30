from pyark.cva_client import CvaClient

from protocols.reports_5_0_0 import Tier, Program, Assembly
import getpass
from itertools import islice
cva = CvaClient('https://bio-prod-cva.gel.zone', user='dpolychronopoulos', password=getpass.getpass())

# report_events = client.report_events().get_report_events(panelName='cakut')
# [re.caseId for re in islice(report_events, 5)]

# report_events = client.report_events().get_report_events(caseId='2915')
# sum(1 for _ in report_events)

# report_events = client.report_events().get_report_events(caseId='2915')
# tiers = set([re.reportEvent.tier for re in report_events])
tiers

# client.cases().get_variants_by_panel('cakut', hasTiered=True, tiers=["TIER1"])

report_events = cva.report_events()
entities = cva.entities()
cases = cva.cases()
variants = cva.variants()

from pyark.cva_client import CvaClient
from protocols.reports_6_0_0 import Program, Assembly
import pandas as pd
import itertools
pd.options.display.float_format = '{:.3f}'.format
import matplotlib.pyplot as plt
import getpass
# %matplotlib inline

cva_url = "https://bio-prod-cva.gel.zone"
gel_user = "dpolychronopoulos"
gel_password = getpass.getpass()
cva = CvaClient(cva_url, user=gel_user, password=gel_password)

cases_client = cva.cases()
overall_summary = cases_client.get_summary()
print("Output type is {}".format(type(overall_summary)))
overall_summary.keys()

overall_summary = cases_client.get_summary(as_data_frame=True)
print("Output type is {}".format(type(overall_summary)))
overall_summary.columns

cases_summary_rd_38 = cases_client.get_summary(
    program=Program.rare_disease, assembly=Assembly.GRCh38, as_data_frame=True)
cases_summary_rd_38.index

cases_summary_rd_38[["avgParticipants", "avgSamples"]].head()

cases_summary_rd_37 = cases_client.get_summary(
    program=Program.rare_disease, assembly=Assembly.GRCh37, as_data_frame=True)
cases_summary_ca_37 = cases_client.get_summary(
    program=Program.cancer, assembly=Assembly.GRCh37, as_data_frame=True)
cases_summary_ca_38 = cases_client.get_summary(
    program=Program.cancer, assembly=Assembly.GRCh38, as_data_frame=True)
cases_summary = pd.concat([cases_summary_rd_38, cases_summary_rd_37, cases_summary_ca_38, cases_summary_ca_37])
cases_summary.transpose().head()

cases_summary.xs(Assembly.GRCh38, level='assembly', axis=0)['avgParticipants']