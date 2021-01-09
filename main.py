# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import LogGeneration
import RulesExtraction
import Other

LOGS_FOLDER="data/logs/"

to_describe = {
    #"sepsis_constr": "sepsis_constraint_tagged.xes", --> missing
    #"sepsis_seq": "sepsis_sequence_tagged.xes",      --> missing
    "sepsis_er": "sepsis_tagged_er.xes",
    "SYNTH_MR2": "synth_mr2_tagged.xes",
    "SYNTH_TR": "synth_tr_tagged.xes",
    #"BPI2011_M16": "EnglishBPIChallenge2011_tagged_m16.xes",
    #"BPI2011_T101": "EnglishBPIChallenge2011_tagged_t101.xes",
    #"BPI2011_M13": "EnglishBPIChallenge2011_tagged_M13.xes",
    #"BPI2011_CC": "EnglishBPIChallenge2011_tagged_cc.xes",
    #"XRAY": "merged_xray.xes"
}

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    LogGeneration.write_xeses(LOGS_FOLDER)
    for key,value in to_describe.items():
        Other.describe(key, LOGS_FOLDER, value)
    #RulesExtraction.get_dtrules()               ## Requires: snapshots
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
