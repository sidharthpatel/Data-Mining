#!/usr/bin/python

from subprocess import Popen, PIPE

files = ["twitter", "wikipedia"]
# targets = [("names_male", "names_female"), ("gender_m",
#                                             "gender_f"), ("names_africa", "names_europe")]
# attributes = [("art", "science"), ("career", "family"), ("insects", "flowers"),
#               ("pleasant", "unpleasant"), ("positive-words", "negative-words")]

targets = [("names_male", "names_female"), ("gender_m",
                                            "gender_f")]
attributes = [("computers_and_maths", "biology")]

for file in files:
    for target in targets:
        for attrs in attributes:
            process = Popen(["./weatTest.py", file, target[0],
                            target[1], attrs[0], attrs[1]], stdout=PIPE)
            (output, err) = process.communicate()
            dupOutput = str(output).split("Effect size: ")[
                1].split("\n\n")[0]
            print("File: %s, targets: (%s, %s), attributes: (%s, %s), Effect size: %s." % (
                file, target[0], target[1], attrs[0], attrs[1], str(dupOutput.strip()[:-5]) if dupOutput != None else "0.0"))
            exit_code = process.wait()
